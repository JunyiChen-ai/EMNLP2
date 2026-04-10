#!/usr/bin/env python3
"""Single-dataset HVGuard checkpoint fine-tuning with threshold calibration."""

import argparse
import json
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

BASE = "/data/jehc223/EMNLP2/baseline/HVGuard"
CFG = {
    "HateMM": {
        "emb_dir": f"{BASE}/embeddings/HateMM",
        "label": f"{BASE}/datasets/HateMM/annotation(new).json",
        "splits": "/data/jehc223/HateMM/splits",
        "ckpt": f"{BASE}/models/HateMM_English_2.pth",
        "label_map": {"Non Hate": 0, "Hate": 1},
    },
    "MultiHateClip_CN": {
        "emb_dir": f"{BASE}/embeddings/Multihateclip/Chinese",
        "label": f"{BASE}/datasets/Multihateclip/Chinese/annotation(new).json",
        "splits": "/data/jehc223/Multihateclip/Chinese/splits",
        "ckpt": f"{BASE}/models/Multihateclip_Chinese_2.pth",
        "label_map": {"Normal": 0, "Offensive": 1, "Hateful": 1},
    },
    "MultiHateClip_EN": {
        "emb_dir": f"{BASE}/embeddings/Multihateclip/English",
        "label": f"{BASE}/datasets/Multihateclip/English/annotation(new).json",
        "splits": "/data/jehc223/Multihateclip/English/splits",
        "ckpt": f"{BASE}/models/Multihateclip_English_2.pth",
        "label_map": {"Normal": 0, "Offensive": 1, "Hateful": 1},
    },
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FeatureDataset(Dataset):
    def __init__(self, ids, labels, label_map, feats):
        self.ids = ids
        self.labels = labels
        self.label_map = label_map
        self.feats = feats

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        vid = self.ids[idx]
        zero = torch.zeros(768)
        x = torch.cat(
            [
                self.feats["text"].get(vid, zero),
                self.feats["audio"].get(vid, zero),
                self.feats["frame"].get(vid, zero),
                self.feats["mix"].get(vid, zero),
            ]
        )
        y = self.label_map[self.labels[vid]]
        return x, torch.tensor(y, dtype=torch.long)


class MoE(nn.Module):
    def __init__(self, input_dim, num_experts=8, expert_dim=128, output_dim=128):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, expert_dim),
                    nn.ReLU(),
                    nn.Linear(expert_dim, output_dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.gate = nn.Sequential(nn.Linear(input_dim, num_experts), nn.Softmax(dim=-1))
        self.gate_dropout = nn.Dropout(0.1)

    def forward(self, x):
        g = self.gate_dropout(self.gate(x))
        ex = torch.stack([e(x) for e in self.experts], dim=1)
        return torch.sum(g.unsqueeze(-1) * ex, dim=1)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, num_classes=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def load_split(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def class_weights(y, n_classes=2):
    cnt = Counter(y)
    total = len(y)
    w = torch.ones(n_classes, dtype=torch.float)
    present = [c for c in range(n_classes) if cnt.get(c, 0) > 0]
    for c in present:
        w[c] = total / (len(present) * cnt[c])
    return w


def pick_threshold(probs, labels):
    best_thr = 0.5
    best_acc = -1
    for thr in np.linspace(0.2, 0.8, 61):
        pred = (probs >= thr).astype(int)
        acc = accuracy_score(labels, pred)
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)
    return best_thr, float(best_acc)


def evaluate_probs(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            p = torch.softmax(logits, dim=1)[:, 1]
            probs.extend(p.cpu().numpy())
            labels.extend(y.numpy())
    return np.array(probs), np.array(labels)


def run(args):
    set_seed(args.seed)
    cfg = CFG[args.dataset]

    with open(cfg["label"], "r", encoding="utf-8") as f:
        raw = json.load(f)
    labels = {d["Video_ID"]: d["Label"] for d in raw if d["Label"] in cfg["label_map"]}

    splits = {
        "train": [v for v in load_split(os.path.join(cfg["splits"], "train.csv")) if v in labels],
        "valid": [v for v in load_split(os.path.join(cfg["splits"], "valid.csv")) if v in labels],
        "test": [v for v in load_split(os.path.join(cfg["splits"], "test.csv")) if v in labels],
    }

    feats = {
        "text": torch.load(os.path.join(cfg["emb_dir"], "text_features.pth"), map_location="cpu", weights_only=True),
        "audio": torch.load(os.path.join(cfg["emb_dir"], "audio_features.pth"), map_location="cpu", weights_only=True),
        "frame": torch.load(os.path.join(cfg["emb_dir"], "frame_features.pth"), map_location="cpu", weights_only=True),
        "mix": torch.load(os.path.join(cfg["emb_dir"], "MLLM_rationale_features.pth"), map_location="cpu", weights_only=True),
    }

    train_ds = FeatureDataset(splits["train"], labels, cfg["label_map"], feats)
    valid_ds = FeatureDataset(splits["valid"], labels, cfg["label_map"], feats)
    test_ds = FeatureDataset(splits["test"], labels, cfg["label_map"], feats)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False)

    model = nn.Sequential(
        MoE(input_dim=3072, num_experts=8, expert_dim=128, output_dim=128),
        MLPClassifier(input_dim=128, hidden_dim=64, num_classes=2),
    )

    if args.init_from_pretrained:
        state = torch.load(cfg["ckpt"], map_location="cpu")
        model.load_state_dict(state, strict=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    y_train = [cfg["label_map"][labels[v]] for v in splits["train"]]
    criterion = nn.CrossEntropyLoss(weight=class_weights(y_train, 2).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_state = None
    best_val_acc = 0.0
    best_thr = 0.5
    patience_left = args.patience

    for ep in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        val_probs, val_true = evaluate_probs(model, valid_loader, device)
        thr, val_acc = pick_threshold(val_probs, val_true)
        print(f"epoch={ep+1} val_acc={val_acc:.4f} thr={thr:.2f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            best_thr = float(thr)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    test_probs, test_true = evaluate_probs(model, test_loader, device)
    test_pred = (test_probs >= best_thr).astype(int)

    acc = accuracy_score(test_true, test_pred)
    f1 = f1_score(test_true, test_pred, average="macro")

    out = {
        "dataset": args.dataset,
        "seed": args.seed,
        "init_from_pretrained": bool(args.init_from_pretrained),
        "best_val_acc": best_val_acc * 100,
        "best_threshold": best_thr,
        "test_acc": acc * 100,
        "test_f1_macro": f1 * 100,
        "test_size": int(len(test_true)),
    }

    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(CFG.keys()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--init_from_pretrained", action="store_true")
    p.add_argument("--output_json", default="")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
