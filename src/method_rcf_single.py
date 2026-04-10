#!/usr/bin/env python3
"""RCF: Reliability-aware Counterfactual Fine-tuning (single dataset only).

Method idea:
- Base model: HVGuard MoE+MLP architecture (single model, no ensemble).
- Training objective:
  1) Supervised CE on full features.
  2) Reliability-weighted consistency between full prediction and counterfactual
     prediction where fragile modalities are masked.

This keeps strict constraints:
- single dataset train/eval only
- no cross-dataset training
- no ensemble
"""

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
        "scores": "/data/jehc223/EMNLP2/results/mllm/HateMM/mllm_scores.json",
        "mllm_results": "/data/jehc223/EMNLP2/results/mllm/HateMM/mllm_results.json",
    },
    "MultiHateClip_CN": {
        "emb_dir": f"{BASE}/embeddings/Multihateclip/Chinese",
        "label": f"{BASE}/datasets/Multihateclip/Chinese/annotation(new).json",
        "splits": "/data/jehc223/Multihateclip/Chinese/splits",
        "ckpt": f"{BASE}/models/Multihateclip_Chinese_2.pth",
        "label_map": {"Normal": 0, "Offensive": 1, "Hateful": 1},
        "scores": "/data/jehc223/EMNLP2/results/mllm/MultiHateClip_CN/mllm_scores.json",
        "mllm_results": "/data/jehc223/EMNLP2/results/mllm/MultiHateClip_CN/mllm_results.json",
    },
    "MultiHateClip_EN": {
        "emb_dir": f"{BASE}/embeddings/Multihateclip/English",
        "label": f"{BASE}/datasets/Multihateclip/English/annotation(new).json",
        "splits": "/data/jehc223/Multihateclip/English/splits",
        "ckpt": f"{BASE}/models/Multihateclip_English_2.pth",
        "label_map": {"Normal": 0, "Offensive": 1, "Hateful": 1},
        "scores": "/data/jehc223/EMNLP2/results/mllm/MultiHateClip_EN/mllm_scores.json",
        "mllm_results": "/data/jehc223/EMNLP2/results/mllm/MultiHateClip_EN/mllm_results.json",
    },
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FeatureDataset(Dataset):
    def __init__(self, ids, labels, label_map, feats, reliability):
        self.ids = ids
        self.labels = labels
        self.label_map = label_map
        self.feats = feats
        self.reliability = reliability

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        vid = self.ids[idx]
        zero = torch.zeros(768)
        t = self.feats["text"].get(vid, zero)
        a = self.feats["audio"].get(vid, zero)
        f = self.feats["frame"].get(vid, zero)
        m = self.feats["mix"].get(vid, zero)
        x = torch.cat([t, a, f, m])
        y = self.label_map[self.labels[vid]]
        rel = self.reliability.get(vid, 0.5)
        return x, torch.tensor(y, dtype=torch.long), torch.tensor(rel, dtype=torch.float)


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
    best_acc = -1.0
    for thr in np.linspace(0.2, 0.8, 61):
        pred = (probs >= thr).astype(int)
        acc = accuracy_score(labels, pred)
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)
    return best_thr, float(best_acc)


def eval_probs(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            logits = model(x)
            p = torch.softmax(logits, dim=1)[:, 1]
            probs.extend(p.cpu().numpy())
            labels.extend(y.numpy())
    return np.array(probs), np.array(labels)


def load_reliability(cfg, vids):
    scores = {}
    if os.path.exists(cfg["scores"]):
        with open(cfg["scores"], "r", encoding="utf-8") as f:
            scores = json.load(f)
    mllm_results = {}
    if os.path.exists(cfg["mllm_results"]):
        with open(cfg["mllm_results"], "r", encoding="utf-8") as f:
            mllm_results = json.load(f)

    rel = {}
    for vid in vids:
        sc = scores.get(vid, {}).get("scores", {})
        conf = float(sc.get("confidence", 5)) / 10.0
        vals = [
            float(sc.get("hate_speech_score", 5)),
            float(sc.get("visual_hate_score", 5)),
            float(sc.get("cross_modal_score", 5)),
            float(sc.get("implicit_hate_score", 5)),
            float(sc.get("overall_hate_score", 5)),
        ]
        disp = float(np.std(vals)) / 10.0
        analysis = mllm_results.get(vid, {}).get("analysis", "")
        text_only = 1.0 if analysis.startswith("[TEXT") else 0.0

        # higher value means less reliable multimodal evidence
        rel_bad = 0.45 * (1.0 - conf) + 0.35 * disp + 0.20 * text_only
        rel[vid] = float(np.clip(rel_bad, 0.0, 1.0))
    return rel


def make_counterfactual(x, rel_bad):
    # x: [B, 3072] = [text, audio, frame, mix]
    x_cf = x.clone()
    bsz = x.size(0)
    rel_bad = rel_bad.view(bsz, 1)

    # mask audio+frame more when evidence is unreliable
    x_cf[:, 768:1536] = x_cf[:, 768:1536] * (1.0 - rel_bad)
    x_cf[:, 1536:2304] = x_cf[:, 1536:2304] * (1.0 - rel_bad)
    return x_cf


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

    all_vids = list(set(splits["train"] + splits["valid"] + splits["test"]))
    reliability = load_reliability(cfg, all_vids)

    feats = {
        "text": torch.load(os.path.join(cfg["emb_dir"], "text_features.pth"), map_location="cpu", weights_only=True),
        "audio": torch.load(os.path.join(cfg["emb_dir"], "audio_features.pth"), map_location="cpu", weights_only=True),
        "frame": torch.load(os.path.join(cfg["emb_dir"], "frame_features.pth"), map_location="cpu", weights_only=True),
        "mix": torch.load(os.path.join(cfg["emb_dir"], "MLLM_rationale_features.pth"), map_location="cpu", weights_only=True),
    }

    train_ds = FeatureDataset(splits["train"], labels, cfg["label_map"], feats, reliability)
    valid_ds = FeatureDataset(splits["valid"], labels, cfg["label_map"], feats, reliability)
    test_ds = FeatureDataset(splits["test"], labels, cfg["label_map"], feats, reliability)

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
    ce = nn.CrossEntropyLoss(weight=class_weights(y_train, 2).to(device))
    kl = nn.KLDivLoss(reduction="none")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_state = None
    best_val_acc = 0.0
    best_thr = 0.5
    wait = args.patience

    for ep in range(args.epochs):
        model.train()
        running = 0.0
        for x, y, rel_bad in train_loader:
            x = x.to(device)
            y = y.to(device)
            rel_bad = rel_bad.to(device)

            logits_full = model(x)
            loss_ce = ce(logits_full, y)

            x_cf = make_counterfactual(x, rel_bad)
            logits_cf = model(x_cf)

            p_full = torch.softmax(logits_full, dim=1)
            p_cf = torch.softmax(logits_cf, dim=1)
            log_p_full = torch.log(torch.clamp(p_full, min=1e-8))

            k = kl(log_p_full, p_cf).sum(dim=1)
            loss_cons = (rel_bad * k).mean()

            loss = loss_ce + args.lambda_consistency * loss_cons
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())

        v_probs, v_true = eval_probs(model, valid_loader, device)
        thr, v_acc = pick_threshold(v_probs, v_true)
        print(
            f"epoch={ep+1} train_loss={running/max(len(train_loader),1):.4f} val_acc={v_acc:.4f} thr={thr:.2f}",
            flush=True,
        )

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_thr = thr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = args.patience
        else:
            wait -= 1
            if wait <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    t_probs, t_true = eval_probs(model, test_loader, device)
    t_pred = (t_probs >= best_thr).astype(int)
    t_acc = accuracy_score(t_true, t_pred)
    t_f1 = f1_score(t_true, t_pred, average="macro")

    out = {
        "dataset": args.dataset,
        "seed": args.seed,
        "init_from_pretrained": bool(args.init_from_pretrained),
        "lambda_consistency": args.lambda_consistency,
        "best_val_acc": best_val_acc * 100,
        "best_threshold": best_thr,
        "test_acc": t_acc * 100,
        "test_f1_macro": t_f1 * 100,
        "test_size": int(len(t_true)),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(CFG.keys()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lambda_consistency", type=float, default=0.25)
    p.add_argument("--init_from_pretrained", action="store_true")
    p.add_argument("--output_json", default="")
    args = p.parse_args()
    run(args)
