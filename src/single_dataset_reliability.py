#!/usr/bin/env python3
"""Single-dataset reliability-conditioned multimodal classifier.

Hard constraints:
- Train on one dataset only.
- Evaluate on the same dataset only.
- No ensemble, no cross-dataset training.
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
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

HVGUARD_DIR = "/data/jehc223/EMNLP2/baseline/HVGuard"
DATASET_CONFIGS = {
    "HateMM": {
        "hvguard_emb": f"{HVGUARD_DIR}/embeddings/HateMM",
        "mllm_results": "/data/jehc223/EMNLP2/results/mllm/HateMM/mllm_results.json",
        "mllm_scores": "/data/jehc223/EMNLP2/results/mllm/HateMM/mllm_scores.json",
        "label_file": f"{HVGUARD_DIR}/datasets/HateMM/data (base).json",
        "splits_dir": "/data/jehc223/HateMM/splits",
        "label_map_bin": {"Non Hate": 0, "Hate": 1},
        "label_map_3": None,
    },
    "MultiHateClip_CN": {
        "hvguard_emb": f"{HVGUARD_DIR}/embeddings/Multihateclip/Chinese",
        "mllm_results": "/data/jehc223/EMNLP2/results/mllm/MultiHateClip_CN/mllm_results.json",
        "mllm_scores": "/data/jehc223/EMNLP2/results/mllm/MultiHateClip_CN/mllm_scores.json",
        "label_file": f"{HVGUARD_DIR}/datasets/Multihateclip/Chinese/data.json",
        "splits_dir": "/data/jehc223/Multihateclip/Chinese/splits",
        "label_map_bin": {"Normal": 0, "Offensive": 1, "Hateful": 1},
        "label_map_3": {"Normal": 0, "Offensive": 1, "Hateful": 2},
    },
    "MultiHateClip_EN": {
        "hvguard_emb": f"{HVGUARD_DIR}/embeddings/Multihateclip/English",
        "mllm_results": "/data/jehc223/EMNLP2/results/mllm/MultiHateClip_EN/mllm_results.json",
        "mllm_scores": "/data/jehc223/EMNLP2/results/mllm/MultiHateClip_EN/mllm_scores.json",
        "label_file": f"{HVGUARD_DIR}/datasets/Multihateclip/English/data.json",
        "splits_dir": "/data/jehc223/Multihateclip/English/splits",
        "label_map_bin": {"Normal": 0, "Offensive": 1, "Hateful": 1},
        "label_map_3": {"Normal": 0, "Offensive": 1, "Hateful": 2},
    },
}
SCORE_KEYS = [
    "hate_speech_score",
    "visual_hate_score",
    "cross_modal_score",
    "implicit_hate_score",
    "overall_hate_score",
    "confidence",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(labels, num_classes):
    counts = Counter(labels)
    total = len(labels)
    present = [c for c in range(num_classes) if counts.get(c, 0) > 0]
    w = torch.ones(num_classes, dtype=torch.float)
    for c in present:
        w[c] = total / (len(present) * counts[c])
    return w


def build_text(vid, mllm_results, mllm_scores, ann_data):
    parts = []
    analysis = mllm_results.get(vid, {}).get("analysis", "")
    if analysis and not analysis.startswith("ERROR") and not analysis.startswith("[TEXT"):
        parts.append("[ANALYSIS] " + analysis[:1800])

    ann = ann_data.get(vid, {})
    mix = ann.get("Mix_description", "").strip()
    if mix:
        parts.append("[HVGUARD] " + mix[:600])

    title = ann.get("Title", "").strip()
    transcript = ann.get("Transcript", "").strip()
    if title:
        parts.append("[TITLE] " + title)
    if transcript:
        parts.append("[TRANSCRIPT] " + transcript[:600])

    sc = mllm_scores.get(vid, {}).get("scores", {})
    score_text = (
        "[SCORES] "
        f"hate={sc.get('hate_speech_score', '?')}/10 "
        f"visual={sc.get('visual_hate_score', '?')}/10 "
        f"overall={sc.get('overall_hate_score', '?')}/10 "
        f"conf={sc.get('confidence', '?')}/10 "
        f"verdict={sc.get('classification', '?')}"
    )
    parts.append(score_text)

    return " ".join(parts) if parts else "[NO CONTENT]"


def reliability_features(vid, mllm_results, mllm_scores, frame_feats):
    analysis = mllm_results.get(vid, {}).get("analysis", "")
    is_text_only = 1.0 if analysis.startswith("[TEXT") else 0.0
    has_valid_analysis = 1.0 if analysis and not analysis.startswith("ERROR") else 0.0
    frame_missing = 0.0 if vid in frame_feats else 1.0

    sc = mllm_scores.get(vid, {}).get("scores", {})
    vals = [
        float(sc.get("hate_speech_score", 5)),
        float(sc.get("visual_hate_score", 5)),
        float(sc.get("cross_modal_score", 5)),
        float(sc.get("implicit_hate_score", 5)),
        float(sc.get("overall_hate_score", 5)),
    ]
    conf = float(sc.get("confidence", 5)) / 10.0
    disp = float(np.std(vals)) / 10.0
    cls_hate = 1.0 if str(sc.get("classification", "")).lower() in ["hateful", "offensive"] else 0.0

    return torch.tensor(
        [is_text_only, has_valid_analysis, frame_missing, conf, disp, cls_hate],
        dtype=torch.float,
    )


class SingleDataset(Dataset):
    def __init__(
        self,
        video_ids,
        texts,
        text_feats,
        audio_feats,
        frame_feats,
        hvg_mllm_feats,
        mllm_results,
        mllm_scores,
        labels_train,
        labels_bin,
        tokenizer,
        max_length,
    ):
        self.video_ids = video_ids
        self.texts = texts
        self.text_feats = text_feats
        self.audio_feats = audio_feats
        self.frame_feats = frame_feats
        self.hvg_mllm = hvg_mllm_feats
        self.mllm_results = mllm_results
        self.mllm_scores = mllm_scores
        self.labels_train = labels_train
        self.labels_bin = labels_bin
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        zero = torch.zeros(768)

        enc = self.tokenizer(
            self.texts.get(vid, "[NO CONTENT]"),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        text_feat = self.text_feats.get(vid, zero)
        audio_feat = self.audio_feats.get(vid, zero)
        frame_feat = self.frame_feats.get(vid, zero)
        hvg_mllm_feat = self.hvg_mllm.get(vid, zero)

        sc = self.mllm_scores.get(vid, {}).get("scores", {})
        score_vec = torch.tensor(
            [
                sc.get("hate_speech_score", 5) / 10.0,
                sc.get("visual_hate_score", 5) / 10.0,
                sc.get("cross_modal_score", 5) / 10.0,
                sc.get("implicit_hate_score", 5) / 10.0,
                sc.get("overall_hate_score", 5) / 10.0,
                sc.get("confidence", 5) / 10.0,
                1.0 if str(sc.get("classification", "")).lower() in ["hateful", "offensive"] else 0.0,
                1.0 if str(sc.get("classification", "")).lower() == "normal" else 0.0,
            ],
            dtype=torch.float,
        )

        multi_feat = torch.cat([text_feat, audio_feat, frame_feat, hvg_mllm_feat, score_vec])
        rel_feat = reliability_features(vid, self.mllm_results, self.mllm_scores, self.frame_feats)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "multi_feat": multi_feat,
            "rel_feat": rel_feat,
            "label_train": torch.tensor(self.labels_train[vid], dtype=torch.long),
            "label_bin": torch.tensor(self.labels_bin[vid], dtype=torch.long),
        }


class ReliabilityFusionClassifier(nn.Module):
    def __init__(self, model_name, num_labels_train=2, multi_feat_dim=3080, rel_dim=6, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        enc_dim = self.encoder.config.hidden_size

        self.text_proj = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
        )
        self.multi_proj = nn.Sequential(
            nn.Linear(multi_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
        )
        self.rel_proj = nn.Sequential(
            nn.Linear(rel_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.1),
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.1),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels_train),
        )

    def forward(self, input_ids, attention_mask, multi_feat, rel_feat):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = enc.last_hidden_state[:, 0]

        text_h = self.text_proj(cls)
        multi_h = self.multi_proj(multi_feat)
        rel_h = self.rel_proj(rel_feat)

        gate_in = torch.cat([text_h, multi_h, rel_h], dim=-1)
        g = self.gate(gate_in)
        fused = g[:, 0:1] * text_h + g[:, 1:2] * multi_h

        logits = self.classifier(fused)
        return logits


def load_dataset(dataset, use_three_class):
    cfg = DATASET_CONFIGS[dataset]

    with open(cfg["label_file"], "r", encoding="utf-8") as f:
        raw = json.load(f)

    label_map_bin = cfg["label_map_bin"]
    if use_three_class and cfg["label_map_3"] is not None:
        label_map_train = cfg["label_map_3"]
        num_labels_train = 3
    else:
        label_map_train = label_map_bin
        num_labels_train = 2

    labels_train = {}
    labels_bin = {}
    ann_data = {}
    for item in raw:
        vid = item["Video_ID"]
        label = item["Label"]
        if label in label_map_bin and label in label_map_train:
            labels_train[vid] = label_map_train[label]
            labels_bin[vid] = label_map_bin[label]
            ann_data[vid] = item

    mllm_results = {}
    if os.path.exists(cfg["mllm_results"]):
        with open(cfg["mllm_results"], "r", encoding="utf-8") as f:
            mllm_results = json.load(f)

    mllm_scores = {}
    if os.path.exists(cfg["mllm_scores"]):
        with open(cfg["mllm_scores"], "r", encoding="utf-8") as f:
            mllm_scores = json.load(f)

    text_feats = torch.load(os.path.join(cfg["hvguard_emb"], "text_features.pth"), map_location="cpu", weights_only=True)
    audio_feats = torch.load(os.path.join(cfg["hvguard_emb"], "audio_features.pth"), map_location="cpu", weights_only=True)
    frame_feats = torch.load(os.path.join(cfg["hvguard_emb"], "frame_features.pth"), map_location="cpu", weights_only=True)
    hvg_mllm = torch.load(os.path.join(cfg["hvguard_emb"], "MLLM_rationale_features.pth"), map_location="cpu", weights_only=True)

    texts = {vid: build_text(vid, mllm_results, mllm_scores, ann_data) for vid in labels_train}

    splits = {}
    for split in ["train", "valid", "test"]:
        with open(os.path.join(cfg["splits_dir"], split + ".csv"), "r", encoding="utf-8") as f:
            splits[split] = [
                line.strip()
                for line in f
                if line.strip() in labels_train
            ]

    return {
        "labels_train": labels_train,
        "labels_bin": labels_bin,
        "texts": texts,
        "mllm_results": mllm_results,
        "mllm_scores": mllm_scores,
        "text_feats": text_feats,
        "audio_feats": audio_feats,
        "frame_feats": frame_feats,
        "hvg_mllm": hvg_mllm,
        "splits": splits,
        "num_labels_train": num_labels_train,
    }


def logits_to_bin_probs(logits, num_labels_train):
    probs = torch.softmax(logits, dim=-1)
    if num_labels_train == 3:
        return probs[:, 1] + probs[:, 2]
    return probs[:, 1]


def choose_threshold(val_probs, val_true):
    best_thr = 0.5
    best_acc = -1.0
    for thr in np.linspace(0.2, 0.8, 61):
        pred = (val_probs >= thr).astype(int)
        acc = accuracy_score(val_true, pred)
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)
    return best_thr, best_acc


def run(args):
    set_seed(args.seed)

    use_three_class = bool(args.three_class_for_multihate and args.dataset.startswith("MultiHateClip"))
    data = load_dataset(args.dataset, use_three_class=use_three_class)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def mk(ids):
        return SingleDataset(
            ids,
            data["texts"],
            data["text_feats"],
            data["audio_feats"],
            data["frame_feats"],
            data["hvg_mllm"],
            data["mllm_results"],
            data["mllm_scores"],
            data["labels_train"],
            data["labels_bin"],
            tokenizer,
            args.max_length,
        )

    train_ds = mk(data["splits"]["train"])
    valid_ds = mk(data["splits"]["valid"])
    test_ds = mk(data["splits"]["test"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False)

    model = ReliabilityFusionClassifier(
        model_name=args.model_name,
        num_labels_train=data["num_labels_train"],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    train_labels = [data["labels_train"][vid] for vid in data["splits"]["train"]]
    weights = class_weights(train_labels, data["num_labels_train"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    enc_params = list(model.encoder.parameters())
    head_params = (
        list(model.text_proj.parameters())
        + list(model.multi_proj.parameters())
        + list(model.rel_proj.parameters())
        + list(model.gate.parameters())
        + list(model.classifier.parameters())
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": args.encoder_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(total_steps // 10, 1),
        num_training_steps=max(total_steps, 1),
    )

    best_state = None
    best_val_acc = 0.0
    best_thr = 0.5
    bad_epochs = 0

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            mf = batch["multi_feat"].to(device)
            rf = batch["rel_feat"].to(device)
            y = batch["label_train"].to(device)

            optimizer.zero_grad()
            logits = model(ids, mask, mf, rf)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        val_probs, val_true = [], []
        with torch.no_grad():
            for batch in valid_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                mf = batch["multi_feat"].to(device)
                rf = batch["rel_feat"].to(device)
                logits = model(ids, mask, mf, rf)
                p = logits_to_bin_probs(logits, data["num_labels_train"])
                val_probs.extend(p.cpu().numpy())
                val_true.extend(batch["label_bin"].numpy())

        val_probs = np.array(val_probs)
        val_true = np.array(val_true)
        thr, val_acc = choose_threshold(val_probs, val_true)

        print(f"epoch={epoch+1} val_acc={val_acc:.4f} thr={thr:.2f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            best_thr = float(thr)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device).eval()

    test_probs, test_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            mf = batch["multi_feat"].to(device)
            rf = batch["rel_feat"].to(device)
            logits = model(ids, mask, mf, rf)
            p = logits_to_bin_probs(logits, data["num_labels_train"])
            test_probs.extend(p.cpu().numpy())
            test_true.extend(batch["label_bin"].numpy())

    test_probs = np.array(test_probs)
    test_true = np.array(test_true)
    test_pred = (test_probs >= best_thr).astype(int)

    test_acc = accuracy_score(test_true, test_pred)
    test_f1 = f1_score(test_true, test_pred, average="macro")

    print(
        f"FINAL {args.dataset}: val_best={best_val_acc*100:.2f}% "
        f"thr={best_thr:.2f} test_acc={test_acc*100:.2f}% test_f1={test_f1*100:.2f}%",
        flush=True,
    )

    out = {
        "dataset": args.dataset,
        "seed": args.seed,
        "model": args.model_name,
        "num_labels_train": data["num_labels_train"],
        "best_val_acc": best_val_acc * 100,
        "best_threshold": best_thr,
        "test_acc": test_acc * 100,
        "test_f1_macro": test_f1 * 100,
        "test_size": int(len(test_true)),
        "test_pos_rate": float(test_true.mean()),
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"saved: {args.output_json}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["HateMM", "MultiHateClip_CN", "MultiHateClip_EN"])
    parser.add_argument("--model_name", default="microsoft/deberta-v3-base")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--encoder_lr", type=float, default=1.5e-5)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--three_class_for_multihate", action="store_true")
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    run(args)
