#!/usr/bin/env python3
"""Unified Multimodal BERT: Text understanding + multimodal feature injection.

Fine-tunes BERT on MLLM analysis + transcript text while injecting raw
audio/visual/MLLM score features into the classification head.

This combines the strengths of:
- Text classifier: captures MLLM semantic reasoning about hate content
- Multimodal features: captures modality-specific signals (audio tone, visual symbols)
- MLLM scores: captures structured assessment from video analysis
"""

import argparse
import json
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report


HVGUARD_DIR = "/data/jehc223/EMNLP2/baseline/HVGuard"

DATASET_CONFIGS = {
    "HateMM": {
        "hvguard_emb": f"{HVGUARD_DIR}/embeddings/HateMM",
        "mllm_results": "/data/jehc223/EMNLP2/results/mllm/HateMM/mllm_results.json",
        "mllm_scores": "/data/jehc223/EMNLP2/results/mllm/HateMM/mllm_scores.json",
        "label_file": f"{HVGUARD_DIR}/datasets/HateMM/data (base).json",
        "splits_dir": "/data/jehc223/HateMM/splits",
        "label_map": {"Hate": 1, "Non Hate": 0},
    },
    "MultiHateClip_CN": {
        "hvguard_emb": f"{HVGUARD_DIR}/embeddings/Multihateclip/Chinese",
        "mllm_results": "/data/jehc223/EMNLP2/results/mllm/MultiHateClip_CN/mllm_results.json",
        "mllm_scores": "/data/jehc223/EMNLP2/results/mllm/MultiHateClip_CN/mllm_scores.json",
        "label_file": f"{HVGUARD_DIR}/datasets/Multihateclip/Chinese/data.json",
        "splits_dir": "/data/jehc223/Multihateclip/Chinese/splits",
        "label_map": {"Normal": 0, "Offensive": 1, "Hateful": 1},
    },
    "MultiHateClip_EN": {
        "hvguard_emb": f"{HVGUARD_DIR}/embeddings/Multihateclip/English",
        "mllm_results": "/data/jehc223/EMNLP2/results/mllm/MultiHateClip_EN/mllm_results.json",
        "mllm_scores": "/data/jehc223/EMNLP2/results/mllm/MultiHateClip_EN/mllm_scores.json",
        "label_file": f"{HVGUARD_DIR}/datasets/Multihateclip/English/data.json",
        "splits_dir": "/data/jehc223/Multihateclip/English/splits",
        "label_map": {"Normal": 0, "Offensive": 1, "Hateful": 1},
    },
}

SCORE_KEYS = ["hate_speech_score", "visual_hate_score", "cross_modal_score",
              "implicit_hate_score", "overall_hate_score", "confidence"]


class UnifiedDataset(Dataset):
    def __init__(self, video_ids, texts, text_feats, audio_feats, frame_feats,
                 hvg_mllm_feats, mllm_scores, labels, label_map,
                 tokenizer, max_length=512):
        self.video_ids = video_ids
        self.texts = texts
        self.text_feats = text_feats
        self.audio_feats = audio_feats
        self.frame_feats = frame_feats
        self.hvg_mllm = hvg_mllm_feats
        self.scores = mllm_scores
        self.labels = labels
        self.label_map = label_map
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dim = 768

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        zero = torch.zeros(self.dim)

        # Text encoding
        text = self.texts.get(vid, "[NO CONTENT]")
        encoded = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt"
        )

        # Multimodal features
        text_feat = self.text_feats.get(vid, zero)
        audio_feat = self.audio_feats.get(vid, zero)
        frame_feat = self.frame_feats.get(vid, zero)
        hvg_mllm_feat = self.hvg_mllm.get(vid, zero)

        # MLLM scores
        sc = self.scores.get(vid, {}).get("scores", {})
        score_vec = torch.tensor([
            sc.get("hate_speech_score", 5) / 10.0,
            sc.get("visual_hate_score", 5) / 10.0,
            sc.get("cross_modal_score", 5) / 10.0,
            sc.get("implicit_hate_score", 5) / 10.0,
            sc.get("overall_hate_score", 5) / 10.0,
            sc.get("confidence", 5) / 10.0,
            1.0 if sc.get("classification", "").lower() in ["hateful", "offensive"] else 0.0,
            1.0 if sc.get("classification", "").lower() == "normal" else 0.0,
        ], dtype=torch.float)

        # Concatenate all non-text features
        multi_feat = torch.cat([text_feat, audio_feat, frame_feat, hvg_mllm_feat, score_vec])

        label = self.label_map[self.labels[vid]]
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "multi_feat": multi_feat,
            "labels": torch.tensor(label, dtype=torch.long),
        }


class UnifiedClassifier(nn.Module):
    """BERT + multimodal feature injection."""

    def __init__(self, bert_model_name, multi_feat_dim=3080, hidden_dim=256,
                 num_classes=2, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_dim = self.bert.config.hidden_size  # 768

        # Multimodal feature projection
        self.multi_proj = nn.Sequential(
            nn.Linear(multi_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
        )

        # BERT CLS projection
        self.bert_proj = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
        )

        # Gated fusion
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 2),
            nn.Softmax(dim=-1),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, input_ids, attention_mask, multi_feat):
        # BERT encoding
        bert_out = self.bert(input_ids, attention_mask=attention_mask)
        cls_repr = bert_out.last_hidden_state[:, 0]  # [CLS] token

        # Project both streams
        bert_h = self.bert_proj(cls_repr)
        multi_h = self.multi_proj(multi_feat)

        # Gated fusion
        gate_input = torch.cat([bert_h, multi_h], dim=-1)
        gate_weights = self.gate(gate_input)  # (B, 2)
        fused = gate_weights[:, 0:1] * bert_h + gate_weights[:, 1:2] * multi_h

        return self.classifier(fused)


def build_text_input(vid, mllm_results, mllm_scores, ann_data):
    parts = []
    if vid in mllm_results:
        analysis = mllm_results[vid].get("analysis", "")
        if analysis and not analysis.startswith("ERROR") and not analysis.startswith("[TEXT-ONLY"):
            parts.append("[ANALYSIS] " + analysis[:1500])
    ann = ann_data.get(vid, {})
    mix = ann.get("Mix_description", "").strip()
    if mix:
        parts.append("[HVGUARD] " + mix[:500])
    title = ann.get("Title", "").strip()
    transcript = ann.get("Transcript", "").strip()
    if title:
        parts.append("[TITLE] " + title)
    if transcript:
        parts.append("[TRANSCRIPT] " + transcript[:500])
    return " ".join(parts) if parts else "[NO CONTENT]"


def train_and_eval(ds_name, seed, args):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cfg = DATASET_CONFIGS[ds_name]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    mllm_results = {}
    if os.path.exists(cfg["mllm_results"]):
        with open(cfg["mllm_results"]) as f:
            mllm_results = json.load(f)

    mllm_scores = {}
    if os.path.exists(cfg["mllm_scores"]):
        with open(cfg["mllm_scores"]) as f:
            mllm_scores = json.load(f)

    with open(cfg["label_file"]) as f:
        raw_data = json.load(f)
    ann_data = {d["Video_ID"]: d for d in raw_data}
    labels = {d["Video_ID"]: d["Label"] for d in raw_data}

    text_feats = torch.load(cfg["hvguard_emb"] + "/text_features.pth", map_location="cpu", weights_only=True)
    audio_feats = torch.load(cfg["hvguard_emb"] + "/audio_features.pth", map_location="cpu", weights_only=True)
    frame_feats = torch.load(cfg["hvguard_emb"] + "/frame_features.pth", map_location="cpu", weights_only=True)
    hvg_mllm = torch.load(cfg["hvguard_emb"] + "/MLLM_rationale_features.pth", map_location="cpu", weights_only=True)

    texts = {vid: build_text_input(vid, mllm_results, mllm_scores, ann_data) for vid in ann_data}

    splits = {}
    for split in ["train", "valid", "test"]:
        with open(os.path.join(cfg["splits_dir"], split + ".csv")) as f:
            splits[split] = [l.strip() for l in f
                             if l.strip() in labels and labels[l.strip()] in cfg["label_map"]]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # multi_feat_dim = text(768) + audio(768) + frame(768) + hvg_mllm(768) + scores(8) = 3080
    multi_feat_dim = 768 * 4 + 8

    mk = lambda ids: UnifiedDataset(ids, texts, text_feats, audio_feats, frame_feats,
                                     hvg_mllm, mllm_scores, labels, cfg["label_map"],
                                     tokenizer, args.max_length)
    train_ds, valid_ds, test_ds = mk(splits["train"]), mk(splits["valid"]), mk(splits["test"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False)

    model = UnifiedClassifier(
        args.model_name, multi_feat_dim=multi_feat_dim,
        hidden_dim=args.hidden_dim, dropout=args.dropout
    ).to(device)

    # Class weights
    train_labels = [cfg["label_map"][labels[vid]] for vid in splits["train"]]
    counts = Counter(train_labels)
    total = sum(counts.values())
    weights = torch.tensor([total / (2 * counts[c]) for c in range(2)], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Separate LR for BERT and classification head
    bert_params = list(model.bert.parameters())
    head_params = list(model.multi_proj.parameters()) + list(model.bert_proj.parameters()) + \
                  list(model.gate.parameters()) + list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {"params": bert_params, "lr": args.bert_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    best_val_acc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            mf = batch["multi_feat"].to(device)
            y = batch["labels"].to(device)

            optimizer.zero_grad()
            out = model(ids, mask, mf)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        # Validate
        model.eval()
        vpreds, vlabels = [], []
        with torch.no_grad():
            for batch in valid_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                mf = batch["multi_feat"].to(device)
                out = model(ids, mask, mf)
                vpreds.extend(out.argmax(1).cpu().numpy())
                vlabels.extend(batch["labels"].numpy())
        val_acc = accuracy_score(vlabels, vpreds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    # Test
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    tpreds, tlabels = [], []
    with torch.no_grad():
        for batch in test_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            mf = batch["multi_feat"].to(device)
            out = model(ids, mask, mf)
            tpreds.extend(out.argmax(1).cpu().numpy())
            tlabels.extend(batch["labels"].numpy())

    test_acc = accuracy_score(tlabels, tpreds)
    test_f1 = f1_score(tlabels, tpreds, average="macro")

    print(f"  {ds_name} seed={seed}: val={best_val_acc:.3f} test={test_acc:.3f} ({test_acc*100:.1f}%) f1={test_f1:.3f}",
          flush=True)
    print(classification_report(tlabels, tpreds, target_names=["Normal", "Hateful"]), flush=True)
    return test_acc, test_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["HateMM", "MultiHateClip_CN", "MultiHateClip_EN"])
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--bert_lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1024])
    args = parser.parse_args()

    for ds in args.datasets:
        print(f"\n{'='*60}\n{ds}\n{'='*60}", flush=True)
        accs, f1s = [], []
        for seed in args.seeds:
            acc, f1 = train_and_eval(ds, seed, args)
            accs.append(acc)
            f1s.append(f1)

        avg = np.mean(accs) * 100
        std = np.std(accs) * 100
        best = max(accs) * 100
        target = 90 if ds == "HateMM" else 85
        gap = target - avg
        status = "TARGET MET" if avg >= target else f"gap: {gap:.1f}%"
        print(f"\n  {ds}: avg={avg:.1f}% std={std:.1f}% best={best:.1f}% — {status}", flush=True)


if __name__ == "__main__":
    main()
