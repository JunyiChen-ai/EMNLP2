#!/usr/bin/env python3
"""CMEV v2 Classifier: Uses structured MLLM scores + multimodal features.

Key improvement over v1: Instead of encoding the full MLLM analysis text into
a single vector (which loses information), we use structured MLLM scores
(6 dimensional: hate_speech, visual_hate, cross_modal, implicit, overall, confidence)
as direct features. These capture the MLLM's multi-dimensional assessment.
"""

import argparse
import json
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report


HVGUARD_DIR = "/data/jehc223/EMNLP2/baseline/HVGuard"

DATASET_CONFIGS = {
    "HateMM": {
        "hvguard_emb": f"{HVGUARD_DIR}/embeddings/HateMM",
        "label_file": f"{HVGUARD_DIR}/datasets/HateMM/data (base).json",
        "splits_dir": "/data/jehc223/HateMM/splits",
        "label_map": {"Hate": 1, "Non Hate": 0},
    },
    "MultiHateClip_CN": {
        "hvguard_emb": f"{HVGUARD_DIR}/embeddings/Multihateclip/Chinese",
        "label_file": f"{HVGUARD_DIR}/datasets/Multihateclip/Chinese/data.json",
        "splits_dir": "/data/jehc223/Multihateclip/Chinese/splits",
        "label_map": {"Normal": 0, "Offensive": 1, "Hateful": 1},
    },
    "MultiHateClip_EN": {
        "hvguard_emb": f"{HVGUARD_DIR}/embeddings/Multihateclip/English",
        "label_file": f"{HVGUARD_DIR}/datasets/Multihateclip/English/data.json",
        "splits_dir": "/data/jehc223/Multihateclip/English/splits",
        "label_map": {"Normal": 0, "Offensive": 1, "Hateful": 1},
    },
}

SCORE_KEYS = ["hate_speech_score", "visual_hate_score", "cross_modal_score",
              "implicit_hate_score", "overall_hate_score", "confidence"]


class CMEVv2Dataset(Dataset):
    def __init__(self, video_ids, text_feats, audio_feats, frame_feats,
                 hvg_mllm_feats, our_mllm_feats, mllm_scores, labels, label_map):
        self.video_ids = video_ids
        self.text = text_feats
        self.audio = audio_feats
        self.frame = frame_feats
        self.hvg_mllm = hvg_mllm_feats
        self.our_mllm = our_mllm_feats
        self.scores = mllm_scores
        self.labels = labels
        self.label_map = label_map
        self.dim = 768

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        zero = torch.zeros(self.dim)

        text = self.text.get(vid, zero)
        audio = self.audio.get(vid, zero)
        frame = self.frame.get(vid, zero)
        hvg_mllm = self.hvg_mllm.get(vid, zero)
        our_mllm = self.our_mllm.get(vid, zero) if self.our_mllm else zero

        # Extract structured MLLM scores (6 dims + classification signal)
        sc = self.scores.get(vid, {}).get("scores", {})
        score_vec = torch.tensor([
            sc.get("hate_speech_score", 5) / 10.0,
            sc.get("visual_hate_score", 5) / 10.0,
            sc.get("cross_modal_score", 5) / 10.0,
            sc.get("implicit_hate_score", 5) / 10.0,
            sc.get("overall_hate_score", 5) / 10.0,
            sc.get("confidence", 5) / 10.0,
            1.0 if sc.get("classification", "").lower() == "hateful" else 0.0,
            1.0 if sc.get("classification", "").lower() == "normal" else 0.0,
        ], dtype=torch.float)

        label = self.label_map[self.labels[vid]]
        return text, audio, frame, hvg_mllm, our_mllm, score_vec, torch.tensor(label, dtype=torch.long)


class CMEVv2Classifier(nn.Module):
    """CMEV v2: Structured MLLM scores + cross-modal attention."""

    def __init__(self, feat_dim=768, hidden_dim=256, num_heads=4, num_classes=2,
                 dropout=0.3, use_our_mllm=True, score_dim=8):
        super().__init__()
        self.use_our_mllm = use_our_mllm

        # Modality projections
        self.text_proj = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout * 0.3))
        self.audio_proj = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout * 0.3))
        self.frame_proj = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout * 0.3))

        # MLLM rationale
        mllm_dim = feat_dim * 2 if use_our_mllm else feat_dim
        self.mllm_proj = nn.Sequential(nn.Linear(mllm_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout * 0.3))

        # Score projection: map structured scores to hidden dim
        self.score_proj = nn.Sequential(
            nn.Linear(score_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.GELU(),
        )

        # Cross-attention: scores+MLLM as query, modalities as keys
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.1)
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # Self-attention over all modalities
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.1)
        self.self_norm = nn.LayerNorm(hidden_dim)

        # Gated fusion of multiple streams
        self.gate_cross = nn.Linear(hidden_dim, 1)
        self.gate_self = nn.Linear(hidden_dim, 1)
        self.gate_score = nn.Linear(hidden_dim, 1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, text, audio, frame, hvg_mllm, our_mllm, scores):
        B = text.size(0)

        # Project modalities
        text_h = self.text_proj(text)
        audio_h = self.audio_proj(audio)
        frame_h = self.frame_proj(frame)

        # MLLM rationale
        if self.use_our_mllm:
            mllm_input = torch.cat([hvg_mllm, our_mllm], dim=-1)
        else:
            mllm_input = hvg_mllm
        mllm_h = self.mllm_proj(mllm_input)

        # Score representation
        score_h = self.score_proj(scores)

        # Stack all modality representations
        modalities = torch.stack([text_h, audio_h, frame_h, mllm_h], dim=1)  # (B, 4, H)

        # Cross-attention: score-guided attention over modalities
        score_query = score_h.unsqueeze(1)  # (B, 1, H)
        cross_out, _ = self.cross_attn(score_query, modalities, modalities)
        cross_out = self.cross_norm(cross_out.squeeze(1))  # (B, H)

        # Self-attention over modalities
        self_out, _ = self.self_attn(modalities, modalities, modalities)
        self_out = self.self_norm(self_out + modalities).mean(dim=1)  # (B, H)

        # Gated fusion of three streams
        g_cross = torch.sigmoid(self.gate_cross(cross_out))
        g_self = torch.sigmoid(self.gate_self(self_out))
        g_score = torch.sigmoid(self.gate_score(score_h))

        # Normalize gates
        g_sum = g_cross + g_self + g_score + 1e-8
        fused = (g_cross / g_sum) * cross_out + (g_self / g_sum) * self_out + (g_score / g_sum) * score_h

        return self.classifier(fused)


def load_features(ds_name, mllm_emb_dir, score_dir):
    cfg = DATASET_CONFIGS[ds_name]

    text = torch.load(f"{cfg['hvguard_emb']}/text_features.pth", map_location="cpu", weights_only=True)
    audio = torch.load(f"{cfg['hvguard_emb']}/audio_features.pth", map_location="cpu", weights_only=True)
    frame = torch.load(f"{cfg['hvguard_emb']}/frame_features.pth", map_location="cpu", weights_only=True)
    hvg_mllm = torch.load(f"{cfg['hvguard_emb']}/MLLM_rationale_features.pth", map_location="cpu", weights_only=True)

    # Our MLLM rationale embedding (from iteration 1)
    our_mllm = None
    our_path = os.path.join(mllm_emb_dir, ds_name, "mllm_rationale_features.pth")
    if os.path.exists(our_path):
        our_mllm = torch.load(our_path, map_location="cpu", weights_only=True)

    # MLLM scores
    score_path = os.path.join(score_dir, ds_name, "mllm_scores.json")
    scores = {}
    if os.path.exists(score_path):
        with open(score_path) as f:
            scores = json.load(f)
        print(f"  Loaded {len(scores)} MLLM scores for {ds_name}")

    with open(cfg["label_file"]) as f:
        labels = {d["Video_ID"]: d["Label"] for d in json.load(f)}

    splits = {}
    for split in ["train", "valid", "test"]:
        with open(f"{cfg['splits_dir']}/{split}.csv") as f:
            splits[split] = [l.strip() for l in f if l.strip() in labels and labels[l.strip()] in cfg["label_map"]]

    return text, audio, frame, hvg_mllm, our_mllm, scores, labels, cfg["label_map"], splits


def train_one(ds_name, seed, mllm_emb_dir, score_dir, args):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    text, audio, frame, hvg_mllm, our_mllm, scores, labels, label_map, splits = \
        load_features(ds_name, mllm_emb_dir, score_dir)

    use_our_mllm = our_mllm is not None
    has_scores = len(scores) > 0

    mk = lambda ids: CMEVv2Dataset(ids, text, audio, frame, hvg_mllm, our_mllm, scores, labels, label_map)
    train_ds, valid_ds, test_ds = mk(splits["train"]), mk(splits["valid"]), mk(splits["test"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CMEVv2Classifier(
        hidden_dim=args.hidden_dim, dropout=args.dropout,
        use_our_mllm=use_our_mllm
    ).to(device)

    # Class weights
    train_labels = [label_map[labels[vid]] for vid in splits["train"]]
    counts = Counter(train_labels)
    total = sum(counts.values())
    weights = torch.tensor([total / (2 * counts[c]) for c in range(2)], dtype=torch.float).to(device)

    # Focal loss
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_val_acc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            t, a, f, hm, om, sc, y = [b.to(device) for b in batch]
            optimizer.zero_grad()
            out = model(t, a, f, hm, om, sc)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        vpreds, vlabels = [], []
        with torch.no_grad():
            for batch in valid_loader:
                t, a, f, hm, om, sc, y = [b.to(device) for b in batch]
                out = model(t, a, f, hm, om, sc)
                vpreds.extend(out.argmax(1).cpu().numpy())
                vlabels.extend(y.cpu().numpy())
        val_acc = accuracy_score(vlabels, vpreds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    tpreds, tlabels = [], []
    with torch.no_grad():
        for batch in test_loader:
            t, a, f, hm, om, sc, y = [b.to(device) for b in batch]
            out = model(t, a, f, hm, om, sc)
            tpreds.extend(out.argmax(1).cpu().numpy())
            tlabels.extend(y.cpu().numpy())

    test_acc = accuracy_score(tlabels, tpreds)
    test_f1 = f1_score(tlabels, tpreds, average="macro")
    print(f"  {ds_name} seed={seed}: val={best_val_acc:.3f} test={test_acc:.3f} ({test_acc*100:.1f}%) f1={test_f1:.3f} [scores={has_scores}]")
    return test_acc, test_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["HateMM", "MultiHateClip_CN", "MultiHateClip_EN"])
    parser.add_argument("--mllm_emb_dir", default="/data/jehc223/EMNLP2/embeddings")
    parser.add_argument("--score_dir", default="/data/jehc223/EMNLP2/results/mllm")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1024])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    for ds in args.datasets:
        print(f"\n{'='*60}\n{ds}\n{'='*60}")
        accs, f1s = [], []
        for seed in args.seeds:
            acc, f1 = train_one(ds, seed, args.mllm_emb_dir, args.score_dir, args)
            accs.append(acc)
            f1s.append(f1)

        avg = np.mean(accs) * 100
        std = np.std(accs) * 100
        best = max(accs) * 100
        target = 90 if ds == "HateMM" else 85
        gap = target - avg
        status = "TARGET MET" if avg >= target else f"gap: {gap:.1f}%"
        print(f"  {ds}: avg={avg:.1f}% std={std:.1f}% best={best:.1f}% — {status}")


if __name__ == "__main__":
    main()
