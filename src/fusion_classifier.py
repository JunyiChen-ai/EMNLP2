"""
Multimodal fusion classifier for hateful video detection.
Fuses: rationale embeddings + frame features + audio features.

Supports multiple fusion strategies for ablation:
  - rationale_only: just rationale MLP
  - concat: simple concatenation + MLP
  - gated: gated residual fusion (rationale as anchor, AV as residual)

Usage:
  python src/fusion_classifier.py --dataset HateMM --prompt-family diagnostic --fusion gated --seeds 10
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path

DATASET_CONFIGS = {
    "HateMM": {
        "annotation": "datasets/HateMM/annotation(new).json",
        "splits_dir": "datasets/HateMM/splits",
        "id_field": "Video_ID",
        "label_field": "Label",
        "label_map": {"Hate": 1, "Non Hate": 0},
        "num_classes": 2,
    },
    "MHClip_EN": {
        "annotation": "datasets/MHClip_EN/annotation(new).json",
        "splits_dir": "datasets/MHClip_EN/splits",
        "id_field": "Video_ID",
        "label_field": "Label",
        "label_map": {"Hateful": 1, "Offensive": 1, "Normal": 0},
        "num_classes": 2,
    },
    "MHClip_ZH": {
        "annotation": "datasets/MHClip_ZH/annotation(new).json",
        "splits_dir": "datasets/MHClip_ZH/splits",
        "id_field": "Video_ID",
        "label_field": "Label",
        "label_map": {"Hateful": 1, "Offensive": 1, "Normal": 0},
        "num_classes": 2,
    },
}


class MultimodalDataset(Dataset):
    def __init__(self, vid_ids, labels, rationale_feats, frame_feats, audio_feats, feat_dim=768):
        self.vid_ids = vid_ids
        self.labels = labels
        self.rationale_feats = rationale_feats
        self.frame_feats = frame_feats
        self.audio_feats = audio_feats
        self.feat_dim = feat_dim

    def __len__(self):
        return len(self.vid_ids)

    def __getitem__(self, idx):
        vid_id = self.vid_ids[idx]
        label = self.labels[idx]
        r = self.rationale_feats.get(vid_id, np.zeros(self.feat_dim, dtype=np.float32))
        f = self.frame_feats.get(vid_id, np.zeros(self.feat_dim, dtype=np.float32))
        a = self.audio_feats.get(vid_id, np.zeros(self.feat_dim, dtype=np.float32))
        return {
            "rationale": torch.tensor(r, dtype=torch.float32),
            "frame": torch.tensor(f, dtype=torch.float32),
            "audio": torch.tensor(a, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }


class RationaleOnlyClassifier(nn.Module):
    def __init__(self, feat_dim=768, hidden_dim=256, num_classes=2, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, batch):
        return self.mlp(batch["rationale"])


class ConcatFusionClassifier(nn.Module):
    """Simple concat of all modalities → MLP."""
    def __init__(self, feat_dim=768, hidden_dim=256, num_classes=2, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, batch):
        x = torch.cat([batch["rationale"], batch["frame"], batch["audio"]], dim=-1)
        return self.mlp(x)


class GatedResidualFusionClassifier(nn.Module):
    """
    Rationale as semantic anchor; frame/audio contribute via gated residual.
    AV can refine but not overwrite the rationale signal.
    """
    def __init__(self, feat_dim=768, hidden_dim=256, num_classes=2, dropout=0.3):
        super().__init__()
        # Project AV to same space
        self.frame_proj = nn.Linear(feat_dim, hidden_dim)
        self.audio_proj = nn.Linear(feat_dim, hidden_dim)
        self.rationale_proj = nn.Linear(feat_dim, hidden_dim)

        # Gate: learned scalar for how much AV residual to add
        self.gate_frame = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.gate_audio = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, batch):
        r = self.rationale_proj(batch["rationale"])   # [B, H]
        f = self.frame_proj(batch["frame"])            # [B, H]
        a = self.audio_proj(batch["audio"])            # [B, H]

        # Gated residual: AV refines rationale
        g_f = self.gate_frame(torch.cat([r, f], dim=-1))  # [B, H]
        g_a = self.gate_audio(torch.cat([r, a], dim=-1))  # [B, H]

        fused = r + g_f * f + g_a * a  # rationale anchor + gated AV residual
        return self.classifier(fused)


def load_split(splits_dir, split_name, project_root):
    path = os.path.join(project_root, splits_dir, f"{split_name}.csv")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def run_single_seed(args, seed, rationale_feats, frame_feats, audio_feats, annotations, cfg, project_root):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build splits
    id2label = {}
    for s in annotations:
        vid_id = s[cfg["id_field"]]
        id2label[vid_id] = cfg["label_map"].get(s[cfg["label_field"]], -1)

    train_ids = load_split(cfg["splits_dir"], "train", project_root)
    val_ids = load_split(cfg["splits_dir"], "valid", project_root)
    test_ids = load_split(cfg["splits_dir"], "test", project_root)

    # Filter to samples that have features
    available = set(rationale_feats.keys())
    train_ids = [v for v in train_ids if v in available and v in id2label]
    val_ids = [v for v in val_ids if v in available and v in id2label]
    test_ids = [v for v in test_ids if v in available and v in id2label]

    train_labels = [id2label[v] for v in train_ids]
    val_labels = [id2label[v] for v in val_ids]
    test_labels = [id2label[v] for v in test_ids]

    train_ds = MultimodalDataset(train_ids, train_labels, rationale_feats, frame_feats, audio_feats)
    val_ds = MultimodalDataset(val_ids, val_labels, rationale_feats, frame_feats, audio_feats)
    test_ds = MultimodalDataset(test_ids, test_labels, rationale_feats, frame_feats, audio_feats)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Model
    if args.fusion == "rationale_only":
        model = RationaleOnlyClassifier(num_classes=cfg["num_classes"], dropout=args.dropout).cuda()
    elif args.fusion == "concat":
        model = ConcatFusionClassifier(num_classes=cfg["num_classes"], dropout=args.dropout).cuda()
    elif args.fusion == "gated":
        model = GatedResidualFusionClassifier(num_classes=cfg["num_classes"], dropout=args.dropout).cuda()
    else:
        raise ValueError(f"Unknown fusion: {args.fusion}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_acc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            logits = model(batch)
            loss = criterion(logits, batch["label"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.cuda() for k, v in batch.items()}
                logits = model(batch)
                preds = logits.argmax(dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(batch["label"].cpu().numpy())

        val_acc = accuracy_score(val_true, val_preds)
        scheduler.step(1 - val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    # Test with best model
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            logits = model(batch)
            preds = logits.argmax(dim=-1).cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(batch["label"].cpu().numpy())

    test_acc = accuracy_score(test_true, test_preds)
    test_f1 = f1_score(test_true, test_preds, average="macro")

    return {
        "seed": seed,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "val_acc": best_val_acc,
        "train_size": len(train_ids),
        "val_size": len(val_ids),
        "test_size": len(test_ids),
        "epochs_trained": epoch + 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--prompt-family", required=True, choices=["hvguard", "mars", "diagnostic"])
    parser.add_argument("--fusion", default="gated", choices=["rationale_only", "concat", "gated"])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--project-root", default="/data/jehc223/EMNLP2")
    args = parser.parse_args()

    project_root = args.project_root
    cfg = DATASET_CONFIGS[args.dataset]

    # Load annotations
    ann_path = os.path.join(project_root, cfg["annotation"])
    with open(ann_path) as f:
        annotations = json.load(f)

    # Load features
    emb_dir = os.path.join(project_root, "embeddings", args.dataset)
    rat_path = os.path.join(emb_dir, f"rationale_features_{args.prompt_family}.npz")
    frame_path = os.path.join(emb_dir, "frame_features.npz")
    audio_path = os.path.join(emb_dir, "audio_features.npz")

    print(f"Loading features for {args.dataset} / {args.prompt_family} / {args.fusion}...")
    rationale_feats = dict(np.load(rat_path, allow_pickle=True))
    frame_feats = dict(np.load(frame_path, allow_pickle=True)) if os.path.exists(frame_path) else {}
    audio_feats = dict(np.load(audio_path, allow_pickle=True)) if os.path.exists(audio_path) else {}

    print(f"  Rationale: {len(rationale_feats)}, Frames: {len(frame_feats)}, Audio: {len(audio_feats)}")

    # Run multiple seeds
    results = []
    for seed in range(args.seeds):
        r = run_single_seed(args, seed, rationale_feats, frame_feats, audio_feats, annotations, cfg, project_root)
        results.append(r)
        print(f"  Seed {seed}: acc={r['test_acc']:.4f} f1={r['test_f1']:.4f}")

    # Summary
    accs = [r["test_acc"] for r in results]
    f1s = [r["test_f1"] for r in results]
    print(f"\n=== {args.dataset} / {args.prompt_family} / {args.fusion} ===")
    print(f"  Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f} (min={np.min(accs):.4f}, max={np.max(accs):.4f})")
    print(f"  F1:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f} (min={np.min(f1s):.4f}, max={np.max(f1s):.4f})")

    # Save results
    out_dir = os.path.join(project_root, "results", args.dataset, args.prompt_family)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.fusion}_results.json")
    summary = {
        "dataset": args.dataset,
        "prompt_family": args.prompt_family,
        "fusion": args.fusion,
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
        "min_acc": float(np.min(accs)),
        "max_acc": float(np.max(accs)),
        "per_seed": results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
