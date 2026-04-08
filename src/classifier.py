#!/usr/bin/env python3
"""Cross-Modal Evidence Verification (CMEV) Classifier.

Scientific motivation: MLLMs can generate rich semantic understanding of video
content but suffer from hallucination and limited discriminative ability. Raw
multimodal features (text, audio, visual) capture modality-specific signals but
lack semantic reasoning. CMEV uses MLLM rationale as a query to selectively
attend to evidence in raw modality features, grounding the reasoning in actual
multimodal evidence. This addresses the key challenge in hateful video detection
where hate cues often emerge from cross-modal interactions that are missed when
modalities are simply concatenated.

Architecture:
  1. Modality-specific projection layers (768 -> hidden_dim per modality)
  2. MLLM rationale as cross-attention query over projected modality features
  3. Gated fusion of attended features
  4. Classification head
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ─── Dataset ─────────────────────────────────────────────────────

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


class MultiModalDataset(Dataset):
    """Dataset combining HVGuard features with our MLLM rationale embeddings."""

    def __init__(self, video_ids, text_feats, audio_feats, frame_feats,
                 hvguard_mllm_feats, our_mllm_feats, labels, label_map):
        self.video_ids = video_ids
        self.text_feats = text_feats
        self.audio_feats = audio_feats
        self.frame_feats = frame_feats
        self.hvguard_mllm = hvguard_mllm_feats
        self.our_mllm = our_mllm_feats
        self.labels = labels
        self.label_map = label_map
        self.feat_dim = 768

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        zero = torch.zeros(self.feat_dim)

        text = self.text_feats.get(vid, zero)
        audio = self.audio_feats.get(vid, zero)
        frame = self.frame_feats.get(vid, zero)
        hvg_mllm = self.hvguard_mllm.get(vid, zero)
        our_mllm = self.our_mllm.get(vid, zero) if self.our_mllm else zero

        label = self.label_map[self.labels[vid]]
        return text, audio, frame, hvg_mllm, our_mllm, torch.tensor(label, dtype=torch.long)


# ─── Model ───────────────────────────────────────────────────────

class CrossModalAttention(nn.Module):
    """Cross-attention: MLLM rationale queries attend to modality features."""

    def __init__(self, query_dim, key_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(key_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, keys):
        """
        query: (B, query_dim) - MLLM rationale
        keys: (B, N, key_dim) - N modality features stacked
        Returns: (B, hidden_dim) - attended representation
        """
        q = self.query_proj(query).unsqueeze(1)  # (B, 1, H)
        k = self.key_proj(keys)  # (B, N, H)
        v = self.value_proj(keys)  # (B, N, H)
        attn_out, _ = self.attn(q, k, v)  # (B, 1, H)
        return self.norm(attn_out.squeeze(1))  # (B, H)


class CMEVClassifier(nn.Module):
    """Cross-Modal Evidence Verification Classifier.

    Uses MLLM rationale as semantic query to attend to raw multimodal evidence,
    then fuses the grounded representation with direct modality features for
    classification.
    """

    def __init__(self, feat_dim=768, hidden_dim=256, num_heads=4, num_classes=2,
                 dropout=0.3, use_our_mllm=True):
        super().__init__()
        self.use_our_mllm = use_our_mllm

        # Modality projections
        self.text_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout * 0.5))
        self.audio_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout * 0.5))
        self.frame_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout * 0.5))

        # MLLM rationale dimension depends on whether we use both
        mllm_dim = feat_dim * 2 if use_our_mllm else feat_dim
        self.mllm_proj = nn.Sequential(
            nn.Linear(mllm_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout * 0.5))

        # Cross-modal attention: MLLM reasoning queries raw modality evidence
        self.cross_attn = CrossModalAttention(hidden_dim, hidden_dim, hidden_dim, num_heads)

        # Gated fusion
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, text, audio, frame, hvg_mllm, our_mllm):
        # Project modalities
        text_h = self.text_proj(text)    # (B, H)
        audio_h = self.audio_proj(audio)  # (B, H)
        frame_h = self.frame_proj(frame)  # (B, H)

        # Project MLLM rationale
        if self.use_our_mllm:
            mllm_input = torch.cat([hvg_mllm, our_mllm], dim=-1)
        else:
            mllm_input = hvg_mllm
        mllm_h = self.mllm_proj(mllm_input)  # (B, H)

        # Stack raw modality features for cross-attention
        modality_stack = torch.stack([text_h, audio_h, frame_h], dim=1)  # (B, 3, H)

        # Cross-attention: MLLM reasoning queries raw evidence
        attended = self.cross_attn(mllm_h, modality_stack)  # (B, H)

        # Gated fusion of attended evidence and direct MLLM representation
        gate_input = torch.cat([attended, mllm_h], dim=-1)
        gate_weight = self.gate(gate_input)
        fused = gate_weight * attended + (1 - gate_weight) * mllm_h  # (B, H)

        return self.classifier(fused)


# ─── Training ────────────────────────────────────────────────────

def load_data(dataset_name, our_mllm_dir=None):
    """Load all features and labels for a dataset."""
    cfg = DATASET_CONFIGS[dataset_name]

    # HVGuard features
    text_feats = torch.load(f"{cfg['hvguard_emb']}/text_features.pth",
                            map_location="cpu", weights_only=True)
    audio_feats = torch.load(f"{cfg['hvguard_emb']}/audio_features.pth",
                             map_location="cpu", weights_only=True)
    frame_feats = torch.load(f"{cfg['hvguard_emb']}/frame_features.pth",
                             map_location="cpu", weights_only=True)
    hvg_mllm_feats = torch.load(f"{cfg['hvguard_emb']}/MLLM_rationale_features.pth",
                                map_location="cpu", weights_only=True)

    # Our MLLM rationale features
    our_mllm_feats = None
    if our_mllm_dir:
        our_path = os.path.join(our_mllm_dir, dataset_name, "mllm_rationale_features.pth")
        if os.path.exists(our_path):
            our_mllm_feats = torch.load(our_path, map_location="cpu", weights_only=True)
            print(f"  Loaded our MLLM features: {len(our_mllm_feats)} samples")

    # Labels
    with open(cfg["label_file"]) as f:
        raw = json.load(f)
    labels = {d["Video_ID"]: d["Label"] for d in raw}

    # Splits
    splits = {}
    for split in ["train", "valid", "test"]:
        fpath = f"{cfg['splits_dir']}/{split}.csv"
        ids = []
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line and line in labels and labels[line] in cfg["label_map"]:
                    ids.append(line)
        splits[split] = ids

    return {
        "text": text_feats,
        "audio": audio_feats,
        "frame": frame_feats,
        "hvg_mllm": hvg_mllm_feats,
        "our_mllm": our_mllm_feats,
        "labels": labels,
        "label_map": cfg["label_map"],
        "splits": splits,
    }


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for text, audio, frame, hvg_mllm, our_mllm, labels in loader:
        text, audio, frame = text.to(device), audio.to(device), frame.to(device)
        hvg_mllm, our_mllm = hvg_mllm.to(device), our_mllm.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        out = model(text, audio, frame, hvg_mllm, our_mllm)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for text, audio, frame, hvg_mllm, our_mllm, labels in loader:
        text, audio, frame = text.to(device), audio.to(device), frame.to(device)
        hvg_mllm, our_mllm = hvg_mllm.to(device), our_mllm.to(device)
        out = model(text, audio, frame, hvg_mllm, our_mllm)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1, all_preds, all_labels


def train_and_eval(dataset_name, seed=42, our_mllm_dir=None, epochs=50,
                   lr=1e-4, hidden_dim=256, dropout=0.3, weight_decay=1e-5,
                   batch_size=32, patience=10):
    """Train and evaluate the CMEV classifier on a single dataset."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data = load_data(dataset_name, our_mllm_dir)
    use_our_mllm = data["our_mllm"] is not None

    # Create datasets
    train_ds = MultiModalDataset(
        data["splits"]["train"], data["text"], data["audio"], data["frame"],
        data["hvg_mllm"], data["our_mllm"], data["labels"], data["label_map"])
    valid_ds = MultiModalDataset(
        data["splits"]["valid"], data["text"], data["audio"], data["frame"],
        data["hvg_mllm"], data["our_mllm"], data["labels"], data["label_map"])
    test_ds = MultiModalDataset(
        data["splits"]["test"], data["text"], data["audio"], data["frame"],
        data["hvg_mllm"], data["our_mllm"], data["labels"], data["label_map"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CMEVClassifier(
        hidden_dim=hidden_dim, dropout=dropout,
        use_our_mllm=use_our_mllm, num_classes=2
    ).to(device)

    # Class weights for imbalanced data
    train_labels = [data["label_map"][data["labels"][vid]] for vid in data["splits"]["train"]]
    class_counts = [train_labels.count(c) for c in range(2)]
    total = sum(class_counts)
    class_weights = torch.tensor([total / (2 * c) for c in class_counts], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    best_model_state = None
    no_improve = 0

    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        val_acc, val_f1, _, _ = evaluate(model, valid_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Test with best model
    model.load_state_dict(best_model_state)
    model.to(device)
    test_acc, test_f1, test_preds, test_labels = evaluate(model, test_loader, device)

    print(f"\n{dataset_name} (seed={seed}, use_our_mllm={use_our_mllm}):")
    print(f"  Val acc: {best_val_acc:.4f}")
    print(f"  Test acc: {test_acc:.4f} ({test_acc * 100:.1f}%)")
    print(f"  Test F1 (macro): {test_f1:.4f}")
    print(classification_report(test_labels, test_preds, target_names=["Normal", "Hateful"]))
    return test_acc, test_f1, best_model_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["HateMM", "MultiHateClip_CN", "MultiHateClip_EN"])
    parser.add_argument("--our_mllm_dir", default="/data/jehc223/EMNLP2/embeddings")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1024])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--save_dir", default="/data/jehc223/EMNLP2/results/classifier")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    all_results = {}
    for ds in args.datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds}")
        print(f"{'=' * 60}")

        accs, f1s = [], []
        best_acc = 0
        best_state = None

        for seed in args.seeds:
            acc, f1, state = train_and_eval(
                ds, seed=seed, our_mllm_dir=args.our_mllm_dir,
                epochs=args.epochs, lr=args.lr, hidden_dim=args.hidden_dim,
                dropout=args.dropout, batch_size=args.batch_size,
                patience=args.patience)
            accs.append(acc)
            f1s.append(f1)
            if acc > best_acc:
                best_acc = acc
                best_state = state

        avg_acc = np.mean(accs) * 100
        std_acc = np.std(accs) * 100
        avg_f1 = np.mean(f1s) * 100
        print(f"\n{ds} Summary:")
        print(f"  Avg acc: {avg_acc:.1f}% (+/- {std_acc:.1f}%)")
        print(f"  Best acc: {best_acc * 100:.1f}%")
        print(f"  Avg F1: {avg_f1:.1f}%")

        all_results[ds] = {
            "avg_acc": avg_acc,
            "std_acc": std_acc,
            "best_acc": best_acc * 100,
            "avg_f1": avg_f1,
            "all_accs": [a * 100 for a in accs],
        }

        # Save best model
        torch.save(best_state, os.path.join(args.save_dir, f"{ds}_best.pth"))

    # Save summary
    with open(os.path.join(args.save_dir, "results_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("OVERALL SUMMARY:")
    for ds, res in all_results.items():
        print(f"  {ds}: {res['avg_acc']:.1f}% (+/- {res['std_acc']:.1f}%) [best: {res['best_acc']:.1f}%]")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
