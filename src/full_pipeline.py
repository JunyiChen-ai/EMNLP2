#!/usr/bin/env python3
"""Full CMEV Pipeline: Encode rationales + Parse MLLM classifications + Train classifier.

This script runs the complete pipeline after MLLM inference is done:
1. Encode MLLM rationales into embeddings
2. Parse MLLM classifications from analysis text
3. Train the CMEV classifier with all signals
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModel


# ─── Constants ──────────────────────────────────────────────────

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


# ─── Step 1: Encode Rationales ──────────────────────────────────

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask, 1) / torch.clamp(input_mask.sum(1), min=1e-9)


def encode_rationales(mllm_results_dir, output_dir, encoder_name="bert-base-uncased",
                      datasets=None, batch_size=32, max_length=512):
    """Encode MLLM analysis text into embeddings."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    model = AutoModel.from_pretrained(encoder_name).to(device)
    model.eval()

    for ds in datasets:
        result_path = os.path.join(mllm_results_dir, ds, "mllm_results.json")
        if not os.path.exists(result_path):
            print(f"  Skipping {ds}: no MLLM results")
            continue

        with open(result_path) as f:
            results = json.load(f)

        video_ids = list(results.keys())
        texts = [results[vid].get("analysis", results[vid].get("transcript", "")) for vid in video_ids]

        # Encode in batches
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                encoded = tokenizer(batch, padding=True, truncation=True,
                                    max_length=max_length, return_tensors="pt").to(device)
                outputs = model(**encoded)
                embs = mean_pooling(outputs, encoded["attention_mask"])
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                all_embeddings.append(embs.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0)
        embedding_dict = {vid: emb for vid, emb in zip(video_ids, all_embeddings)}

        out_dir = os.path.join(output_dir, ds)
        os.makedirs(out_dir, exist_ok=True)
        torch.save(embedding_dict, os.path.join(out_dir, "mllm_rationale_features.pth"))
        print(f"  {ds}: encoded {len(embedding_dict)} rationales ({all_embeddings.shape[1]}d)")


# ─── Step 2: Parse MLLM Classifications ─────────────────────────

def parse_classification(analysis: str) -> dict:
    """Extract classification + confidence from MLLM analysis."""
    analysis_lower = analysis.lower()

    # Split into sections
    sections = re.split(r'\n\s*\d+\.', analysis)
    hate_section = ""
    for section in sections:
        if any(kw in section.lower() for kw in ["hate assessment", "classification", "verdict"]):
            hate_section = section
            break
    if not hate_section and len(sections) >= 4:
        hate_section = sections[3] if len(sections) > 3 else sections[-1]

    text = (hate_section.lower() if hate_section else analysis_lower[-1000:])

    # Check for "not hateful/offensive" first (negation)
    negation = bool(re.search(
        r'(not|no|neither|does not|doesn\'t)\s+(hateful|offensive|hate\s*speech|promoting\s+hate|contain)',
        text))

    # Check for hateful indicators
    hateful = bool(re.search(
        r'(hateful|offensive|hate\s*speech|promoting\s+hate|derogatory|discriminat)',
        text))

    # Check for normal indicators
    normal = bool(re.search(
        r'(normal|benign|harmless|non[-\s]?hat|not.{0,20}(hateful|offensive))',
        text))

    # Confidence
    confidence = 0.5
    conf_text = analysis_lower
    conf_match = re.search(r'(\d+)\s*/\s*10|(\d+)\s*out of\s*10|confidence[:\s]+(\d+)', conf_text)
    if conf_match:
        val = int(next(g for g in conf_match.groups() if g is not None))
        confidence = min(val / 10.0, 1.0)
    elif "very confident" in conf_text or "high confidence" in conf_text or "highly confident" in conf_text:
        confidence = 0.9
    elif "confident" in conf_text and "not" not in conf_text[:conf_text.index("confident")+20]:
        confidence = 0.75
    elif "moderate" in conf_text:
        confidence = 0.6
    elif "uncertain" in conf_text or "low confidence" in conf_text:
        confidence = 0.4

    if negation and not (hateful and not normal):
        pred = "Normal"
    elif hateful and not normal:
        pred = "Hateful"
    elif normal and not hateful:
        pred = "Normal"
    elif hateful and normal:
        # Ambiguous - check context more carefully
        # If negation pattern found, lean normal
        if negation:
            pred = "Normal"
        else:
            pred = "Hateful"
    else:
        pred = "Unknown"

    return {"prediction": pred, "confidence": confidence}


def parse_all_classifications(mllm_results_dir, datasets):
    """Parse MLLM classifications for all datasets."""
    all_classifications = {}
    for ds in datasets:
        result_path = os.path.join(mllm_results_dir, ds, "mllm_results.json")
        if not os.path.exists(result_path):
            continue

        with open(result_path) as f:
            results = json.load(f)

        classifications = {}
        for vid, res in results.items():
            cls = parse_classification(res.get("analysis", ""))
            classifications[vid] = cls

        all_classifications[ds] = classifications

        # Report accuracy
        label_map = DATASET_CONFIGS[ds]["label_map"]
        correct = total = 0
        for vid, cls in classifications.items():
            true_label = res.get("label", "") if vid == list(results.keys())[-1] else results[vid].get("label", "")
            true_label = results[vid].get("label", "")
            if true_label in label_map:
                total += 1
                pred_binary = 1 if cls["prediction"] == "Hateful" else 0
                true_binary = label_map[true_label]
                if pred_binary == true_binary:
                    correct += 1
        if total:
            print(f"  {ds} MLLM classification: {correct}/{total} = {correct/total*100:.1f}%")

    return all_classifications


# ─── Step 3: Dataset & Model ────────────────────────────────────

class CMEVDataset(Dataset):
    """Dataset with all features + MLLM classification signal."""

    def __init__(self, video_ids, text_feats, audio_feats, frame_feats,
                 hvg_mllm_feats, our_mllm_feats, mllm_classifications,
                 labels, label_map):
        self.video_ids = video_ids
        self.text = text_feats
        self.audio = audio_feats
        self.frame = frame_feats
        self.hvg_mllm = hvg_mllm_feats
        self.our_mllm = our_mllm_feats
        self.mllm_cls = mllm_classifications or {}
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

        # MLLM classification features: [pred_hateful, pred_normal, confidence]
        cls = self.mllm_cls.get(vid, {"prediction": "Unknown", "confidence": 0.5})
        mllm_pred = torch.tensor([
            1.0 if cls["prediction"] == "Hateful" else 0.0,
            1.0 if cls["prediction"] == "Normal" else 0.0,
            cls["confidence"],
        ], dtype=torch.float)

        label = self.label_map[self.labels[vid]]
        return text, audio, frame, hvg_mllm, our_mllm, mllm_pred, torch.tensor(label, dtype=torch.long)


class CrossModalAttention(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.k_proj = nn.Linear(key_dim, hidden_dim)
        self.v_proj = nn.Linear(key_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, keys):
        q = self.q_proj(query).unsqueeze(1)
        k = self.k_proj(keys)
        v = self.v_proj(keys)
        out, attn_weights = self.attn(q, k, v)
        return self.norm(out.squeeze(1)), attn_weights.squeeze(1)


class CMEVClassifier(nn.Module):
    """Cross-Modal Evidence Verification Classifier with MLLM classification signal."""

    def __init__(self, feat_dim=768, hidden_dim=256, num_heads=4, num_classes=2,
                 dropout=0.3, use_our_mllm=True, use_mllm_cls=True):
        super().__init__()
        self.use_our_mllm = use_our_mllm
        self.use_mllm_cls = use_mllm_cls

        # Modality projections
        self.text_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout * 0.3))
        self.audio_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout * 0.3))
        self.frame_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout * 0.3))

        # MLLM rationale projection
        mllm_dim = feat_dim * 2 if use_our_mllm else feat_dim
        self.mllm_proj = nn.Sequential(
            nn.Linear(mllm_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout * 0.3))

        # Cross-modal attention
        self.cross_attn = CrossModalAttention(hidden_dim, hidden_dim, hidden_dim, num_heads)

        # Self-attention over modality representations
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.1)
        self.self_norm = nn.LayerNorm(hidden_dim)

        # Gated fusion
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())

        # MLLM classification signal integration
        cls_input_dim = hidden_dim + (3 if use_mllm_cls else 0)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(cls_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, text, audio, frame, hvg_mllm, our_mllm, mllm_pred):
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

        # Stack modalities for attention
        modalities = torch.stack([text_h, audio_h, frame_h], dim=1)  # (B, 3, H)

        # Cross-attention: MLLM queries evidence
        attended, attn_weights = self.cross_attn(mllm_h, modalities)

        # Self-attention over modalities for inter-modal reasoning
        self_attended, _ = self.self_attn(modalities, modalities, modalities)
        self_attended = self.self_norm(self_attended + modalities)
        pooled = self_attended.mean(dim=1)  # (B, H)

        # Gated fusion
        gate_input = torch.cat([attended, pooled], dim=-1)
        gate_weight = self.gate(gate_input)
        fused = gate_weight * attended + (1 - gate_weight) * pooled

        # Add MLLM classification signal
        if self.use_mllm_cls:
            fused = torch.cat([fused, mllm_pred], dim=-1)

        return self.classifier(fused)


# ─── Training ────────────────────────────────────────────────────

def load_features(ds_name, our_mllm_dir, mllm_classifications):
    cfg = DATASET_CONFIGS[ds_name]

    text = torch.load(f"{cfg['hvguard_emb']}/text_features.pth", map_location="cpu", weights_only=True)
    audio = torch.load(f"{cfg['hvguard_emb']}/audio_features.pth", map_location="cpu", weights_only=True)
    frame = torch.load(f"{cfg['hvguard_emb']}/frame_features.pth", map_location="cpu", weights_only=True)
    hvg_mllm = torch.load(f"{cfg['hvguard_emb']}/MLLM_rationale_features.pth", map_location="cpu", weights_only=True)

    our_mllm = None
    if our_mllm_dir:
        path = os.path.join(our_mllm_dir, ds_name, "mllm_rationale_features.pth")
        if os.path.exists(path):
            our_mllm = torch.load(path, map_location="cpu", weights_only=True)

    with open(cfg["label_file"]) as f:
        raw = json.load(f)
    labels = {d["Video_ID"]: d["Label"] for d in raw}

    splits = {}
    for split in ["train", "valid", "test"]:
        with open(f"{cfg['splits_dir']}/{split}.csv") as f:
            ids = [l.strip() for l in f if l.strip() and l.strip() in labels
                   and labels[l.strip()] in cfg["label_map"]]
        splits[split] = ids

    cls = mllm_classifications.get(ds_name, {})

    return text, audio, frame, hvg_mllm, our_mllm, cls, labels, cfg["label_map"], splits


def train_one(ds_name, seed, our_mllm_dir, mllm_classifications, args):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    text, audio, frame, hvg_mllm, our_mllm, cls, labels, label_map, splits = \
        load_features(ds_name, our_mllm_dir, mllm_classifications)

    use_our_mllm = our_mllm is not None
    use_mllm_cls = len(cls) > 0

    train_ds = CMEVDataset(splits["train"], text, audio, frame, hvg_mllm, our_mllm, cls, labels, label_map)
    valid_ds = CMEVDataset(splits["valid"], text, audio, frame, hvg_mllm, our_mllm, cls, labels, label_map)
    test_ds = CMEVDataset(splits["test"], text, audio, frame, hvg_mllm, our_mllm, cls, labels, label_map)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CMEVClassifier(
        hidden_dim=args.hidden_dim, dropout=args.dropout,
        use_our_mllm=use_our_mllm, use_mllm_cls=use_mllm_cls
    ).to(device)

    # Class-balanced weights
    train_labels = [label_map[labels[vid]] for vid in splits["train"]]
    counts = Counter(train_labels)
    total = sum(counts.values())
    weights = torch.tensor([total / (2 * counts[c]) for c in range(2)], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_val_acc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            t, a, f, hm, om, mp, y = [b.to(device) for b in batch]
            optimizer.zero_grad()
            out = model(t, a, f, hm, om, mp)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in valid_loader:
                t, a, f, hm, om, mp, y = [b.to(device) for b in batch]
                out = model(t, a, f, hm, om, mp)
                val_preds.extend(out.argmax(1).cpu().numpy())
                val_labels.extend(y.cpu().numpy())
        val_acc = accuracy_score(val_labels, val_preds)

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
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            t, a, f, hm, om, mp, y = [b.to(device) for b in batch]
            out = model(t, a, f, hm, om, mp)
            test_preds.extend(out.argmax(1).cpu().numpy())
            test_labels.extend(y.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average="macro")
    print(f"  {ds_name} seed={seed}: val={best_val_acc:.3f} test_acc={test_acc:.3f} ({test_acc*100:.1f}%) "
          f"test_f1={test_f1:.3f} [our_mllm={use_our_mllm}, mllm_cls={use_mllm_cls}]")
    return test_acc, test_f1, best_state, test_preds, test_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["HateMM", "MultiHateClip_CN", "MultiHateClip_EN"])
    parser.add_argument("--mllm_results_dir", default="/data/jehc223/EMNLP2/results/mllm")
    parser.add_argument("--our_mllm_dir", default="/data/jehc223/EMNLP2/embeddings")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1024])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--save_dir", default="/data/jehc223/EMNLP2/results/classifier")
    parser.add_argument("--skip_encode", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Step 1: Encode rationales (if not already done)
    if not args.skip_encode:
        print("Step 1: Encoding MLLM rationales...")
        encode_rationales(args.mllm_results_dir, args.our_mllm_dir, datasets=args.datasets)

    # Step 2: Parse MLLM classifications
    print("\nStep 2: Parsing MLLM classifications...")
    mllm_cls = parse_all_classifications(args.mllm_results_dir, args.datasets)

    # Step 3: Train and evaluate
    print("\nStep 3: Training CMEV classifier...")
    all_results = {}
    for ds in args.datasets:
        print(f"\n{'='*60}\n{ds}\n{'='*60}")
        accs, f1s = [], []
        best_acc = 0
        best_state = None

        for seed in args.seeds:
            acc, f1, state, _, _ = train_one(ds, seed, args.our_mllm_dir, mllm_cls, args)
            accs.append(acc)
            f1s.append(f1)
            if acc > best_acc:
                best_acc = acc
                best_state = state

        avg_acc = np.mean(accs) * 100
        std_acc = np.std(accs) * 100
        print(f"\n  {ds}: avg={avg_acc:.1f}% std={std_acc:.1f}% best={best_acc*100:.1f}%")
        all_results[ds] = {
            "avg_acc": avg_acc, "std_acc": std_acc,
            "best_acc": best_acc * 100, "avg_f1": np.mean(f1s) * 100,
            "all_accs": [a * 100 for a in accs],
        }
        torch.save(best_state, os.path.join(args.save_dir, f"{ds}_best.pth"))

    # Summary
    with open(os.path.join(args.save_dir, "results_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}\nOVERALL SUMMARY:")
    for ds, r in all_results.items():
        target = 90 if ds == "HateMM" else 85
        gap = target - r["avg_acc"]
        status = "TARGET MET" if r["avg_acc"] >= target else f"gap: {gap:.1f}%"
        print(f"  {ds}: {r['avg_acc']:.1f}% +/- {r['std_acc']:.1f}% [best: {r['best_acc']:.1f}%] — {status}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
