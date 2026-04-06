"""
Compare different MLLM prompt variants through the same WholeRationaleMLP classifier.
Also compute MLLM direct classification (training-free) from overall_judgment field.

Prompt variants: generic, scm, scm_v2, itt, iet, att
Datasets: HateMM, MHClip-EN (binary), MHClip-ZH (binary), ImpliHateVid
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

# ─── Label maps ────────────────────────────────────────────────────
LABEL_MAP_BINARY = {"Non Hate": 0, "Hate": 1}
LABEL_MAP_TERNARY_AS_BINARY = {"Normal": 0, "Offensive": 1, "Hateful": 1}


# ─── Dataset configs ───────────────────────────────────────────────
DATASET_CONFIGS = {
    "HateMM": {
        "emb_dir": "/home/junyi/EMNLP2/embeddings/HateMM",
        "ann_path": "/home/junyi/EMNLP2/datasets/HateMM/annotation(new).json",
        "split_dir": "/home/junyi/EMNLP2/datasets/HateMM/splits",
        "data_json": "/home/junyi/EMNLP2026/datasets/HateMM/generic_data.json",
    },
    "MHClip-EN": {
        "emb_dir": "/home/junyi/EMNLP2/embeddings/Multihateclip/English",
        "ann_path": "/home/junyi/EMNLP2/datasets/Multihateclip/English/annotation(new).json",
        "split_dir": "/home/junyi/EMNLP2/datasets/Multihateclip/English/splits",
        "data_json": "/home/junyi/EMNLP2026/datasets/Multihateclip/English/generic_data.json",
    },
    "MHClip-ZH": {
        "emb_dir": "/home/junyi/EMNLP2/embeddings/Multihateclip/Chinese",
        "ann_path": "/home/junyi/EMNLP2/datasets/Multihateclip/Chinese/annotation(new).json",
        "split_dir": "/home/junyi/EMNLP2/datasets/Multihateclip/Chinese/splits",
        "data_json": "/home/junyi/EMNLP2026/datasets/Multihateclip/Chinese/generic_data.json",
    },
    "ImpliHateVid": {
        "emb_dir": "/home/junyi/EMNLP2/embeddings/ImpliHateVid",
        "ann_path": "/home/junyi/EMNLP2/datasets/ImpliHateVid/annotation(new).json",
        "split_dir": "/home/junyi/EMNLP2/datasets/ImpliHateVid/splits",
        "data_json": "/home/junyi/EMNLP2026/datasets/ImpliHateVid/generic_data.json",
    },
}

PROMPT_VARIANTS = ["generic", "scm", "scm_v2", "itt", "iet", "att"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_labels(ann_path, force_binary=True):
    with open(ann_path, "r") as f:
        data = json.load(f)
    all_labels = set(d["Label"] for d in data)
    if all_labels <= {"Non Hate", "Hate"}:
        label_map = LABEL_MAP_BINARY
    elif all_labels <= {"Normal", "Offensive", "Hateful"}:
        label_map = LABEL_MAP_TERNARY_AS_BINARY if force_binary else None
    else:
        raise ValueError(f"Unknown labels: {all_labels}")
    return {d["Video_ID"]: label_map[d["Label"]] for d in data}


def load_split_ids(split_dir):
    import pandas as pd
    splits = {}
    for name in ["train", "valid", "test"]:
        df = pd.read_csv(Path(split_dir) / f"{name}.csv", header=None)
        splits[name] = df.iloc[:, 0].tolist()
    return splits


class SimpleDataset(Dataset):
    def __init__(self, ids, features, labels):
        valid = set(ids) & set(features.keys()) & set(labels.keys())
        self.ids = [v for v in ids if v in valid]
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        vid = self.ids[idx]
        return self.features[vid], self.labels[vid]


def collate(batch):
    feats = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return feats, labels


def train_and_eval_mlp(features, labels, splits, seed, num_classes=2):
    """Train WholeRationaleMLP on one prompt variant, return test acc & F1."""
    set_seed(seed)

    train_ds = SimpleDataset(splits["train"], features, labels)
    valid_ds = SimpleDataset(splits["valid"], features, labels)
    test_ds = SimpleDataset(splits["test"], features, labels)

    if len(train_ds) == 0 or len(test_ds) == 0:
        return None, None

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate)
    valid_dl = DataLoader(valid_ds, batch_size=32, shuffle=False, collate_fn=collate)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate)

    # Compute class weights
    counts = Counter(labels.values())
    total = sum(counts.values())
    weights = []
    for c in range(num_classes):
        freq = counts.get(c, 1)
        weights.append(total / (num_classes * freq))
    min_w = min(weights)
    class_weight = torch.tensor([w / min_w for w in weights], dtype=torch.float).to(DEVICE)

    model = nn.Sequential(
        nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
    criterion = nn.CrossEntropyLoss(weight=class_weight)

    total_steps = 50 * len(train_dl)
    warmup_steps = 5 * len(train_dl)

    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-2, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_acc = -1
    best_state = None
    no_improve = 0

    for epoch in range(50):
        model.train()
        for feats, labs in train_dl:
            feats, labs = feats.to(DEVICE), labs.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(feats), labs)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for feats, labs in valid_dl:
                preds.extend(model(feats.to(DEVICE)).argmax(1).cpu().numpy())
                gts.extend(labs.numpy())
        val_acc = accuracy_score(gts, preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 10:
                break

    model.load_state_dict(best_state)
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for feats, labs in test_dl:
            preds.extend(model(feats.to(DEVICE)).argmax(1).cpu().numpy())
            gts.extend(labs.numpy())

    acc = round(accuracy_score(gts, preds) * 100, 2)
    f1 = round(f1_score(gts, preds, average="macro") * 100, 2)
    return acc, f1


def compute_direct_classification(data_json, ann_path, split_dir, force_binary=True):
    """Compute MLLM direct classification from overall_judgment field on test set."""
    if not Path(data_json).exists():
        return None, None

    with open(data_json, "r") as f:
        data = json.load(f)

    labels = load_labels(ann_path, force_binary=force_binary)
    splits = load_split_ids(split_dir)
    test_ids = set(splits["test"])

    # Build prediction map from overall_judgment
    preds_map = {}
    for item in data:
        vid = item["Video_ID"]
        if vid not in test_ids:
            continue
        resp = item.get("generic_response", {})
        judgment = resp.get("overall_judgment", "").strip().lower()
        # Map to binary: yes/hate -> 1, no/non-hate -> 0
        if judgment in ("yes", "hate", "hateful", "true"):
            preds_map[vid] = 1
        elif judgment in ("no", "non hate", "non-hate", "normal", "false"):
            preds_map[vid] = 0
        else:
            # Try to parse
            if "yes" in judgment:
                preds_map[vid] = 1
            elif "no" in judgment:
                preds_map[vid] = 0
            else:
                preds_map[vid] = 1  # default to hate if unclear

    # Match with ground truth
    valid_ids = set(preds_map.keys()) & set(labels.keys()) & test_ids
    if len(valid_ids) < 5:
        return None, None

    y_true = [labels[vid] for vid in valid_ids]
    y_pred = [preds_map[vid] for vid in valid_ids]

    acc = round(accuracy_score(y_true, y_pred) * 100, 2)
    f1 = round(f1_score(y_true, y_pred, average="macro") * 100, 2)
    return acc, f1


def main():
    print("=" * 80)
    print("PROMPT VARIANT COMPARISON: MLLM Direct + MLP Classifier (10 seeds)")
    print("=" * 80)

    all_results = {}

    for ds_name, cfg in DATASET_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        labels = load_labels(cfg["ann_path"], force_binary=True)
        splits = load_split_ids(cfg["split_dir"])
        num_classes = max(labels.values()) + 1

        # 1. MLLM Direct Classification (training-free)
        direct_acc, direct_f1 = compute_direct_classification(
            cfg["data_json"], cfg["ann_path"], cfg["split_dir"], force_binary=True
        )
        print(f"\n  MLLM Direct (overall_judgment): ACC={direct_acc}, F1={direct_f1}")

        ds_results = {"direct": {"acc": direct_acc, "f1": direct_f1}}

        # 2. Each prompt variant through MLP
        for variant in PROMPT_VARIANTS:
            feat_path = Path(cfg["emb_dir"]) / f"{variant}_rationale_features.pth"
            if not feat_path.exists():
                print(f"  {variant}: SKIPPED (no feature file)")
                ds_results[variant] = None
                continue

            features = torch.load(feat_path, map_location="cpu")

            accs, f1s = [], []
            for seed in SEEDS:
                acc, f1 = train_and_eval_mlp(features, labels, splits, seed, num_classes)
                if acc is not None:
                    accs.append(acc)
                    f1s.append(f1)

            if accs:
                mean_acc = np.mean(accs)
                std_acc = np.std(accs)
                mean_f1 = np.mean(f1s)
                std_f1 = np.std(f1s)
                worst_f1 = np.min(f1s)
                best_f1 = np.max(f1s)
                print(f"  {variant:10s}: ACC={mean_acc:.2f}±{std_acc:.2f}  "
                      f"F1={mean_f1:.2f}±{std_f1:.2f}  "
                      f"[worst={worst_f1:.2f}, best={best_f1:.2f}]")
                ds_results[variant] = {
                    "mean_acc": round(mean_acc, 2),
                    "std_acc": round(std_acc, 2),
                    "mean_f1": round(mean_f1, 2),
                    "std_f1": round(std_f1, 2),
                    "worst_f1": round(worst_f1, 2),
                    "best_f1": round(best_f1, 2),
                }
            else:
                print(f"  {variant}: FAILED (no valid runs)")
                ds_results[variant] = None

        all_results[ds_name] = ds_results

    # ─── Print summary tables ──────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("SUMMARY TABLES")
    print(f"{'='*80}")

    for ds_name, ds_results in all_results.items():
        print(f"\n### {ds_name}")
        print(f"| Prompt | Mean ACC | Std ACC | Mean F1 | Std F1 | Worst F1 | Best F1 |")
        print(f"|--------|----------|---------|---------|--------|----------|---------|")

        # Direct classification
        d = ds_results.get("direct", {})
        if d and d.get("acc") is not None:
            print(f"| MLLM direct | {d['acc']} | — | {d['f1']} | — | — | — |")

        # MLP variants
        for variant in PROMPT_VARIANTS:
            r = ds_results.get(variant)
            if r is None:
                print(f"| {variant} | — | — | — | — | — | — |")
            else:
                print(f"| {variant} | {r['mean_acc']:.2f} | {r['std_acc']:.2f} | "
                      f"{r['mean_f1']:.2f} | {r['std_f1']:.2f} | "
                      f"{r['worst_f1']:.2f} | {r['best_f1']:.2f} |")

    # Save
    with open("kill_test/results/gpt_experiments/prompt_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to kill_test/results/gpt_experiments/prompt_comparison.json")


if __name__ == "__main__":
    main()
