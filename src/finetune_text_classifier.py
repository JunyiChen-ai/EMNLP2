#!/usr/bin/env python3
"""Fine-tune DeBERTa on MLLM analysis text + transcript for hate classification.

Key insight: The MLLM analysis contains structured reasoning about visual content,
cross-modal interactions, and hate indicators. A fine-tuned text classifier can
learn to extract this information even when the MLLM's own classification is wrong.

This is one arm of the ensemble that combines:
1. Text classifier (MLLM analysis + transcript)  <-- this script
2. Multimodal classifier (audio + visual + text features + MLLM scores)
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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report


DATASET_CONFIGS = {
    "HateMM": {
        "mllm_results": "/data/jehc223/EMNLP2/results/mllm/HateMM/mllm_results.json",
        "mllm_scores": "/data/jehc223/EMNLP2/results/mllm/HateMM/mllm_scores.json",
        "label_file": "/data/jehc223/EMNLP2/baseline/HVGuard/datasets/HateMM/data (base).json",
        "splits_dir": "/data/jehc223/HateMM/splits",
        "label_map": {"Hate": 1, "Non Hate": 0},
    },
    "MultiHateClip_CN": {
        "mllm_results": "/data/jehc223/EMNLP2/results/mllm/MultiHateClip_CN/mllm_results.json",
        "mllm_scores": "/data/jehc223/EMNLP2/results/mllm/MultiHateClip_CN/mllm_scores.json",
        "label_file": "/data/jehc223/EMNLP2/baseline/HVGuard/datasets/Multihateclip/Chinese/data.json",
        "splits_dir": "/data/jehc223/Multihateclip/Chinese/splits",
        "label_map": {"Normal": 0, "Offensive": 1, "Hateful": 1},
    },
    "MultiHateClip_EN": {
        "mllm_results": "/data/jehc223/EMNLP2/results/mllm/MultiHateClip_EN/mllm_results.json",
        "mllm_scores": "/data/jehc223/EMNLP2/results/mllm/MultiHateClip_EN/mllm_scores.json",
        "label_file": "/data/jehc223/EMNLP2/baseline/HVGuard/datasets/Multihateclip/English/data.json",
        "splits_dir": "/data/jehc223/Multihateclip/English/splits",
        "label_map": {"Normal": 0, "Offensive": 1, "Hateful": 1},
    },
}


class TextDataset(Dataset):
    def __init__(self, video_ids, texts, labels, label_map, tokenizer, max_length=512):
        self.video_ids = video_ids
        self.texts = texts
        self.labels = labels
        self.label_map = label_map
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        text = self.texts[vid]
        label = self.label_map[self.labels[vid]]

        encoded = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def build_text_input(vid, mllm_results, mllm_scores, ann_data):
    """Combine MLLM analysis + transcript + title + scores into classifier input."""
    parts = []

    # MLLM analysis (from detailed analysis pass)
    if vid in mllm_results:
        analysis = mllm_results[vid].get("analysis", "")
        if analysis and not analysis.startswith("ERROR") and not analysis.startswith("[TEXT-ONLY"):
            parts.append(f"[VIDEO ANALYSIS] {analysis}")

    # MLLM scores summary (from scoring pass)
    if vid in mllm_scores:
        sc = mllm_scores[vid].get("scores", {})
        score_summary = (
            f"[SCORES] hate_speech={sc.get('hate_speech_score', '?')}/10 "
            f"visual_hate={sc.get('visual_hate_score', '?')}/10 "
            f"cross_modal={sc.get('cross_modal_score', '?')}/10 "
            f"implicit={sc.get('implicit_hate_score', '?')}/10 "
            f"overall={sc.get('overall_hate_score', '?')}/10 "
            f"confidence={sc.get('confidence', '?')}/10 "
            f"verdict={sc.get('classification', '?')}"
        )
        parts.append(score_summary)

    # Original transcript and title
    ann = ann_data.get(vid, {})
    title = ann.get("Title", "").strip()
    transcript = ann.get("Transcript", "").strip()
    if title:
        parts.append(f"[TITLE] {title}")
    if transcript:
        parts.append(f"[TRANSCRIPT] {transcript}")

    # HVGuard's MLLM description if available
    mix_desc = ann.get("Mix_description", "").strip()
    if mix_desc:
        parts.append(f"[HVGuard ANALYSIS] {mix_desc}")

    return " ".join(parts) if parts else "[NO CONTENT]"


def train_and_eval(ds_name, seed, args):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cfg = DATASET_CONFIGS[ds_name]

    # Load all data sources
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

    # Build text inputs
    texts = {}
    for vid in ann_data:
        texts[vid] = build_text_input(vid, mllm_results, mllm_scores, ann_data)

    # Load splits
    splits = {}
    for split in ["train", "valid", "test"]:
        with open(os.path.join(cfg["splits_dir"], split + ".csv")) as f:
            splits[split] = [l.strip() for l in f
                             if l.strip() in labels and labels[l.strip()] in cfg["label_map"]]

    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    ).to(device)

    train_ds = TextDataset(splits["train"], texts, labels, cfg["label_map"], tokenizer, args.max_length)
    valid_ds = TextDataset(splits["valid"], texts, labels, cfg["label_map"], tokenizer, args.max_length)
    test_ds = TextDataset(splits["test"], texts, labels, cfg["label_map"], tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False)

    # Class weights
    train_labels = [cfg["label_map"][labels[vid]] for vid in splits["train"]]
    counts = Counter(train_labels)
    total = sum(counts.values())
    class_weights = torch.tensor([total / (2 * counts[c]) for c in range(2)], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    best_val_acc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                val_preds.extend(outputs.logits.argmax(1).cpu().numpy())
                val_labels.extend(batch["labels"].numpy())
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
    test_preds, test_labels_list, test_logits = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            test_preds.extend(outputs.logits.argmax(1).cpu().numpy())
            test_labels_list.extend(batch["labels"].numpy())
            test_logits.extend(probs.cpu().numpy())

    test_acc = accuracy_score(test_labels_list, test_preds)
    test_f1 = f1_score(test_labels_list, test_preds, average="macro")
    print(f"  {ds_name} seed={seed}: val={best_val_acc:.3f} test={test_acc:.3f} ({test_acc*100:.1f}%) f1={test_f1:.3f}")

    # Save test predictions for ensemble
    pred_dict = {}
    for i, vid in enumerate(splits["test"]):
        pred_dict[vid] = {
            "pred": int(test_preds[i]),
            "prob": test_logits[i].tolist(),
            "true": int(test_labels_list[i]),
        }

    return test_acc, test_f1, pred_dict, best_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["HateMM", "MultiHateClip_CN", "MultiHateClip_EN"])
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--save_dir", default="/data/jehc223/EMNLP2/results/text_classifier")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    for ds in args.datasets:
        print(f"\n{'='*60}\n{ds}\n{'='*60}")
        accs, f1s = [], []
        all_preds = {}

        for seed in args.seeds:
            acc, f1, preds, state = train_and_eval(ds, seed, args)
            accs.append(acc)
            f1s.append(f1)
            all_preds[seed] = preds

        avg = np.mean(accs) * 100
        std = np.std(accs) * 100
        best = max(accs) * 100
        target = 90 if ds == "HateMM" else 85
        gap = target - avg
        status = "TARGET MET" if avg >= target else f"gap: {gap:.1f}%"
        print(f"\n  {ds}: avg={avg:.1f}% std={std:.1f}% best={best:.1f}% — {status}")

        # Save predictions for ensemble
        with open(os.path.join(args.save_dir, f"{ds}_predictions.json"), "w") as f:
            json.dump(all_preds, f)


if __name__ == "__main__":
    main()
