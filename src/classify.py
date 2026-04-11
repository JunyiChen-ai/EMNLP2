"""
Step 6: Aggregate precondition scores into final verdict + evaluate.

For each video:
  - A rule is violated if ALL its preconditions are satisfied.
  - A video is Hateful if ANY rule is violated.

Usage:
  python src/classify.py --dataset HateMM
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter

import numpy as np

PROJECT_ROOT = "/data/jehc223/EMNLP2"

DATASET_CONSTITUTION = {
    "HateMM": "youtube",
    "MHClip_EN": "youtube",
    "MHClip_ZH": "bilibili",
}

DATASET_CONFIGS = {
    "HateMM": {
        "annotation": "datasets/HateMM/annotation(new).json",
        "splits_dir": "datasets/HateMM/splits",
        "id_field": "Video_ID",
        "label_field": "Label",
        "label_map": {"Hate": 1, "Non Hate": 0},
    },
    "MHClip_EN": {
        "annotation": "datasets/MHClip_EN/annotation(new).json",
        "splits_dir": "datasets/MHClip_EN/splits",
        "id_field": "Video_ID",
        "label_field": "Label",
        "label_map": {"Hateful": 1, "Offensive": 1, "Normal": 0},
    },
    "MHClip_ZH": {
        "annotation": "datasets/MHClip_ZH/annotation(new).json",
        "splits_dir": "datasets/MHClip_ZH/splits",
        "id_field": "Video_ID",
        "label_field": "Label",
        "label_map": {"Hateful": 1, "Offensive": 1, "Normal": 0},
    },
}


def load_scores(scores_path):
    """Load per-video precondition scores from JSONL, skipping metadata header."""
    records = []
    with open(scores_path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec.get("_metadata"):
                    continue
                records.append(rec)
    return {r["video_id"]: r for r in records}


def load_gt_labels(cfg, split, project_root):
    """Load ground-truth labels for a split."""
    with open(os.path.join(project_root, cfg["annotation"])) as f:
        annotations = json.load(f)
    id2label = {}
    for s in annotations:
        vid_id = s[cfg["id_field"]]
        label_str = s.get(cfg["label_field"], "")
        if label_str in cfg["label_map"]:
            id2label[vid_id] = cfg["label_map"][label_str]

    split_path = os.path.join(project_root, cfg["splits_dir"], f"{split}.csv")
    with open(split_path) as f:
        split_ids = [line.strip() for line in f if line.strip()]

    return {vid: id2label[vid] for vid in split_ids if vid in id2label}


def classify_video(video_record, rules):
    """Apply AND/ANY aggregation. Returns (hateful: bool, violated_rules: list)."""
    violated = []
    for rule in rules:
        rule_id = rule["rule_id"]
        rule_data = video_record.get("rules", {}).get(rule_id, {})

        if not rule_data.get("relevant", False):
            continue

        if rule_data.get("all_satisfied", False):
            violated.append(rule_id)

    return len(violated) > 0, violated


def compute_metrics(y_true, y_pred):
    """Compute ACC, precision, recall, F1, F1-macro."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def main():
    parser = argparse.ArgumentParser(description="Classify and evaluate")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--constitution-suffix", default="",
                        help="Suffix for constitution file, e.g. '_merged'")
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    root = args.project_root
    constitution_name = DATASET_CONSTITUTION[args.dataset]
    out_suffix = args.constitution_suffix

    # Setup logging
    log_dir = os.path.join(root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"classify_{args.dataset}_{args.split}{out_suffix}.log")),
            logging.StreamHandler(),
        ],
    )

    # Load precondition structure (for rule list)
    precond_path = os.path.join(root, "constitution", f"preconditions_{constitution_name}{args.constitution_suffix}.json")
    with open(precond_path) as f:
        rules = json.load(f)

    # Load scores
    scores_path = os.path.join(root, "results", "scores", args.dataset, f"{args.split}{out_suffix}.jsonl")
    if not os.path.exists(scores_path):
        logging.error(f"Scores not found: {scores_path}")
        sys.exit(1)
    scores = load_scores(scores_path)
    logging.info(f"Loaded scores for {len(scores)} videos from {scores_path}")

    # Load GT labels
    gt_labels = load_gt_labels(cfg, args.split, root)
    logging.info(f"Loaded GT labels for {len(gt_labels)} videos")

    # Classify each video
    y_true, y_pred = [], []
    predictions = []
    rule_violation_counts = Counter()

    for vid_id, gt in gt_labels.items():
        if vid_id not in scores:
            logging.warning(f"  {vid_id}: no scores, skipping")
            continue

        hateful, violated_rules = classify_video(scores[vid_id], rules)
        pred = 1 if hateful else 0

        y_true.append(gt)
        y_pred.append(pred)
        predictions.append({
            "video_id": vid_id,
            "gt_label": gt,
            "predicted": pred,
            "violated_rules": violated_rules,
        })

        for r in violated_rules:
            rule_violation_counts[r] += 1

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    logging.info(f"\n{'='*60}")
    logging.info(f"Results: {args.dataset} ({args.split})")
    logging.info(f"  n={len(y_true)}")
    logging.info(f"  ACC:       {metrics['accuracy']:.4f}")
    logging.info(f"  F1:        {metrics['f1']:.4f}")
    logging.info(f"  F1 macro:  {metrics['f1_macro']:.4f}")
    logging.info(f"  Precision: {metrics['precision']:.4f}")
    logging.info(f"  Recall:    {metrics['recall']:.4f}")

    # Per-rule violation analysis
    logging.info(f"\nRule violation counts:")
    for rule in rules:
        count = rule_violation_counts.get(rule["rule_id"], 0)
        logging.info(f"  {rule['rule_id']} ({rule.get('name', '')}): {count} videos")

    # Prediction distribution
    pred_counter = Counter(y_pred)
    gt_counter = Counter(y_true)
    logging.info(f"\nPredicted: hateful={pred_counter.get(1,0)}, not_hateful={pred_counter.get(0,0)}")
    logging.info(f"GT:        hateful={gt_counter.get(1,0)}, not_hateful={gt_counter.get(0,0)}")

    # Per-rule ablation: remove each rule, recompute ACC
    logging.info(f"\nPer-rule ablation (ACC when rule is removed):")
    for exclude_rule in rules:
        excluded_id = exclude_rule["rule_id"]
        remaining_rules = [r for r in rules if r["rule_id"] != excluded_id]
        y_pred_ablation = []
        for vid_id, gt in gt_labels.items():
            if vid_id not in scores:
                continue
            hateful, _ = classify_video(scores[vid_id], remaining_rules)
            y_pred_ablation.append(1 if hateful else 0)
        if y_pred_ablation:
            from sklearn.metrics import accuracy_score
            acc_ablation = accuracy_score(y_true, y_pred_ablation)
            delta = acc_ablation - metrics["accuracy"]
            logging.info(f"  Without {excluded_id}: ACC={acc_ablation:.4f} (delta={delta:+.4f})")

    # Save results
    out_dir = os.path.join(root, "results", "eval", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"results_{args.split}{out_suffix}.json")
    with open(out_path, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "split": args.split,
            "constitution": constitution_name,
            "n_videos": len(y_true),
            "metrics": metrics,
            "rule_violation_counts": dict(rule_violation_counts),
            "predictions": predictions,
        }, f, indent=2, ensure_ascii=False)

    logging.info(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
