"""Reproduce best label-free unified method: Binary Raw + Otsu (train-derived threshold).

Steps:
1. Load train binary scores for each dataset
2. Fit Otsu threshold on train score distribution (no labels)
3. Apply threshold to test scores
4. Compute ACC with test labels
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_annotations


def otsu_threshold(scores):
    """Otsu's method — same implementation as architect's calibrate_and_threshold.py.
    Finds the threshold that minimizes intra-class variance."""
    scores = np.asarray(scores)
    sorted_scores = np.sort(scores)
    n = len(sorted_scores)
    best_thresh = 0.5
    best_variance = float("inf")

    thresholds = np.linspace(sorted_scores.min(), sorted_scores.max(), 200)
    for t in thresholds:
        c0 = scores[scores < t]
        c1 = scores[scores >= t]
        if len(c0) == 0 or len(c1) == 0:
            continue
        w0 = len(c0) / n
        w1 = len(c1) / n
        var0 = c0.var()
        var1 = c1.var()
        within_var = w0 * var0 + w1 * var1
        if within_var < best_variance:
            best_variance = within_var
            best_thresh = t

    return float(best_thresh)


def load_scores(path):
    scores = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            vid = r.get("video_id")
            s = r.get("score")
            if vid and s is not None:
                scores[vid] = float(s)
    return scores


def eval_dataset(dataset):
    print(f"\n=== {dataset} ===")

    train_path = f"/data/jehc223/EMNLP2/results/holistic_2b/{dataset}/train_binary.jsonl"
    test_path = f"/data/jehc223/EMNLP2/results/holistic_2b/{dataset}/test_binary.jsonl"

    # Load scores
    train_scores_dict = load_scores(train_path)
    test_scores_dict = load_scores(test_path)
    print(f"  Train scored: {len(train_scores_dict)}")
    print(f"  Test scored: {len(test_scores_dict)}")

    # Fit Otsu threshold on TRAIN scores (no labels)
    train_scores = list(train_scores_dict.values())
    otsu_t = otsu_threshold(train_scores)
    print(f"  Otsu threshold (train-derived): {otsu_t:.4f}")

    # Load test annotations
    ann = load_annotations(dataset)

    # Build test arrays
    scores, labels = [], []
    for vid, score in test_scores_dict.items():
        if vid not in ann:
            continue
        lbl = ann[vid]["label"]
        gt = 1 if lbl in ("Hateful", "Offensive") else 0
        scores.append(score)
        labels.append(gt)

    scores = np.array(scores)
    labels = np.array(labels)
    n = len(labels)
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    print(f"  Valid test: {n} (pos={n_pos}, neg={n_neg})")

    # Apply Otsu threshold
    preds = (scores >= otsu_t).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"  ACC={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")
    print(f"  Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")

    return {"dataset": dataset, "threshold": otsu_t, "acc": acc, "f1": f1,
            "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_test": n, "n_pos": n_pos, "n_neg": n_neg}


def main():
    print("Reproducing: 2B Binary Raw + Otsu (train-derived) — UNIFIED METHOD")
    results = {}
    for ds in ["MHClip_EN", "MHClip_ZH"]:
        results[ds] = eval_dataset(ds)

    print("\n=== FINAL UNIFIED RESULT ===")
    print(f"{'Dataset':<12} {'Threshold':<10} {'ACC':<8} {'Target 80%':<10}")
    for ds in ["MHClip_EN", "MHClip_ZH"]:
        r = results[ds]
        status = "PASS" if r["acc"] >= 0.80 else f"FAIL ({(r['acc']-0.80)*100:+.2f}pp)"
        print(f"{ds:<12} {r['threshold']:<10.4f} {r['acc']:<8.4f} {status}")

    os.makedirs("/data/jehc223/EMNLP2/results/reproduction", exist_ok=True)
    with open("/data/jehc223/EMNLP2/results/reproduction/best_unified.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to: results/reproduction/best_unified.json")


if __name__ == "__main__":
    main()
