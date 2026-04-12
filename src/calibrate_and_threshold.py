"""
Iteration 1: Content-free calibration + unsupervised threshold selection.

Applies affine calibration to remove MLLM base-rate bias, then selects
a decision threshold via 2-component GMM crossover or Otsu's method.

Usage:
  python src/calibrate_and_threshold.py \
    --dataset MHClip_EN --mode binary \
    --p-base 0.02 \
    --train-scores results/holistic_2b/MHClip_EN/train_binary.jsonl \
    --test-scores results/holistic_2b/MHClip_EN/test_binary.jsonl

  Or with auto p_base from calibration file:
  python src/calibrate_and_threshold.py \
    --dataset MHClip_EN --mode binary \
    --calibration-file results/holistic_2b/content_free.json \
    --train-scores results/holistic_2b/MHClip_EN/train_binary.jsonl \
    --test-scores results/holistic_2b/MHClip_EN/test_binary.jsonl
"""

import argparse
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_annotations

PROJECT_ROOT = "/data/jehc223/EMNLP2"


def load_scores(path):
    """Load scores from JSONL. Returns list of (video_id, score) tuples."""
    records = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            vid = r["video_id"]
            s = r.get("score")
            if s is not None:
                records.append((vid, float(s)))
    return records


def calibrate_scores(scores, p_base):
    """Apply affine calibration: score_cal = max(0, (score - p_base)) / (1 - p_base).
    Shifts the base-rate bias so content-free input maps to 0."""
    denom = 1.0 - p_base
    if denom <= 0:
        logging.warning(f"p_base={p_base} >= 1.0, skipping calibration")
        return scores
    return np.clip((scores - p_base) / denom, 0.0, 1.0)


def otsu_threshold(scores):
    """Compute Otsu's threshold on a 1D score array.
    Finds the threshold that minimizes intra-class variance."""
    sorted_scores = np.sort(scores)
    n = len(sorted_scores)
    best_thresh = 0.5
    best_variance = float("inf")

    # Sweep over unique midpoints between consecutive scores
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


def gmm_threshold(scores):
    """Fit 2-component GMM and find crossover point where
    P(high_component | score) = 0.5."""
    from sklearn.mixture import GaussianMixture

    X = scores.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
    gmm.fit(X)

    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()

    # Identify low and high components by mean
    if means[0] < means[1]:
        low_idx, high_idx = 0, 1
    else:
        low_idx, high_idx = 1, 0

    logging.info(f"  GMM low  component: mean={means[low_idx]:.4f}, std={stds[low_idx]:.4f}, weight={weights[low_idx]:.4f}")
    logging.info(f"  GMM high component: mean={means[high_idx]:.4f}, std={stds[high_idx]:.4f}, weight={weights[high_idx]:.4f}")

    # Find crossover: sweep between means, find where posterior of high = 0.5
    search_lo = means[low_idx]
    search_hi = means[high_idx]
    search_points = np.linspace(search_lo, search_hi, 1000)

    probs = gmm.predict_proba(search_points.reshape(-1, 1))
    # probs[:, high_idx] is P(high | score)
    p_high = probs[:, high_idx]

    # Find where p_high crosses 0.5
    crossover_idx = np.argmin(np.abs(p_high - 0.5))
    crossover = float(search_points[crossover_idx])

    logging.info(f"  GMM crossover threshold: {crossover:.4f}")
    return crossover, {
        "low_mean": float(means[low_idx]),
        "low_std": float(stds[low_idx]),
        "low_weight": float(weights[low_idx]),
        "high_mean": float(means[high_idx]),
        "high_std": float(stds[high_idx]),
        "high_weight": float(weights[high_idx]),
        "crossover": crossover,
    }


def evaluate(scores, labels, threshold):
    """Compute ACC, F1, Precision, Recall, confusion matrix at a given threshold."""
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    acc = (tp + tn) / len(labels) if len(labels) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "acc": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main():
    parser = argparse.ArgumentParser(description="Content-free calibration + unsupervised threshold")
    parser.add_argument("--dataset", required=True, choices=["MHClip_EN", "MHClip_ZH"])
    parser.add_argument("--mode", default="binary", choices=["binary", "triclass"])
    parser.add_argument("--p-base", type=float, default=None,
                        help="Content-free P(Yes) base rate. If not set, read from calibration file.")
    parser.add_argument("--calibration-file", default=None,
                        help="JSON file with en_p_base and zh_p_base fields")
    parser.add_argument("--train-scores", required=True,
                        help="JSONL of training split scores (for threshold selection)")
    parser.add_argument("--test-scores", required=True,
                        help="JSONL of test split scores (for evaluation)")
    args = parser.parse_args()

    os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(PROJECT_ROOT, "logs",
                                             f"calibrate_{args.dataset}_{args.mode}.log")),
            logging.StreamHandler(),
        ],
    )

    # Determine p_base
    if args.p_base is not None:
        p_base = args.p_base
    elif args.calibration_file is not None:
        with open(args.calibration_file) as f:
            cal = json.load(f)
        key = "en_p_base" if args.dataset == "MHClip_EN" else "zh_p_base"
        p_base = cal[key]
    else:
        parser.error("Must specify --p-base or --calibration-file")

    logging.info(f"Config: dataset={args.dataset} mode={args.mode} p_base={p_base:.6f}")

    # Load annotations for ground truth
    annotations = load_annotations(args.dataset)

    # Load and calibrate training scores
    train_records = load_scores(args.train_scores)
    train_vids = [v for v, s in train_records]
    train_raw = np.array([s for v, s in train_records])
    train_cal = calibrate_scores(train_raw, p_base)

    logging.info(f"Training: {len(train_records)} scored videos")
    logging.info(f"  Raw scores:  mean={train_raw.mean():.4f}, std={train_raw.std():.4f}")
    logging.info(f"  Cal scores:  mean={train_cal.mean():.4f}, std={train_cal.std():.4f}")

    # Load and calibrate test scores
    test_records = load_scores(args.test_scores)
    test_vids = [v for v, s in test_records]
    test_raw = np.array([s for v, s in test_records])
    test_cal = calibrate_scores(test_raw, p_base)

    # Build test labels
    test_labels = []
    valid_mask = []
    for vid in test_vids:
        if vid in annotations:
            gt_label = annotations[vid]["label"]
            test_labels.append(1 if gt_label in ("Hateful", "Offensive") else 0)
            valid_mask.append(True)
        else:
            test_labels.append(0)
            valid_mask.append(False)
    test_labels = np.array(test_labels)
    valid_mask = np.array(valid_mask)

    test_cal_valid = test_cal[valid_mask]
    test_labels_valid = test_labels[valid_mask]

    logging.info(f"Test: {len(test_records)} scored, {valid_mask.sum()} with GT "
                 f"({int(test_labels_valid.sum())} pos, {int((1-test_labels_valid).sum())} neg)")

    # --- Threshold selection on training scores (full 2x3 ablation) ---
    logging.info("\n=== Unsupervised Threshold Selection (on training scores) ===")

    # Raw thresholds
    raw_thresh_otsu = otsu_threshold(train_raw)
    logging.info(f"Raw Otsu threshold:  {raw_thresh_otsu:.4f}")
    raw_thresh_gmm, raw_gmm_info = gmm_threshold(train_raw)
    logging.info(f"Raw GMM threshold:   {raw_thresh_gmm:.4f}")

    # Calibrated thresholds
    cal_thresh_otsu = otsu_threshold(train_cal)
    logging.info(f"Cal Otsu threshold:  {cal_thresh_otsu:.4f}")
    cal_thresh_gmm, cal_gmm_info = gmm_threshold(train_cal)
    logging.info(f"Cal GMM threshold:   {cal_thresh_gmm:.4f}")

    # --- Evaluate on test set: full 2x3 ablation ---
    logging.info("\n=== Test Set Evaluation (2x3 ablation) ===")

    test_raw_valid = test_raw[valid_mask]

    # Helper: find oracle threshold maximizing ACC (our target metric)
    def find_oracle(scores, labels):
        best_acc, best_f1 = 0, 0
        best_acc_t, best_f1_t = 0.5, 0.5
        for t in np.arange(0.0, 1.001, 0.01):
            m = evaluate(scores, labels, t)
            if m["acc"] > best_acc:
                best_acc = m["acc"]
                best_acc_t = t
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_f1_t = t
        return best_acc_t, best_f1_t

    # All 6 conditions: (Raw|Cal) x (Otsu|GMM|Oracle)
    conditions = {}

    # Raw + Otsu
    conditions["Raw + Otsu"] = {
        "thresh": raw_thresh_otsu,
        "metrics": evaluate(test_raw_valid, test_labels_valid, raw_thresh_otsu),
    }
    # Raw + GMM
    conditions["Raw + GMM"] = {
        "thresh": raw_thresh_gmm,
        "metrics": evaluate(test_raw_valid, test_labels_valid, raw_thresh_gmm),
    }
    # Raw + Oracle (ACC)
    raw_oracle_acc_t, raw_oracle_f1_t = find_oracle(test_raw_valid, test_labels_valid)
    conditions["Raw + Oracle(ACC)"] = {
        "thresh": raw_oracle_acc_t,
        "metrics": evaluate(test_raw_valid, test_labels_valid, raw_oracle_acc_t),
    }

    # Cal + Otsu
    conditions["Cal + Otsu"] = {
        "thresh": cal_thresh_otsu,
        "metrics": evaluate(test_cal_valid, test_labels_valid, cal_thresh_otsu),
    }
    # Cal + GMM
    conditions["Cal + GMM"] = {
        "thresh": cal_thresh_gmm,
        "metrics": evaluate(test_cal_valid, test_labels_valid, cal_thresh_gmm),
    }
    # Cal + Oracle (ACC)
    cal_oracle_acc_t, cal_oracle_f1_t = find_oracle(test_cal_valid, test_labels_valid)
    conditions["Cal + Oracle(ACC)"] = {
        "thresh": cal_oracle_acc_t,
        "metrics": evaluate(test_cal_valid, test_labels_valid, cal_oracle_acc_t),
    }

    for name, cond in conditions.items():
        m = cond["metrics"]
        logging.info(f"{name:<22} thresh={cond['thresh']:.4f}  "
                     f"ACC={m['acc']:.4f} F1={m['f1']:.4f} "
                     f"Prec={m['precision']:.4f} Recall={m['recall']:.4f}  "
                     f"TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")

    # --- Save results ---
    out_dir = os.path.join(PROJECT_ROOT, "results", "calibrated", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.mode}_results.json")

    results = {
        "dataset": args.dataset,
        "mode": args.mode,
        "p_base": p_base,
        "n_train": len(train_records),
        "n_test": int(valid_mask.sum()),
        "n_test_pos": int(test_labels_valid.sum()),
        "n_test_neg": int((1 - test_labels_valid).sum()),
        "raw_thresh_otsu": raw_thresh_otsu,
        "raw_thresh_gmm": raw_thresh_gmm,
        "raw_gmm_info": raw_gmm_info,
        "cal_thresh_otsu": cal_thresh_otsu,
        "cal_thresh_gmm": cal_thresh_gmm,
        "cal_gmm_info": cal_gmm_info,
        "raw_oracle_acc_thresh": raw_oracle_acc_t,
        "cal_oracle_acc_thresh": cal_oracle_acc_t,
    }
    for name, cond in conditions.items():
        key = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        results[f"metrics_{key}"] = cond["metrics"]
        results[f"thresh_{key}"] = cond["thresh"]

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info(f"\nResults saved to {out_path}")

    # Print summary table
    logging.info("\n=== Summary (2x3 Ablation) ===")
    logging.info(f"{'Method':<22} {'Thresh':>6} {'ACC':>6} {'F1':>6} {'Prec':>6} {'Recall':>6}")
    logging.info("-" * 60)
    for name, cond in conditions.items():
        m = cond["metrics"]
        logging.info(f"{name:<22} {cond['thresh']:>6.4f} {m['acc']:>6.4f} {m['f1']:>6.4f} "
                     f"{m['precision']:>6.4f} {m['recall']:>6.4f}")


if __name__ == "__main__":
    main()
