"""
Iteration 1 analysis: content-free calibration + unsupervised threshold evaluation.

Analyzes:
  1. Raw vs calibrated score distributions (per class)
  2. Unsupervised threshold (Otsu, GMM) on raw and calibrated scores
  3. Oracle vs unsupervised ACC gap
  4. Calibration destruction analysis (how many TPs collapse to zero)
  5. 8B model comparison (if results exist)
  6. Gap analysis to 80% target

Reads:
  - results/calibrated/{dataset}/binary_results.json (from calibrate_and_threshold.py)
  - results/holistic_2b/{dataset}/test_binary.jsonl (raw test scores)
  - results/holistic_2b/{dataset}/train_binary.jsonl (raw train scores)
  - results/holistic_2b/content_free.json (p_base values)
  - results/holistic_8b/{dataset}/test_binary.jsonl (8B scores, optional)
  - results/analysis/iteration_0.json (for comparison)

Outputs:
  - results/analysis/iteration_1.json
"""

import json
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
    """Apply affine calibration: score_cal = max(0, (score - p_base)) / (1 - p_base)."""
    denom = 1.0 - p_base
    if denom <= 0:
        return scores
    return np.clip((scores - p_base) / denom, 0.0, 1.0)


def evaluate(scores, labels, threshold):
    """Compute ACC, F1, Precision, Recall at a given threshold."""
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
        "acc": round(float(acc), 6),
        "f1": round(float(f1), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def find_oracle(scores, labels, metric="acc"):
    """Find threshold maximizing the given metric."""
    best_val = 0
    best_thresh = 0.5
    for t in np.arange(0.0, 1.001, 0.005):
        m = evaluate(scores, labels, t)
        if m[metric] > best_val:
            best_val = m[metric]
            best_thresh = float(t)
    return best_thresh, best_val


def analyze_calibration_destruction(raw_scores, labels, p_base):
    """Analyze how many positive-class scores collapse to zero after calibration."""
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_raw = raw_scores[pos_mask]
    neg_raw = raw_scores[neg_mask]

    # How many positives have raw score < p_base (will be clipped to 0)?
    pos_below_pbase = (pos_raw < p_base).sum()
    neg_below_pbase = (neg_raw < p_base).sum()

    return {
        "p_base": round(float(p_base), 6),
        "n_pos": int(pos_mask.sum()),
        "n_neg": int(neg_mask.sum()),
        "pos_below_pbase": int(pos_below_pbase),
        "pos_below_pbase_pct": round(float(pos_below_pbase / pos_mask.sum() * 100), 1) if pos_mask.sum() > 0 else 0,
        "neg_below_pbase": int(neg_below_pbase),
        "neg_below_pbase_pct": round(float(neg_below_pbase / neg_mask.sum() * 100), 1) if neg_mask.sum() > 0 else 0,
        "pos_mean_raw": round(float(pos_raw.mean()), 6),
        "pos_median_raw": round(float(np.median(pos_raw)), 6),
        "neg_mean_raw": round(float(neg_raw.mean()), 6),
        "neg_median_raw": round(float(np.median(neg_raw)), 6),
    }


def analyze_dataset(dataset_name, annotations, content_free_json, results_8b=None):
    """Full analysis for one dataset (EN or ZH)."""
    ds_key = dataset_name  # "MHClip_EN" or "MHClip_ZH"
    lang = "en" if "EN" in ds_key else "zh"
    p_base = content_free_json[f"{lang}_p_base"]

    # Load test scores
    test_path = os.path.join(PROJECT_ROOT, f"results/holistic_2b/{ds_key}/test_binary.jsonl")
    test_records = load_scores(test_path)
    test_vids = [v for v, s in test_records]
    test_raw = np.array([s for v, s in test_records])

    # Build labels
    labels = []
    valid_mask = []
    gt_classes = []
    for vid in test_vids:
        if vid in annotations:
            gt = annotations[vid]["label"]
            labels.append(1 if gt in ("Hateful", "Offensive") else 0)
            valid_mask.append(True)
            gt_classes.append(gt)
        else:
            labels.append(0)
            valid_mask.append(False)
            gt_classes.append("Unknown")
    labels = np.array(labels)
    valid_mask = np.array(valid_mask)

    test_raw_valid = test_raw[valid_mask]
    labels_valid = labels[valid_mask]
    gt_classes_valid = [gt_classes[i] for i in range(len(gt_classes)) if valid_mask[i]]

    # Calibrated scores
    test_cal = calibrate_scores(test_raw, p_base)
    test_cal_valid = test_cal[valid_mask]

    result = {
        "dataset": ds_key,
        "p_base": round(float(p_base), 6),
        "n_test_total": len(test_records),
        "n_test_valid": int(valid_mask.sum()),
        "n_pos": int(labels_valid.sum()),
        "n_neg": int((1 - labels_valid).sum()),
    }

    # --- AUC-ROC and Cohen's d ---
    from sklearn.metrics import roc_auc_score
    pos_scores = test_raw_valid[labels_valid == 1]
    neg_scores = test_raw_valid[labels_valid == 0]
    auc = float(roc_auc_score(labels_valid, test_raw_valid))
    pooled_std = np.sqrt(
        ((len(pos_scores) - 1) * pos_scores.std() ** 2 +
         (len(neg_scores) - 1) * neg_scores.std() ** 2)
        / (len(pos_scores) + len(neg_scores) - 2)
    )
    cohens_d = float((pos_scores.mean() - neg_scores.mean()) / pooled_std)
    result["auc_roc"] = round(auc, 4)
    result["cohens_d"] = round(cohens_d, 4)

    # --- Calibration destruction analysis ---
    result["calibration_destruction"] = analyze_calibration_destruction(
        test_raw_valid, labels_valid, p_base
    )

    # --- Raw score oracle ---
    raw_oracle_t, raw_oracle_acc = find_oracle(test_raw_valid, labels_valid, "acc")
    result["raw_oracle"] = {
        "threshold": round(raw_oracle_t, 4),
        **evaluate(test_raw_valid, labels_valid, raw_oracle_t),
    }

    # --- Calibrated score oracle ---
    cal_oracle_t, cal_oracle_acc = find_oracle(test_cal_valid, labels_valid, "acc")
    result["cal_oracle"] = {
        "threshold": round(cal_oracle_t, 4),
        **evaluate(test_cal_valid, labels_valid, cal_oracle_t),
    }

    # --- FN breakdown at oracle threshold ---
    oracle_t = raw_oracle_t
    fn_mask = (test_raw_valid < oracle_t) & (labels_valid == 1)
    fn_scores = test_raw_valid[fn_mask]
    fn_classes = [gt_classes_valid[i] for i in range(len(gt_classes_valid)) if fn_mask[i]]
    result["fn_breakdown"] = {
        "n_fn": int(fn_mask.sum()),
        "n_fn_hateful": fn_classes.count("Hateful"),
        "n_fn_offensive": fn_classes.count("Offensive"),
        "fn_score_lt_005": int((fn_scores < 0.05).sum()),
        "fn_score_005_020": int(((fn_scores >= 0.05) & (fn_scores < 0.20)).sum()),
        "fn_score_gte_020": int((fn_scores >= 0.20).sum()),
        "fn_mean": round(float(fn_scores.mean()), 4) if len(fn_scores) > 0 else None,
        "fn_median": round(float(np.median(fn_scores)), 4) if len(fn_scores) > 0 else None,
    }

    # --- Load calibration pipeline results (if available) ---
    cal_result_path = os.path.join(PROJECT_ROOT, f"results/calibrated/{ds_key}/binary_results.json")
    if os.path.exists(cal_result_path):
        with open(cal_result_path) as f:
            cal_pipeline = json.load(f)
        result["calibration_pipeline"] = cal_pipeline
    else:
        result["calibration_pipeline"] = None

    # --- Unsupervised thresholds on raw test scores (direct, no train) ---
    from calibrate_and_threshold import otsu_threshold, gmm_threshold

    raw_otsu_t = otsu_threshold(test_raw_valid)
    raw_gmm_t, raw_gmm_info = gmm_threshold(test_raw_valid)
    cal_otsu_t = otsu_threshold(test_cal_valid)
    cal_gmm_t, cal_gmm_info = gmm_threshold(test_cal_valid)

    result["unsupervised_on_test"] = {
        "raw_otsu": {"threshold": round(raw_otsu_t, 4), **evaluate(test_raw_valid, labels_valid, raw_otsu_t)},
        "raw_gmm": {"threshold": round(raw_gmm_t, 4), **evaluate(test_raw_valid, labels_valid, raw_gmm_t)},
        "cal_otsu": {"threshold": round(cal_otsu_t, 4), **evaluate(test_cal_valid, labels_valid, cal_otsu_t)},
        "cal_gmm": {"threshold": round(cal_gmm_t, 4), **evaluate(test_cal_valid, labels_valid, cal_gmm_t)},
    }

    # --- Per-class analysis of calibration effect ---
    per_class = {}
    for cls in ["Hateful", "Offensive", "Normal"]:
        cls_mask = np.array([g == cls for g in gt_classes_valid])
        n = int(cls_mask.sum())
        if n == 0:
            continue
        cls_raw = test_raw_valid[cls_mask]
        cls_cal = test_cal_valid[cls_mask]
        per_class[cls] = {
            "n": n,
            "raw_mean": round(float(cls_raw.mean()), 6),
            "raw_median": round(float(np.median(cls_raw)), 6),
            "cal_mean": round(float(cls_cal.mean()), 6),
            "cal_median": round(float(np.median(cls_cal)), 6),
            "n_below_pbase": int((cls_raw < p_base).sum()),
            "pct_below_pbase": round(float((cls_raw < p_base).sum() / n * 100), 1),
        }
    result["per_class_calibration"] = per_class

    # --- 8B comparison (if available) ---
    if results_8b is not None:
        vids_8b = [v for v, s in results_8b]
        scores_8b = np.array([s for v, s in results_8b])
        labels_8b = []
        valid_8b = []
        for vid in vids_8b:
            if vid in annotations:
                gt = annotations[vid]["label"]
                labels_8b.append(1 if gt in ("Hateful", "Offensive") else 0)
                valid_8b.append(True)
            else:
                labels_8b.append(0)
                valid_8b.append(False)
        labels_8b = np.array(labels_8b)
        valid_8b = np.array(valid_8b)
        scores_8b_valid = scores_8b[valid_8b]
        labels_8b_valid = labels_8b[valid_8b]

        oracle_8b_t, oracle_8b_acc = find_oracle(scores_8b_valid, labels_8b_valid, "acc")
        result["model_8b"] = {
            "n_scored": len(results_8b),
            "n_valid": int(valid_8b.sum()),
            "oracle_threshold": round(oracle_8b_t, 4),
            **evaluate(scores_8b_valid, labels_8b_valid, oracle_8b_t),
            "mean_score": round(float(scores_8b_valid.mean()), 6),
            "mean_pos": round(float(scores_8b_valid[labels_8b_valid == 1].mean()), 6) if labels_8b_valid.sum() > 0 else None,
            "mean_neg": round(float(scores_8b_valid[labels_8b_valid == 0].mean()), 6) if (1 - labels_8b_valid).sum() > 0 else None,
        }

    # --- Gap analysis ---
    target = 0.80
    best_acc = result["raw_oracle"]["acc"]
    n_valid = result["n_test_valid"]
    n_correct = int(best_acc * n_valid)
    n_needed = int(np.ceil(target * n_valid))
    result["gap"] = {
        "target": target,
        "best_raw_acc": best_acc,
        "n_correct_now": n_correct,
        "n_needed": n_needed,
        "n_more_needed": max(0, n_needed - n_correct),
        "gap_pp": round((target - best_acc) * 100, 1),
    }

    return result


def main():
    # Load annotations
    en_annot = load_annotations("MHClip_EN")
    zh_annot = load_annotations("MHClip_ZH")

    # Load content-free calibration values
    cf_path = os.path.join(PROJECT_ROOT, "results/holistic_2b/content_free.json")
    with open(cf_path) as f:
        content_free = json.load(f)

    # Check for 8B results
    en_8b_path = os.path.join(PROJECT_ROOT, "results/holistic_8b/MHClip_EN/test_binary.jsonl")
    zh_8b_path = os.path.join(PROJECT_ROOT, "results/holistic_8b/MHClip_ZH/test_binary.jsonl")

    en_8b = load_scores(en_8b_path) if os.path.exists(en_8b_path) else None
    zh_8b = load_scores(zh_8b_path) if os.path.exists(zh_8b_path) else None

    if en_8b:
        print(f"Found 8B EN test results: {len(en_8b)} records")
    if zh_8b:
        print(f"Found 8B ZH test results: {len(zh_8b)} records")

    # Analyze each dataset
    en_result = analyze_dataset("MHClip_EN", en_annot, content_free, en_8b)
    zh_result = analyze_dataset("MHClip_ZH", zh_annot, content_free, zh_8b)

    # --- Load Iteration 0 for comparison ---
    iter0_path = os.path.join(PROJECT_ROOT, "results/analysis/iteration_0.json")
    iter0 = None
    if os.path.exists(iter0_path):
        with open(iter0_path) as f:
            iter0 = json.load(f)

    # Build comparison summary
    summary = {
        "iteration": 1,
        "method": "content-free calibration + unsupervised threshold (Otsu/GMM)",
        "model": "Qwen3-VL-2B-Instruct",
    }

    # Best label-free ACC for each dataset
    for key, res in [("EN", en_result), ("ZH", zh_result)]:
        unsup = res.get("unsupervised_on_test", {})
        best_unsup_acc = 0
        best_unsup_method = ""
        for method_name, method_res in unsup.items():
            if method_res["acc"] > best_unsup_acc:
                best_unsup_acc = method_res["acc"]
                best_unsup_method = method_name
        summary[f"{key}_best_unsupervised"] = {
            "method": best_unsup_method,
            "acc": best_unsup_acc,
        }
        summary[f"{key}_oracle_acc"] = res["raw_oracle"]["acc"]
        summary[f"{key}_gap_pp"] = res["gap"]["gap_pp"]
        if "model_8b" in res:
            summary[f"{key}_8b_oracle_acc"] = res["model_8b"]["acc"]

    # Comparison with Iteration 0
    if iter0:
        summary["comparison_with_iter0"] = {
            "EN_iter0_oracle_acc": iter0["EN_binary"]["best_acc"]["acc"],
            "EN_iter0_unsup_gmm_acc": iter0["EN_binary"]["unsupervised_thresholds"]["gmm_acc"],
            "ZH_iter0_oracle_acc": iter0["ZH_binary"]["best_acc"]["acc"],
            "ZH_iter0_unsup_gmm_acc": iter0["ZH_binary"]["unsupervised_thresholds"]["gmm_acc"],
        }

    results = {
        "summary": summary,
        "MHClip_EN": en_result,
        "MHClip_ZH": zh_result,
    }

    # Save
    out_dir = os.path.join(PROJECT_ROOT, "results/analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "iteration_1.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("ITERATION 1 ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Method: {summary['method']}")
    print(f"Model: {summary['model']}")
    print()

    for key, res in [("EN", en_result), ("ZH", zh_result)]:
        ds = res["dataset"]
        print(f"\n--- {ds} ({res['n_test_valid']} valid, {res['n_pos']} pos, {res['n_neg']} neg) ---")
        print(f"  p_base: {res['p_base']:.6f}")

        cd = res["calibration_destruction"]
        print(f"  Calibration destruction: {cd['pos_below_pbase']}/{cd['n_pos']} "
              f"({cd['pos_below_pbase_pct']}%) positives collapse to 0")
        print(f"  Positive raw: mean={cd['pos_mean_raw']:.4f}, median={cd['pos_median_raw']:.4f}")
        print(f"  Negative raw: mean={cd['neg_mean_raw']:.4f}, median={cd['neg_median_raw']:.4f}")

        print(f"\n  Oracle ACC (raw):  {res['raw_oracle']['acc']:.4f} @ t={res['raw_oracle']['threshold']:.4f}")
        print(f"  Oracle ACC (cal):  {res['cal_oracle']['acc']:.4f} @ t={res['cal_oracle']['threshold']:.4f}")

        unsup = res["unsupervised_on_test"]
        print(f"\n  Unsupervised thresholds (on test scores):")
        for method, mr in unsup.items():
            print(f"    {method:<12} t={mr['threshold']:.4f}  ACC={mr['acc']:.4f}  "
                  f"F1={mr['f1']:.4f}  Rec={mr['recall']:.4f}")

        if res.get("calibration_pipeline"):
            print(f"\n  Calibration pipeline (train-selected thresholds):")
            cp = res["calibration_pipeline"]
            for cond_key in ["metrics_raw_+_otsu", "metrics_raw_+_gmm",
                             "metrics_cal_+_otsu", "metrics_cal_+_gmm"]:
                if cond_key in cp:
                    m = cp[cond_key]
                    print(f"    {cond_key:<24} ACC={m['acc']:.4f}  F1={m['f1']:.4f}")

        if "model_8b" in res:
            m8 = res["model_8b"]
            print(f"\n  8B model: ACC={m8['acc']:.4f} @ t={m8['oracle_threshold']:.4f}  "
                  f"({m8['n_valid']} valid)")
            print(f"    mean_pos={m8.get('mean_pos', 'N/A')}, mean_neg={m8.get('mean_neg', 'N/A')}")

        g = res["gap"]
        print(f"\n  Gap to 80%: {g['gap_pp']}pp ({g['n_more_needed']} more correct needed)")

        # Per-class
        print(f"\n  Per-class calibration effect:")
        for cls, ci in res.get("per_class_calibration", {}).items():
            print(f"    {cls:<12} n={ci['n']:<3} raw_mean={ci['raw_mean']:.4f} "
                  f"cal_mean={ci['cal_mean']:.4f} "
                  f"below_pbase={ci['n_below_pbase']}/{ci['n']} ({ci['pct_below_pbase']}%)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
