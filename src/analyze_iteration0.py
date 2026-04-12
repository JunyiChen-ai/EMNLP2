"""
Iteration 0 analysis: comprehensive metrics, threshold analysis, and diagnostics
for holistic 2B scoring on MHClip_EN and MHClip_ZH (binary + triclass).
"""

import json
import os
import sys
import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_annotations

PROJECT_ROOT = "/data/jehc223/EMNLP2"

RESULT_FILES = {
    "EN_binary": os.path.join(PROJECT_ROOT, "results/holistic_2b/MHClip_EN/test_binary.jsonl"),
    "EN_triclass": os.path.join(PROJECT_ROOT, "results/holistic_2b/MHClip_EN/test_triclass.jsonl"),
    "ZH_binary": os.path.join(PROJECT_ROOT, "results/holistic_2b/MHClip_ZH/test_binary.jsonl"),
    "ZH_triclass": os.path.join(PROJECT_ROOT, "results/holistic_2b/MHClip_ZH/test_triclass.jsonl"),
}

DATASET_MAP = {
    "EN_binary": "MHClip_EN",
    "EN_triclass": "MHClip_EN",
    "ZH_binary": "MHClip_ZH",
    "ZH_triclass": "MHClip_ZH",
}


def load_scores_and_labels(variant):
    """Load scores + ground truth for a variant. Returns (scores, labels, video_ids)."""
    path = RESULT_FILES[variant]
    dataset = DATASET_MAP[variant]
    annotations = load_annotations(dataset)

    scores = []
    labels = []
    video_ids = []

    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            vid = r["video_id"]
            s = r["score"]
            if s is None or vid not in annotations:
                continue
            gt_label = annotations[vid]["label"]
            gt = 1 if gt_label in ("Hateful", "Offensive") else 0
            scores.append(s)
            labels.append(gt)
            video_ids.append(vid)

    return np.array(scores), np.array(labels), video_ids


def threshold_sweep(scores, labels, step=0.01):
    """Fine threshold sweep. Returns list of dicts with metrics at each threshold."""
    results = []
    for thresh in np.arange(0.0, 1.001, step):
        preds = (scores >= thresh).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        acc = (tp + tn) / len(labels) if len(labels) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        results.append({
            "threshold": round(float(thresh), 2),
            "acc": acc, "f1": f1, "precision": precision, "recall": recall,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    return results


def best_metrics(sweep_results):
    """Extract best ACC and best F1 from sweep results."""
    best_acc_entry = max(sweep_results, key=lambda x: (x["acc"], x["f1"]))
    best_f1_entry = max(sweep_results, key=lambda x: (x["f1"], x["acc"]))
    return {
        "best_acc": best_acc_entry,
        "best_f1": best_f1_entry,
    }


def score_distribution_stats(scores, labels):
    """Compute score distribution stats per class."""
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]

    def stats(arr, name):
        if len(arr) == 0:
            return {"class": name, "n": 0}
        return {
            "class": name,
            "n": int(len(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
        }

    return {
        "positive": stats(pos_scores, "positive"),
        "negative": stats(neg_scores, "negative"),
        "all": stats(scores, "all"),
    }


def otsu_threshold(scores):
    """Compute Otsu's threshold on continuous scores (histogram-based)."""
    # Bin scores into 256 bins
    hist, bin_edges = np.histogram(scores, bins=256, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total = hist.sum()
    if total == 0:
        return 0.5

    # Otsu's method
    best_thresh = 0.5
    best_var = 0.0

    cum_sum = 0
    cum_mean = 0
    global_mean = np.sum(hist * bin_centers) / total

    for i in range(len(hist)):
        cum_sum += hist[i]
        if cum_sum == 0:
            continue
        if cum_sum == total:
            break
        cum_mean += hist[i] * bin_centers[i]

        w0 = cum_sum / total
        w1 = 1.0 - w0
        mu0 = cum_mean / cum_sum
        mu1 = (global_mean * total - cum_mean) / (total - cum_sum)

        between_var = w0 * w1 * (mu0 - mu1) ** 2
        if between_var > best_var:
            best_var = between_var
            best_thresh = bin_centers[i]

    return float(best_thresh)


def gmm_threshold(scores):
    """Fit 2-component GMM and return decision boundary."""
    from sklearn.mixture import GaussianMixture

    X = scores.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
    gmm.fit(X)

    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()

    # The threshold is the point where the two component posteriors are equal
    # Simple approach: sweep and find crossing point
    x_range = np.linspace(0, 1, 10000)
    probs = gmm.predict_proba(x_range.reshape(-1, 1))

    # Find where component assignments switch
    assignments = gmm.predict(x_range.reshape(-1, 1))
    changes = np.where(np.diff(assignments) != 0)[0]

    if len(changes) > 0:
        # Take the crossing point closest to the middle
        mid_idx = changes[np.argmin(np.abs(x_range[changes] - 0.5))]
        threshold = float(x_range[mid_idx])
    else:
        # No crossing - use midpoint of means
        threshold = float(np.mean(means))

    # Determine which component is "positive" (higher mean)
    pos_comp = int(np.argmax(means))

    return {
        "threshold": threshold,
        "means": [float(m) for m in means],
        "stds": [float(s) for s in stds],
        "weights": [float(w) for w in weights],
        "positive_component": pos_comp,
    }


def bimodality_coefficient(scores):
    """Compute bimodality coefficient: BC = (skewness^2 + 1) / (kurtosis + 3 * (n-1)^2 / ((n-2)(n-3))).
    BC > 0.555 suggests bimodality."""
    n = len(scores)
    if n < 4:
        return 0.0

    skew = float(scipy_stats.skew(scores))
    kurt = float(scipy_stats.kurtosis(scores, fisher=True))  # excess kurtosis

    # Standard bimodality coefficient
    # BC = (g1^2 + 1) / (g2 + 3*(n-1)^2/((n-2)*(n-3)))
    # where g1=skewness, g2=excess kurtosis
    numerator = skew ** 2 + 1
    denominator = kurt + 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3))

    if denominator == 0:
        return 0.0

    bc = numerator / denominator
    return float(bc)


def compare_binary_triclass(variant_binary, variant_triclass):
    """Compare which videos flipped between binary and triclass modes."""
    scores_b, labels_b, vids_b = load_scores_and_labels(variant_binary)
    scores_t, labels_t, vids_t = load_scores_and_labels(variant_triclass)

    # Get best thresholds
    sweep_b = threshold_sweep(scores_b, labels_b)
    sweep_t = threshold_sweep(scores_t, labels_t)
    best_b = best_metrics(sweep_b)["best_acc"]
    best_t = best_metrics(sweep_t)["best_acc"]

    thresh_b = best_b["threshold"]
    thresh_t = best_t["threshold"]

    # Build prediction maps
    score_map_b = {vid: s for vid, s in zip(vids_b, scores_b)}
    score_map_t = {vid: s for vid, s in zip(vids_t, scores_t)}
    label_map_b = {vid: l for vid, l in zip(vids_b, labels_b)}

    common_vids = sorted(set(vids_b) & set(vids_t))

    flipped_to_correct = []  # wrong in binary, correct in triclass
    flipped_to_wrong = []    # correct in binary, wrong in triclass
    both_correct = 0
    both_wrong = 0

    for vid in common_vids:
        gt = label_map_b[vid]
        pred_b = 1 if score_map_b[vid] >= thresh_b else 0
        pred_t = 1 if score_map_t[vid] >= thresh_t else 0
        correct_b = (pred_b == gt)
        correct_t = (pred_t == gt)

        if correct_b and correct_t:
            both_correct += 1
        elif not correct_b and not correct_t:
            both_wrong += 1
        elif not correct_b and correct_t:
            flipped_to_correct.append({
                "video_id": vid,
                "gt": int(gt),
                "score_binary": float(score_map_b[vid]),
                "score_triclass": float(score_map_t[vid]),
            })
        else:
            flipped_to_wrong.append({
                "video_id": vid,
                "gt": int(gt),
                "score_binary": float(score_map_b[vid]),
                "score_triclass": float(score_map_t[vid]),
            })

    return {
        "n_common": len(common_vids),
        "thresh_binary": thresh_b,
        "thresh_triclass": thresh_t,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "flipped_to_correct_in_triclass": len(flipped_to_correct),
        "flipped_to_wrong_in_triclass": len(flipped_to_wrong),
        "flipped_to_correct_details": flipped_to_correct,
        "flipped_to_wrong_details": flipped_to_wrong,
    }


def unsupervised_threshold_analysis(scores, labels, variant_name):
    """Apply Otsu and GMM thresholds, compare to oracle optimal."""
    sweep = threshold_sweep(scores, labels)
    oracle = best_metrics(sweep)["best_acc"]
    oracle_thresh = oracle["threshold"]
    oracle_acc = oracle["acc"]

    # Otsu
    otsu_t = otsu_threshold(scores)
    preds_otsu = (scores >= otsu_t).astype(int)
    otsu_acc = float(np.mean(preds_otsu == labels))

    # GMM
    gmm_result = gmm_threshold(scores)
    gmm_t = gmm_result["threshold"]
    # Determine if we should flip: if the "positive" component has lower mean, flip predictions
    preds_gmm = (scores >= gmm_t).astype(int)
    gmm_acc_normal = float(np.mean(preds_gmm == labels))
    gmm_acc_flipped = float(np.mean((1 - preds_gmm) == labels))
    if gmm_acc_flipped > gmm_acc_normal:
        gmm_acc = gmm_acc_flipped
        gmm_flipped = True
    else:
        gmm_acc = gmm_acc_normal
        gmm_flipped = False

    return {
        "variant": variant_name,
        "oracle_threshold": oracle_thresh,
        "oracle_acc": oracle_acc,
        "otsu_threshold": otsu_t,
        "otsu_acc": otsu_acc,
        "otsu_acc_gap": oracle_acc - otsu_acc,
        "gmm_threshold": gmm_t,
        "gmm_acc": gmm_acc,
        "gmm_flipped": gmm_flipped,
        "gmm_acc_gap": oracle_acc - gmm_acc,
        "gmm_details": gmm_result,
    }


def per_class_error_breakdown(scores, labels, video_ids, variant_name, dataset):
    """Break errors down by original 3-class labels (Hateful, Offensive, Normal)."""
    annotations = load_annotations(dataset)

    # Get best threshold
    sweep = threshold_sweep(scores, labels)
    best = best_metrics(sweep)["best_acc"]
    thresh = best["threshold"]

    categories = {"Hateful": [], "Offensive": [], "Normal": []}

    for i, vid in enumerate(video_ids):
        ann = annotations.get(vid, {})
        orig_label = ann.get("label", "Unknown")
        pred = 1 if scores[i] >= thresh else 0
        gt = labels[i]
        correct = (pred == gt)
        categories.setdefault(orig_label, []).append({
            "video_id": vid,
            "score": float(scores[i]),
            "pred": pred,
            "gt": int(gt),
            "correct": correct,
        })

    summary = {}
    for cat, items in categories.items():
        if not items:
            continue
        n = len(items)
        n_correct = sum(1 for x in items if x["correct"])
        n_wrong = sum(1 for x in items if not x["correct"])
        avg_score = np.mean([x["score"] for x in items])
        summary[cat] = {
            "n": n,
            "n_correct": n_correct,
            "n_wrong": n_wrong,
            "acc": n_correct / n if n > 0 else 0,
            "avg_score": float(avg_score),
        }

    return summary


def main():
    print("=" * 60)
    print("ITERATION 0 ANALYSIS: Holistic 2B Scoring")
    print("=" * 60)

    # Check which files exist
    available = {}
    for variant, path in RESULT_FILES.items():
        if os.path.isfile(path):
            with open(path) as f:
                n_lines = sum(1 for line in f if line.strip())
            available[variant] = n_lines
            print(f"  {variant}: {n_lines} records at {path}")
        else:
            print(f"  {variant}: NOT FOUND")

    if not available:
        print("No result files found. Exiting.")
        return

    all_results = {}

    # ========================================
    # 1. Metrics table + confusion matrices
    # ========================================
    print("\n" + "=" * 60)
    print("1. METRICS TABLE (oracle-optimal threshold)")
    print("=" * 60)

    header = f"{'Variant':<16} {'ACC':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Thresh':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}"
    print(header)
    print("-" * len(header))

    for variant in ["EN_binary", "EN_triclass", "ZH_binary", "ZH_triclass"]:
        if variant not in available:
            continue

        scores, labels, vids = load_scores_and_labels(variant)
        sweep = threshold_sweep(scores, labels)
        best = best_metrics(sweep)

        # Use best ACC threshold as primary
        b = best["best_acc"]
        print(f"{variant:<16} {b['acc']:>6.4f} {b['f1']:>6.4f} {b['precision']:>6.4f} {b['recall']:>6.4f} "
              f"{b['threshold']:>6.2f} {b['tp']:>4d} {b['fp']:>4d} {b['fn']:>4d} {b['tn']:>4d}")

        # Also show best F1
        bf = best["best_f1"]
        if bf["threshold"] != b["threshold"]:
            print(f"  (best-F1 @{bf['threshold']:.2f}: ACC={bf['acc']:.4f} F1={bf['f1']:.4f} P={bf['precision']:.4f} R={bf['recall']:.4f})")

        all_results[variant] = {
            "n_samples": len(scores),
            "n_pos": int(labels.sum()),
            "n_neg": int(len(labels) - labels.sum()),
            "best_acc": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in b.items()},
            "best_f1": {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in bf.items()},
        }

    # ========================================
    # 2. Confusion matrices (detailed)
    # ========================================
    print("\n" + "=" * 60)
    print("2. CONFUSION MATRICES")
    print("=" * 60)

    for variant in ["EN_binary", "EN_triclass", "ZH_binary", "ZH_triclass"]:
        if variant not in available:
            continue

        b = all_results[variant]["best_acc"]
        print(f"\n  {variant} (threshold={b['threshold']:.2f}):")
        print(f"                Pred=Hate   Pred=Normal")
        print(f"    GT=Hate     TP={b['tp']:>4d}      FN={b['fn']:>4d}")
        print(f"    GT=Normal   FP={b['fp']:>4d}      TN={b['tn']:>4d}")

    # ========================================
    # 3. Score distribution stats
    # ========================================
    print("\n" + "=" * 60)
    print("3. SCORE DISTRIBUTION STATS")
    print("=" * 60)

    for variant in ["EN_binary", "EN_triclass", "ZH_binary", "ZH_triclass"]:
        if variant not in available:
            continue

        scores, labels, vids = load_scores_and_labels(variant)
        dist_stats = score_distribution_stats(scores, labels)
        all_results[variant]["score_distribution"] = dist_stats

        print(f"\n  {variant}:")
        for cls in ["positive", "negative", "all"]:
            s = dist_stats[cls]
            if s["n"] == 0:
                continue
            print(f"    {cls:<10}: n={s['n']:>3d}  mean={s['mean']:.4f}  std={s['std']:.4f}  "
                  f"min={s['min']:.4f}  max={s['max']:.4f}  "
                  f"median={s['median']:.4f}  [Q25={s['q25']:.4f}, Q75={s['q75']:.4f}]")

    # ========================================
    # 4. Binary vs Triclass comparison
    # ========================================
    print("\n" + "=" * 60)
    print("4. BINARY vs TRICLASS COMPARISON")
    print("=" * 60)

    for lang in ["EN", "ZH"]:
        b_key = f"{lang}_binary"
        t_key = f"{lang}_triclass"
        if b_key not in available or t_key not in available:
            print(f"  {lang}: skipping (missing data)")
            continue

        comp = compare_binary_triclass(b_key, t_key)
        all_results[f"{lang}_comparison"] = comp

        print(f"\n  {lang} ({comp['n_common']} common videos):")
        print(f"    Binary thresh={comp['thresh_binary']:.2f}, Triclass thresh={comp['thresh_triclass']:.2f}")
        print(f"    Both correct:  {comp['both_correct']}")
        print(f"    Both wrong:    {comp['both_wrong']}")
        print(f"    Fixed by triclass (wrong->correct): {comp['flipped_to_correct_in_triclass']}")
        print(f"    Broken by triclass (correct->wrong): {comp['flipped_to_wrong_in_triclass']}")

        if comp['flipped_to_correct_in_triclass'] > 0:
            print(f"    Videos fixed by triclass:")
            for item in comp['flipped_to_correct_details'][:10]:
                print(f"      {item['video_id']}: gt={item['gt']} binary_score={item['score_binary']:.4f} triclass_score={item['score_triclass']:.4f}")

        if comp['flipped_to_wrong_in_triclass'] > 0:
            print(f"    Videos broken by triclass:")
            for item in comp['flipped_to_wrong_details'][:10]:
                print(f"      {item['video_id']}: gt={item['gt']} binary_score={item['score_binary']:.4f} triclass_score={item['score_triclass']:.4f}")

    # ========================================
    # 5. Unsupervised threshold analysis
    # ========================================
    print("\n" + "=" * 60)
    print("5. UNSUPERVISED THRESHOLD ANALYSIS")
    print("=" * 60)

    for variant in ["EN_binary", "EN_triclass", "ZH_binary", "ZH_triclass"]:
        if variant not in available:
            continue

        scores, labels, vids = load_scores_and_labels(variant)
        ut = unsupervised_threshold_analysis(scores, labels, variant)
        all_results[variant]["unsupervised_thresholds"] = ut

        print(f"\n  {variant}:")
        print(f"    Oracle:   thresh={ut['oracle_threshold']:.2f}  ACC={ut['oracle_acc']:.4f}")
        print(f"    Otsu:     thresh={ut['otsu_threshold']:.4f}  ACC={ut['otsu_acc']:.4f}  gap={ut['otsu_acc_gap']:.4f}")
        print(f"    GMM:      thresh={ut['gmm_threshold']:.4f}  ACC={ut['gmm_acc']:.4f}  gap={ut['gmm_acc_gap']:.4f}  flipped={ut['gmm_flipped']}")
        print(f"    GMM means: {ut['gmm_details']['means']}, stds: {ut['gmm_details']['stds']}, weights: {ut['gmm_details']['weights']}")

    # ========================================
    # 6. Bimodality analysis
    # ========================================
    print("\n" + "=" * 60)
    print("6. BIMODALITY ANALYSIS")
    print("=" * 60)

    bimodality_results = {}
    for variant in ["EN_binary", "EN_triclass", "ZH_binary", "ZH_triclass"]:
        if variant not in available:
            continue

        scores, labels, vids = load_scores_and_labels(variant)
        bc = bimodality_coefficient(scores)
        bimodality_results[variant] = bc
        all_results[variant]["bimodality_coefficient"] = bc

        bimodal_str = "BIMODAL" if bc > 0.555 else "NOT bimodal"
        print(f"  {variant}: BC={bc:.4f}  ({bimodal_str}, threshold=0.555)")

    if bimodality_results:
        best_variant = max(bimodality_results, key=bimodality_results.get)
        print(f"\n  Most bimodal: {best_variant} (BC={bimodality_results[best_variant]:.4f})")
        print(f"  -> Best candidate for label-free proxy signal")

    all_results["bimodality_summary"] = bimodality_results

    # ========================================
    # 7. Per-class error breakdown
    # ========================================
    print("\n" + "=" * 60)
    print("7. PER-CLASS ERROR BREAKDOWN (Hateful / Offensive / Normal)")
    print("=" * 60)

    for variant in ["EN_binary", "EN_triclass", "ZH_binary", "ZH_triclass"]:
        if variant not in available:
            continue

        dataset = DATASET_MAP[variant]
        scores, labels, vids = load_scores_and_labels(variant)
        breakdown = per_class_error_breakdown(scores, labels, vids, variant, dataset)
        all_results[variant]["per_class_breakdown"] = breakdown

        print(f"\n  {variant}:")
        for cat in ["Hateful", "Offensive", "Normal"]:
            if cat in breakdown:
                b = breakdown[cat]
                print(f"    {cat:<10}: n={b['n']:>3d}  acc={b['acc']:.4f}  avg_score={b['avg_score']:.4f}  "
                      f"correct={b['n_correct']}/{b['n']}  wrong={b['n_wrong']}")

    # ========================================
    # 8. Gap analysis
    # ========================================
    print("\n" + "=" * 60)
    print("8. GAP ANALYSIS (target: 80% ACC)")
    print("=" * 60)

    target_acc = 0.80
    for variant in ["EN_binary", "EN_triclass", "ZH_binary", "ZH_triclass"]:
        if variant not in available:
            continue

        b = all_results[variant]["best_acc"]
        gap = target_acc - b["acc"]
        n = all_results[variant]["n_samples"]
        n_wrong = b["fp"] + b["fn"]
        n_to_fix = max(0, int(np.ceil(gap * n)))

        print(f"\n  {variant}:")
        print(f"    Current ACC: {b['acc']:.4f} ({int(b['acc']*n)}/{n} correct)")
        print(f"    Target ACC:  {target_acc:.4f} ({int(target_acc*n)}/{n} correct)")
        print(f"    Gap:         {gap:.4f} ({n_to_fix} more videos needed)")
        print(f"    Errors:      FP={b['fp']} (Normal->Hate), FN={b['fn']} (Hate->Normal)")

        if b["fp"] > b["fn"]:
            print(f"    Dominant error: FALSE POSITIVES (over-predicting hate)")
            print(f"    -> Model is too sensitive / threshold too low")
        elif b["fn"] > b["fp"]:
            print(f"    Dominant error: FALSE NEGATIVES (missing hate)")
            print(f"    -> Model is not sensitive enough / threshold too high")
        else:
            print(f"    Error types balanced")

    all_results["gap_target"] = target_acc

    # ========================================
    # 9. Unsupervised vs Oracle summary
    # ========================================
    print("\n" + "=" * 60)
    print("9. LABEL-FREE VIABILITY SUMMARY")
    print("=" * 60)

    for variant in ["EN_binary", "EN_triclass", "ZH_binary", "ZH_triclass"]:
        if variant not in available or "unsupervised_thresholds" not in all_results[variant]:
            continue

        ut = all_results[variant]["unsupervised_thresholds"]
        bc = all_results[variant].get("bimodality_coefficient", 0)

        # Best unsupervised method
        best_unsup = "Otsu" if ut["otsu_acc"] >= ut["gmm_acc"] else "GMM"
        best_unsup_acc = max(ut["otsu_acc"], ut["gmm_acc"])
        best_unsup_gap = ut["oracle_acc"] - best_unsup_acc

        print(f"  {variant}:")
        print(f"    Oracle ACC:          {ut['oracle_acc']:.4f}")
        print(f"    Best unsupervised:   {best_unsup} ACC={best_unsup_acc:.4f}")
        print(f"    Label-free cost:     {best_unsup_gap:.4f} ACC drop")
        print(f"    Bimodality (BC):     {bc:.4f} {'(good)' if bc > 0.555 else '(poor - distribution not separable enough)'}")
        print(f"    Viable for 80%?      {'YES' if best_unsup_acc >= 0.80 else 'NO'} (need {max(0, 0.80 - best_unsup_acc):.4f} more)")

    # Save results
    out_path = os.path.join(PROJECT_ROOT, "results/analysis/iteration_0.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            converted = convert(obj)
            if converted is not obj:
                return converted
            return super().default(obj)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)

    print(f"\n\nResults saved to {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
