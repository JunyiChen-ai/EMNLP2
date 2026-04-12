"""Evaluate test-fit label-free performance on all existing triclass score files.

For each triclass file:
- Load scores (score = p_hateful + p_offensive)
- Fit Otsu and GMM on the test score distribution (no labels)
- Apply threshold and compute ACC
- Also report oracle ACC for comparison
"""

import json
import os
import sys
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_annotations


def otsu_threshold(scores):
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
        within_var = w0 * c0.var() + w1 * c1.var()
        if within_var < best_variance:
            best_variance = within_var
            best_thresh = t
    return float(best_thresh)


def gmm_threshold(scores):
    from sklearn.mixture import GaussianMixture
    X = scores.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
    gmm.fit(X)
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()
    if means[0] < means[1]:
        low_idx, high_idx = 0, 1
    else:
        low_idx, high_idx = 1, 0
    # Sweep between means, find where posterior of high = 0.5
    from scipy.stats import norm
    xs = np.linspace(means[low_idx], means[high_idx], 2000)
    best_t = (means[low_idx] + means[high_idx]) / 2
    for x in xs:
        p_low = weights[low_idx] * norm.pdf(x, means[low_idx], stds[low_idx])
        p_high = weights[high_idx] * norm.pdf(x, means[high_idx], stds[high_idx])
        post_high = p_high / (p_low + p_high + 1e-12)
        if post_high >= 0.5:
            best_t = x
            break
    return float(best_t)


def load_triclass_scores(path):
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            vid = r.get("video_id")
            s = r.get("score")
            if vid and s is not None:
                out[vid] = float(s)
    return out


def evaluate(file_path, dataset):
    name = os.path.basename(file_path).replace(".jsonl", "")
    model_dir = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
    scores_dict = load_triclass_scores(file_path)
    ann = load_annotations(dataset)

    scores_list, labels_list = [], []
    for vid, s in scores_dict.items():
        if vid not in ann:
            continue
        lbl = ann[vid]["label"]
        gt = 1 if lbl in ("Hateful", "Offensive") else 0
        scores_list.append(s)
        labels_list.append(gt)

    scores = np.array(scores_list)
    labels = np.array(labels_list)
    n = len(labels)

    def compute_metrics(thresh):
        preds = (scores >= thresh).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        acc = (tp + tn) / n
        return acc, tp, fp, fn, tn

    # Oracle sweep
    best_acc_oracle = 0
    best_t_oracle = 0.5
    for t in np.arange(0.0, 1.001, 0.01):
        acc, _, _, _, _ = compute_metrics(t)
        if acc > best_acc_oracle:
            best_acc_oracle = acc
            best_t_oracle = t

    # Test-fit thresholds
    otsu_t = otsu_threshold(scores)
    gmm_t = gmm_threshold(scores)
    otsu_acc, *_ = compute_metrics(otsu_t)
    gmm_acc, *_ = compute_metrics(gmm_t)

    return {
        "file": f"{model_dir}/{dataset}/{name}",
        "n": n,
        "n_pos": int(labels.sum()),
        "oracle_acc": float(best_acc_oracle),
        "oracle_thresh": float(best_t_oracle),
        "testfit_otsu_acc": float(otsu_acc),
        "testfit_otsu_thresh": float(otsu_t),
        "testfit_gmm_acc": float(gmm_acc),
        "testfit_gmm_thresh": float(gmm_t),
    }


def main():
    results = []
    patterns = [
        ("MHClip_EN", "/data/jehc223/EMNLP2/results/holistic_2b/MHClip_EN/test_triclass*.jsonl"),
        ("MHClip_ZH", "/data/jehc223/EMNLP2/results/holistic_2b/MHClip_ZH/test_triclass*.jsonl"),
        ("MHClip_EN", "/data/jehc223/EMNLP2/results/holistic_8b/MHClip_EN/test_triclass*.jsonl"),
        ("MHClip_ZH", "/data/jehc223/EMNLP2/results/holistic_8b/MHClip_ZH/test_triclass*.jsonl"),
    ]
    for dataset, pattern in patterns:
        for file_path in sorted(glob.glob(pattern)):
            if ".backup" in file_path:
                continue
            try:
                r = evaluate(file_path, dataset)
                results.append(r)
            except Exception as e:
                print(f"Failed {file_path}: {e}")

    # Print sorted table
    print(f"{'File':<60} {'N':<5} {'Oracle':<8} {'TF-Otsu':<8} {'TF-GMM':<8}")
    print("-" * 100)
    for r in results:
        print(f"{r['file']:<60} {r['n']:<5} {r['oracle_acc']:<8.4f} {r['testfit_otsu_acc']:<8.4f} {r['testfit_gmm_acc']:<8.4f}")

    # Save
    os.makedirs("/data/jehc223/EMNLP2/results/analysis", exist_ok=True)
    with open("/data/jehc223/EMNLP2/results/analysis/triclass_testfit.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: results/analysis/triclass_testfit.json")


if __name__ == "__main__":
    main()
