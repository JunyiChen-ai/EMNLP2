"""Quick eval: compute test-fit and train-derived metrics for all existing score files.

For each (model, dataset, config) combo with a test file:
- Test-fit Otsu/GMM: fit on test scores, apply on test
- Train-derived Otsu/GMM: fit on train scores (if available), apply on test
Metrics: ACC, macro-F1, macro-Precision, macro-Recall
"""

import json
import os
import sys
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_annotations


def otsu_threshold(scores):
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
        within_var = w0 * c0.var() + w1 * c1.var()
        if within_var < best_variance:
            best_variance = within_var
            best_thresh = t
    return float(best_thresh)


def gmm_threshold(scores):
    from sklearn.mixture import GaussianMixture
    from scipy.stats import norm
    X = np.asarray(scores).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
    gmm.fit(X)
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()
    if means[0] < means[1]:
        low, high = 0, 1
    else:
        low, high = 1, 0
    xs = np.linspace(means[low], means[high], 2000)
    best = (means[low] + means[high]) / 2
    for x in xs:
        p_lo = weights[low] * norm.pdf(x, means[low], stds[low])
        p_hi = weights[high] * norm.pdf(x, means[high], stds[high])
        post = p_hi / (p_lo + p_hi + 1e-12)
        if post >= 0.5:
            best = x
            break
    return float(best)


def metrics(scores, labels, thresh):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    preds = (scores >= thresh).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    n = len(labels)
    acc = (tp + tn) / n if n else 0

    # Positive class
    p_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
    r_pos = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_pos = 2 * p_pos * r_pos / (p_pos + r_pos) if (p_pos + r_pos) > 0 else 0
    # Negative class
    p_neg = tn / (tn + fn) if (tn + fn) > 0 else 0
    r_neg = tn / (tn + fp) if (tn + fp) > 0 else 0
    f_neg = 2 * p_neg * r_neg / (p_neg + r_neg) if (p_neg + r_neg) > 0 else 0

    macro_p = (p_pos + p_neg) / 2
    macro_r = (r_pos + r_neg) / 2
    macro_f = (f_pos + f_neg) / 2
    return {"acc": acc, "mp": macro_p, "mr": macro_r, "mf": macro_f,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def load_scores_file(path):
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


def build_arrays(scores_dict, ann, include_labels=True):
    xs, ys = [], []
    for vid, s in scores_dict.items():
        if vid not in ann:
            continue
        lbl = ann[vid]["label"]
        gt = 1 if lbl in ("Hateful", "Offensive") else 0
        xs.append(s)
        ys.append(gt)
    return np.array(xs), np.array(ys)


CONFIGS = {
    "binary_nodef": ("binary", ""),
    "binary_withdef": ("binary", "_withdef"),
    "binary_deflected": ("binary", "_deflected"),
    "binary_minimal": ("binary", "_minimal"),
    "triclass_narrow": ("triclass", ""),
    "triclass_broad": ("triclass", "_broad"),
    "triclass_nodef": ("triclass", "_nodef"),
    "triclass_broad_t1000": ("triclass", "_broad_t1000"),
    "triclass_t1000": ("triclass", "_t1000"),
    "triclass_norules_t1000": ("triclass", "_norules_t1000"),
}


def main():
    rows = []
    for model_tag, model_dir in [("2B", "holistic_2b"), ("8B", "holistic_8b")]:
        for dataset in ["MHClip_EN", "MHClip_ZH"]:
            ann = load_annotations(dataset)
            base = f"/data/jehc223/EMNLP2/results/{model_dir}/{dataset}"
            for cfg_name, (mode, suffix) in CONFIGS.items():
                test_path = f"{base}/test_{mode}{suffix}.jsonl"
                train_path = f"{base}/train_{mode}{suffix}.jsonl"
                if not os.path.isfile(test_path):
                    continue

                test_dict = load_scores_file(test_path)
                test_x, test_y = build_arrays(test_dict, ann)
                if len(test_x) == 0:
                    continue

                row = {
                    "model": model_tag,
                    "dataset": dataset,
                    "config": cfg_name,
                    "n_test": len(test_x),
                    "n_test_pos": int(test_y.sum()),
                }

                # Oracle (diagnostic upper bound — uses test labels)
                best_acc = 0.0
                best_t = 0.5
                for t in np.arange(0.0, 1.001, 0.01):
                    m = metrics(test_x, test_y, float(t))
                    if m["acc"] > best_acc:
                        best_acc = m["acc"]
                        best_t = float(t)
                row["oracle"] = metrics(test_x, test_y, best_t)
                row["oracle"]["t"] = best_t

                # Test-fit
                try:
                    tf_otsu_t = otsu_threshold(test_x)
                    row["tf_otsu"] = metrics(test_x, test_y, tf_otsu_t)
                    row["tf_otsu"]["t"] = tf_otsu_t
                except Exception as e:
                    row["tf_otsu"] = {"err": str(e)}
                try:
                    tf_gmm_t = gmm_threshold(test_x)
                    row["tf_gmm"] = metrics(test_x, test_y, tf_gmm_t)
                    row["tf_gmm"]["t"] = tf_gmm_t
                except Exception as e:
                    row["tf_gmm"] = {"err": str(e)}

                # Train-derived (only if train file exists)
                if os.path.isfile(train_path):
                    train_dict = load_scores_file(train_path)
                    train_scores = np.array(list(train_dict.values()))
                    row["n_train"] = len(train_scores)
                    try:
                        tr_otsu_t = otsu_threshold(train_scores)
                        row["tr_otsu"] = metrics(test_x, test_y, tr_otsu_t)
                        row["tr_otsu"]["t"] = tr_otsu_t
                    except Exception as e:
                        row["tr_otsu"] = {"err": str(e)}
                    try:
                        tr_gmm_t = gmm_threshold(train_scores)
                        row["tr_gmm"] = metrics(test_x, test_y, tr_gmm_t)
                        row["tr_gmm"]["t"] = tr_gmm_t
                    except Exception as e:
                        row["tr_gmm"] = {"err": str(e)}

                rows.append(row)

    # Print summary table
    def fmt(m):
        if "err" in m:
            return "ERR"
        return f"{m['acc']:.3f}/{m['mf']:.3f}"

    print(f"{'Model':<4} {'Dataset':<12} {'Config':<25} {'N':<6} {'Oracle':<14} {'TF-Otsu':<14} {'TF-GMM':<14} {'Tr-Otsu':<14} {'Tr-GMM':<14}")
    print("-" * 140)
    for r in rows:
        oracle = fmt(r.get("oracle", {}))
        tf_o = fmt(r.get("tf_otsu", {}))
        tf_g = fmt(r.get("tf_gmm", {}))
        tr_o = fmt(r.get("tr_otsu", {"err": "no train"}))
        tr_g = fmt(r.get("tr_gmm", {"err": "no train"}))
        print(f"{r['model']:<4} {r['dataset']:<12} {r['config']:<25} {r['n_test']:<6} {oracle:<14} {tf_o:<14} {tf_g:<14} {tr_o:<14} {tr_g:<14}")

    os.makedirs("/data/jehc223/EMNLP2/results/analysis", exist_ok=True)
    with open("/data/jehc223/EMNLP2/results/analysis/quick_eval_all.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved: results/analysis/quick_eval_all.json")


if __name__ == "__main__":
    main()
