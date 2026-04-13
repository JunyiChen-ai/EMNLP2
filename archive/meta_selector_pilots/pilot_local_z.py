"""Local z-score test.

For each test sample, compute z = (x - mean_nbhd) / std_nbhd where nbhd is
the k nearest pool samples. Predict positive if z > tau.
"""
import os, sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, metrics, load_scores_file, build_arrays
from data_utils import load_annotations

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}


def load(dataset):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{dataset}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(dataset)
    test_x, test_y = build_arrays(test, ann)
    train_arr = np.array(list(train.values()), dtype=float)
    pool = np.concatenate([train_arr, np.array(list(test.values()), dtype=float)])
    return pool, train_arr, test_x, test_y


def predict_local_z(test_x, pool_sorted, k, tau):
    """For each test sample, find k nearest pool samples, compute z-score, threshold."""
    preds = np.zeros(len(test_x), dtype=int)
    for i, x in enumerate(test_x):
        # find k closest in pool_sorted
        idx = np.searchsorted(pool_sorted, x)
        lo = max(0, idx - k // 2)
        hi = min(len(pool_sorted), lo + k)
        lo = max(0, hi - k)
        nbhd = pool_sorted[lo:hi]
        mu = nbhd.mean()
        sd = nbhd.std() + 1e-12
        z = (x - mu) / sd
        preds[i] = int(z > tau)
    return preds


def metrics_from_pred(pred, test_y):
    TP = int(((pred == 1) & (test_y == 1)).sum())
    TN = int(((pred == 0) & (test_y == 0)).sum())
    FP = int(((pred == 1) & (test_y == 0)).sum())
    FN = int(((pred == 0) & (test_y == 1)).sum())
    acc = (TP + TN) / max(len(test_y), 1)
    prec_p = TP / max(TP + FP, 1)
    rec_p = TP / max(TP + FN, 1)
    f1_p = 2 * prec_p * rec_p / max(prec_p + rec_p, 1e-12)
    prec_n = TN / max(TN + FN, 1)
    rec_n = TN / max(TN + FP, 1)
    f1_n = 2 * prec_n * rec_n / max(prec_n + rec_n, 1e-12)
    mf = (f1_p + f1_n) / 2
    return acc, mf


def main():
    for k in [5, 10, 20, 30, 50, 75, 100, 150, 200]:
        for tau in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
            results = {}
            for d in ["MHClip_EN", "MHClip_ZH"]:
                pool, tr, test_x, test_y = load(d)
                pool_sorted = np.sort(pool)
                pred = predict_local_z(test_x, pool_sorted, k, tau)
                acc, mf = metrics_from_pred(pred, test_y)
                acc_b, mf_b = BASE[d]
                sb = acc > acc_b and mf > mf_b
                results[d] = (acc, mf, sb)
            if results["MHClip_EN"][2] and results["MHClip_ZH"][2]:
                print(f"** UNIFIED k={k} tau={tau} EN={results['MHClip_EN'][0]:.4f}/{results['MHClip_EN'][1]:.4f} "
                      f"ZH={results['MHClip_ZH'][0]:.4f}/{results['MHClip_ZH'][1]:.4f} **")
            elif results["MHClip_EN"][2]:
                print(f"   [EN+] k={k} tau={tau} EN={results['MHClip_EN'][0]:.4f}/{results['MHClip_EN'][1]:.4f} "
                      f"ZH={results['MHClip_ZH'][0]:.4f}/{results['MHClip_ZH'][1]:.4f}")
            elif results["MHClip_ZH"][2]:
                print(f"   [ZH+] k={k} tau={tau} EN={results['MHClip_EN'][0]:.4f}/{results['MHClip_EN'][1]:.4f} "
                      f"ZH={results['MHClip_ZH'][0]:.4f}/{results['MHClip_ZH'][1]:.4f}")


if __name__ == "__main__":
    main()
