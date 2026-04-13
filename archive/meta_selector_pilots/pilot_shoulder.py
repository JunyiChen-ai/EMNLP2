"""Find the shoulder of the survival function log(1-F) on log-spaced grid.

The hypothesis: the hateful-video score distribution has a 'positive tail' where
the rate of decrease in survival slows. At that shoulder, the samples are
predominantly positive. Use the second derivative / curvature of log(1-F).
"""
import os, sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
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


def shoulder_threshold(src, method="max_curvature_logS"):
    """Find the shoulder of log(1-F) vs x."""
    s = np.sort(np.asarray(src))
    n = len(s)
    F = (np.arange(1, n + 1)) / (n + 1)  # avoid log(0)
    S = 1 - F
    logS = np.log(S)
    x = s
    if method == "max_curvature_logS":
        # approximate second derivative: use smoothed finite differences
        # smooth via moving average
        window = max(5, n // 20)
        if window % 2 == 0:
            window += 1
        half = window // 2
        smoothed = np.zeros_like(logS)
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            smoothed[i] = logS[lo:hi].mean()
        d1 = np.diff(smoothed) / (np.diff(x) + 1e-12)
        d2 = np.diff(d1) / (np.diff(x[:-1]) + 1e-12)
        # Shoulder = where d2 is maximum (log(1-F) starts decreasing faster/slower)
        idx = np.argmax(d2)
        return float(x[idx + 1])
    elif method == "min_d2_logS":
        window = max(5, n // 20)
        if window % 2 == 0:
            window += 1
        half = window // 2
        smoothed = np.zeros_like(logS)
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            smoothed[i] = logS[lo:hi].mean()
        d1 = np.diff(smoothed) / (np.diff(x) + 1e-12)
        d2 = np.diff(d1) / (np.diff(x[:-1]) + 1e-12)
        idx = np.argmin(d2)
        return float(x[idx + 1])
    elif method == "kneedle_logS":
        # Classic kneedle on log(1-F) vs x
        x1, y1 = x[0], logS[0]
        x2, y2 = x[-1], logS[-1]
        if x1 == x2:
            return float(x[0])
        # normalize
        xn = (x - x1) / (x2 - x1)
        yn = (logS - y2) / (y1 - y2)  # y1 high, y2 low -> normalized [0,1]
        # knee = max distance to diagonal line from (0,1) to (1,0)
        d = yn - (1 - xn)
        idx = np.argmax(d)
        return float(x[idx])
    elif method == "median_of_tail":
        # Find a rough bend via coarse histogram, then take median of upper tail
        hist, edges = np.histogram(src, bins=50)
        # Normalize and find first bin where hist is less than 20% of peak
        peak = hist.max()
        bend = None
        for i, h in enumerate(hist):
            if h < 0.2 * peak and i > np.argmax(hist):
                bend = (edges[i] + edges[i + 1]) / 2
                break
        if bend is None:
            return float(np.median(src))
        # return median of scores above bend
        tail = src[src > bend]
        if len(tail) == 0:
            return float(bend)
        return float(np.median(tail))
    return float(np.median(src))


def main():
    methods = ["max_curvature_logS", "min_d2_logS", "kneedle_logS", "median_of_tail"]
    for d in ["MHClip_EN", "MHClip_ZH"]:
        pool, train_arr, test_x, test_y = load(d)
        acc_b, mf_b = BASE[d]
        print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}")
        for src_name, src in [("pool", pool), ("train", train_arr), ("test", test_x)]:
            for mth in methods:
                t = shoulder_threshold(src, method=mth)
                m = metrics(test_x, test_y, t)
                sb = m["acc"] > acc_b and m["mf"] > mf_b
                tag = " ***STRICT***" if sb else ""
                print(f"  {src_name:<6} {mth:<22} t={t:.4f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}{tag}")


if __name__ == "__main__":
    main()
