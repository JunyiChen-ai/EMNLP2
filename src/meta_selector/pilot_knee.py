"""CDF knee-point detection.

The pool CDF rises rapidly through the negative-heavy low-score region,
then slows in the positive-heavy tail. The knee (point of maximum curvature)
marks the boundary. Should adapt to different shapes (steeper knee for ZH's
J-shape, broader for EN's bimodal tail).
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


def knee_kneedle(x):
    """Kneedle algorithm: max distance from sorted data to line endpoints."""
    s = np.sort(np.asarray(x))
    n = len(s)
    y = np.arange(n) / max(n - 1, 1)
    # Line from (s[0], 0) to (s[-1], 1)
    if s[-1] == s[0]:
        return s[0]
    # Vector along the line
    x1, y1 = s[0], 0.0
    x2, y2 = s[-1], 1.0
    num = np.abs((y2 - y1) * s - (x2 - x1) * y + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    d = num / den
    # Use sign: knee should be BELOW the chord (CDF is concave going left→right→up)
    # We want maximum distance regardless of sign
    idx = np.argmax(d)
    return s[idx]


def knee_max_curvature(x):
    """Maximum curvature of smoothed CDF.

    CDF = F(x). We want point where second derivative is largest in magnitude.
    Approximate: use discrete histogram with bins = unique values.
    """
    s = np.sort(np.asarray(x))
    n = len(s)
    F = np.arange(1, n + 1) / n
    # Finite differences
    dx = np.diff(s)
    dx[dx == 0] = 1e-12
    dF = np.diff(F) / dx
    # second derivative
    d2F = np.diff(dF) / dx[:-1]
    idx = np.argmin(d2F)  # most concave down point (steepest drop in slope)
    return (s[idx + 1] + s[idx + 2]) / 2


def knee_kneedle_inverse(x):
    """Like kneedle but on 1-F (survival function)."""
    s = np.sort(np.asarray(x))
    n = len(s)
    y = 1 - np.arange(n) / max(n - 1, 1)
    x1, y1 = s[0], 1.0
    x2, y2 = s[-1], 0.0
    num = np.abs((y2 - y1) * s - (x2 - x1) * y + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    d = num / den
    idx = np.argmax(d)
    return s[idx]


def knee_elbow(x):
    """Elbow method in log(survival) space.

    log(1 - F(x)) vs x for heavy-tailed data typically has a bend near the transition."""
    s = np.sort(np.asarray(x))
    n = len(s)
    F = (np.arange(1, n + 1) - 0.5) / n
    logS = np.log(np.clip(1 - F, 1e-10, 1))
    # line from first to last
    x1, y1 = s[0], logS[0]
    x2, y2 = s[-1], logS[-1]
    num = np.abs((y2 - y1) * s - (x2 - x1) * logS + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    d = num / den
    idx = np.argmax(d)
    return s[idx]


def main():
    for d in ["MHClip_EN", "MHClip_ZH"]:
        pool, train_arr, test_x, test_y = load(d)
        acc_b, mf_b = BASE[d]
        print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}")
        for src_name, src in [("pool", pool), ("train", train_arr), ("test", test_x)]:
            for kname, fn in [("kneedle", knee_kneedle),
                              ("kneedle_inv", knee_kneedle_inverse),
                              ("max_curv", knee_max_curvature),
                              ("elbow_logS", knee_elbow)]:
                try:
                    t = float(fn(src))
                except Exception as e:
                    print(f"  {src_name:<6} {kname:<12} ERR {e}")
                    continue
                m = metrics(test_x, test_y, t)
                sb = m["acc"] > acc_b and m["mf"] > mf_b
                tag = " ***STRICT***" if sb else ""
                print(f"  {src_name:<6} {kname:<12}  t={t:.4f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}{tag}")


if __name__ == "__main__":
    main()
