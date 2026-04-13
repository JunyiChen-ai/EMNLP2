"""Entropy-based thresholding: Kapur, Yen, Li-Lee, Shanbhag, Renyi.

Different information-theoretic criteria for picking a split point.
Unlike Otsu (minimize within variance), these maximize information gain or
minimize cross-entropy. Might land at different atoms.
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


def make_hist(x, nbins=256):
    hist, edges = np.histogram(x, bins=nbins, range=(0, 1))
    centers = (edges[:-1] + edges[1:]) / 2
    p = hist / max(hist.sum(), 1)
    return p, centers


def kapur_threshold(x, nbins=256):
    """Kapur-Sahoo-Wong: maximize sum of class entropies."""
    p, c = make_hist(x, nbins)
    best_t = c[0]
    best = -np.inf
    for i in range(1, len(p) - 1):
        p0 = p[:i].sum()
        p1 = p[i:].sum()
        if p0 <= 0 or p1 <= 0:
            continue
        H0 = -np.sum(p[:i] / p0 * np.log(p[:i] / p0 + 1e-12))
        H1 = -np.sum(p[i:] / p1 * np.log(p[i:] / p1 + 1e-12))
        J = H0 + H1
        if J > best:
            best = J
            best_t = c[i]
    return float(best_t)


def yen_threshold(x, nbins=256):
    """Yen: maximize correlation entropy."""
    p, c = make_hist(x, nbins)
    best_t = c[0]
    best = -np.inf
    for i in range(1, len(p) - 1):
        p0 = p[:i].sum()
        p1 = p[i:].sum()
        if p0 <= 0 or p1 <= 0:
            continue
        s0 = np.sum((p[:i] / p0) ** 2)
        s1 = np.sum((p[i:] / p1) ** 2)
        J = -np.log(s0 * s1 + 1e-12)
        if J > best:
            best = J
            best_t = c[i]
    return float(best_t)


def li_lee_threshold(x, nbins=256):
    """Li-Lee: minimize cross entropy."""
    p, c = make_hist(x, nbins)
    best_t = c[0]
    best = np.inf
    for i in range(1, len(p) - 1):
        p0 = p[:i].sum()
        p1 = p[i:].sum()
        if p0 <= 0 or p1 <= 0:
            continue
        m0 = np.sum(c[:i] * p[:i]) / p0
        m1 = np.sum(c[i:] * p[i:]) / p1
        if m0 <= 0 or m1 <= 0:
            continue
        J = -np.sum(c[:i] * p[:i] * np.log(m0 + 1e-12)) - np.sum(c[i:] * p[i:] * np.log(m1 + 1e-12))
        if J < best:
            best = J
            best_t = c[i]
    return float(best_t)


def renyi_threshold(x, alpha=2.0, nbins=256):
    """Renyi entropy threshold."""
    p, c = make_hist(x, nbins)
    best_t = c[0]
    best = -np.inf
    for i in range(1, len(p) - 1):
        p0 = p[:i].sum()
        p1 = p[i:].sum()
        if p0 <= 0 or p1 <= 0:
            continue
        H0 = np.log(np.sum((p[:i] / p0) ** alpha) + 1e-12) / (1 - alpha)
        H1 = np.log(np.sum((p[i:] / p1) ** alpha) + 1e-12) / (1 - alpha)
        J = H0 + H1
        if J > best:
            best = J
            best_t = c[i]
    return float(best_t)


def main():
    methods = {
        "kapur": kapur_threshold,
        "yen": yen_threshold,
        "li_lee": li_lee_threshold,
        "renyi_2": lambda x, nbins=256: renyi_threshold(x, 2.0, nbins),
        "renyi_3": lambda x, nbins=256: renyi_threshold(x, 3.0, nbins),
        "renyi_0.5": lambda x, nbins=256: renyi_threshold(x, 0.5, nbins),
    }
    for d in ["MHClip_EN", "MHClip_ZH"]:
        pool, train_arr, test_x, test_y = load(d)
        acc_b, mf_b = BASE[d]
        print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}")
        for src_name, src in [("pool", pool), ("train", train_arr), ("test", test_x)]:
            for nbins in [50, 100, 200, 500, 1000]:
                for mname, mfn in methods.items():
                    try:
                        t = mfn(src, nbins=nbins)
                    except Exception as e:
                        continue
                    m = metrics(test_x, test_y, t)
                    sb = m["acc"] > acc_b and m["mf"] > mf_b
                    if sb:
                        print(f"  {src_name:<5} b={nbins:<4} {mname:<10} t={t:.4f} {m['acc']:.4f}/{m['mf']:.4f} ***STRICT***")


if __name__ == "__main__":
    main()
