"""KDE valley-finding thresholds.

Find the deepest valley in the pool KDE between the lowest and highest modes.
This responds to shape differences (EN has visible tail modes, ZH is J-shaped)
and may land at different absolute values for EN and ZH.
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


def kde(x, bw, grid):
    """Gaussian KDE at each grid point."""
    diffs = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * diffs ** 2).sum(axis=1) / (len(x) * bw * np.sqrt(2 * np.pi))


def find_valleys(density):
    """Return indices of local minima."""
    valleys = []
    for i in range(1, len(density) - 1):
        if density[i] < density[i - 1] and density[i] < density[i + 1]:
            valleys.append(i)
    return valleys


def find_peaks(density):
    peaks = []
    for i in range(1, len(density) - 1):
        if density[i] > density[i - 1] and density[i] > density[i + 1]:
            peaks.append(i)
    return peaks


def silverman_bw(x):
    n = len(x)
    s = x.std()
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    sigma = min(s, iqr / 1.349)
    return 0.9 * sigma * n ** (-1 / 5)


def main():
    for d in ["MHClip_EN", "MHClip_ZH"]:
        pool, train_arr, test_x, test_y = load(d)
        acc_b, mf_b = BASE[d]
        print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}")
        for src_name, src in [("pool", pool), ("train", train_arr), ("test", test_x)]:
            for bw_mult in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
                bw = silverman_bw(src) * bw_mult
                if bw <= 0:
                    continue
                grid = np.linspace(0, 1, 1001)
                dens = kde(src, bw, grid)
                peaks = find_peaks(dens)
                valleys = find_valleys(dens)
                if len(peaks) < 2 or len(valleys) < 1:
                    continue
                # pick the deepest valley BETWEEN the two highest peaks
                for v in valleys:
                    t = grid[v]
                    m = metrics(test_x, test_y, t)
                    sb = m["acc"] > acc_b and m["mf"] > mf_b
                    tag = " ***STRICT***" if sb else ""
                    if m["acc"] > acc_b or m["mf"] > mf_b:
                        print(f"  {src_name:<6} bw*{bw_mult:<4}  valley t={t:.4f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}{tag}")


if __name__ == "__main__":
    main()
