"""Triangle method thresholding.

Zack 1977: build histogram, find peak, draw line from peak to tail end, find
the bin maximally far from the line. Doesn't assume bimodality. Works well on
skewed distributions where one class dominates.
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


def triangle_threshold(x, nbins=50):
    hist, edges = np.histogram(x, bins=nbins, range=(0, 1))
    centers = (edges[:-1] + edges[1:]) / 2
    # Find the peak (mode)
    peak = np.argmax(hist)
    # Line from peak to the right tail (far end)
    # The right tail is the last nonzero bin
    right = len(hist) - 1
    while hist[right] == 0 and right > peak:
        right -= 1
    if right <= peak:
        return float(centers[peak])
    # Find bin with max distance to line (peak_center, peak_h) -> (right_center, right_h)
    x1, y1 = centers[peak], hist[peak]
    x2, y2 = centers[right], hist[right]
    distances = []
    for i in range(peak, right + 1):
        xi, yi = centers[i], hist[i]
        num = abs((y2 - y1) * xi - (x2 - x1) * yi + x2 * y1 - y2 * x1)
        den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distances.append(num / den if den > 0 else 0)
    best = np.argmax(distances) + peak
    return float(centers[best])


def triangle_log_hist(x, nbins=50):
    """Triangle on log-histogram (emphasizes tail)."""
    hist, edges = np.histogram(x, bins=nbins, range=(0, 1))
    hist = np.log1p(hist)  # flatten peak
    centers = (edges[:-1] + edges[1:]) / 2
    peak = np.argmax(hist)
    right = len(hist) - 1
    while hist[right] == 0 and right > peak:
        right -= 1
    if right <= peak:
        return float(centers[peak])
    x1, y1 = centers[peak], hist[peak]
    x2, y2 = centers[right], hist[right]
    distances = []
    for i in range(peak, right + 1):
        xi, yi = centers[i], hist[i]
        num = abs((y2 - y1) * xi - (x2 - x1) * yi + x2 * y1 - y2 * x1)
        den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distances.append(num / den if den > 0 else 0)
    best = np.argmax(distances) + peak
    return float(centers[best])


def main():
    for d in ["MHClip_EN", "MHClip_ZH"]:
        pool, train_arr, test_x, test_y = load(d)
        acc_b, mf_b = BASE[d]
        print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}")
        for src_name, src in [("pool", pool), ("train", train_arr), ("test", test_x)]:
            for nbins in [20, 30, 40, 50, 60, 80, 100, 150, 200]:
                t = triangle_threshold(src, nbins=nbins)
                t_log = triangle_log_hist(src, nbins=nbins)
                m = metrics(test_x, test_y, t)
                m_log = metrics(test_x, test_y, t_log)
                sb = m["acc"] > acc_b and m["mf"] > mf_b
                sb_log = m_log["acc"] > acc_b and m_log["mf"] > mf_b
                tag = " ***" if sb else ""
                tag_log = " ***" if sb_log else ""
                print(f"  {src_name:<6} b={nbins:<4}  tri t={t:.4f} {m['acc']:.4f}/{m['mf']:.4f}{tag}  "
                      f"tri-log t={t_log:.4f} {m_log['acc']:.4f}/{m_log['mf']:.4f}{tag_log}")


if __name__ == "__main__":
    main()
