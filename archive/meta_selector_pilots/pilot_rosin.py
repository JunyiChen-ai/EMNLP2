"""Rosin's unimodal thresholding (Paul Rosin, Pattern Recognition 2001).

Reference: Rosin, P. L. (2001). "Unimodal thresholding." Pattern Recognition,
34(11), 2083-2096.

Method:
1. Build histogram of scores with n_bins bins (published default: 256).
2. Find the peak: largest bin (the dominant mode).
3. Find the rightmost non-empty bin.
4. Draw a straight line from peak top to rightmost bin top.
5. For each bin between peak and rightmost, compute perpendicular distance
   from line to bin top.
6. Threshold = bin with maximum perpendicular distance.

No free parameters beyond n_bins (which is fixed at 256 in the original paper).

Rationale for hateful video: ZH pool is a J-shaped unimodal distribution (dense
near 0, thin tail). Rosin's method is designed exactly for this shape: find the
elbow of the tail. Otsu fails on unimodal distributions; Rosin is the published
alternative.
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
from data_utils import load_annotations

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    return np.array(list(train.values()), dtype=float), test_x, test_y


def rosin_threshold(scores, n_bins=256):
    """Rosin's unimodal thresholding. Published standard n_bins=256."""
    s = np.asarray(scores, dtype=float)
    hist, edges = np.histogram(s, bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2

    # Find the peak (max bin count)
    peak_idx = int(np.argmax(hist))
    peak_val = hist[peak_idx]
    peak_pos = centers[peak_idx]

    # Find the rightmost non-zero bin
    non_zero = np.where(hist > 0)[0]
    if len(non_zero) == 0:
        return float(np.median(s))
    rightmost_idx = int(non_zero[-1])
    rightmost_pos = centers[rightmost_idx]
    rightmost_val = hist[rightmost_idx]

    if rightmost_idx <= peak_idx:
        # All mass is left of peak - no tail to threshold
        return float(np.median(s))

    # Line from (peak_pos, peak_val) to (rightmost_pos, rightmost_val)
    # Parametric form: P = peak + t*(rightmost - peak)
    # We want perpendicular distance from each bin point (x_i, y_i) to this line
    p1 = np.array([peak_pos, peak_val], dtype=float)
    p2 = np.array([rightmost_pos, rightmost_val], dtype=float)
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-12:
        return float(np.median(s))
    # Normal vector perpendicular to line
    line_norm = np.array([-line_vec[1], line_vec[0]]) / line_len

    best_dist = -1
    best_i = peak_idx
    for i in range(peak_idx, rightmost_idx + 1):
        pt = np.array([centers[i], hist[i]])
        # Perpendicular distance = |(pt - p1) . line_norm|
        d = abs(np.dot(pt - p1, line_norm))
        if d > best_dist:
            best_dist = d
            best_i = i

    return float(centers[best_i])


# Apply to pool, test, and train for comparison
print("=== Rosin unimodal thresholding ===")
for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    pool = np.concatenate([train, test_x])
    acc_b, mf_b = BASE[d]
    print(f"\n{d}: baseline {acc_b:.4f}/{mf_b:.4f}")

    for name, scores in [("pool", pool), ("train", train), ("test", test_x)]:
        t = rosin_threshold(scores, n_bins=256)
        m = metrics(test_x, test_y, t)
        tag = "STRICT" if m["acc"] > acc_b and m["mf"] > mf_b else ""
        print(f"  Rosin on {name:5s}: t={t:.4f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}  {tag}")

    # Also try different bin counts to test robustness
    print(f"  [bin sensitivity on pool]:")
    for nb in [64, 128, 200, 256, 300, 512]:
        t = rosin_threshold(pool, n_bins=nb)
        m = metrics(test_x, test_y, t)
        tag = "STRICT" if m["acc"] > acc_b and m["mf"] > mf_b else ""
        print(f"    nb={nb:4d}: t={t:.4f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}  {tag}")
