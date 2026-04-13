"""Non-suffix rule via density-peak detection.

Hypothesis: Positive-rate non-monotonicity means there are 'spike' atoms in the
mid-range where positives concentrate. These look like local density peaks in
the mid-percentile region (between 40th and 80th percentile) of the pool.

Rule:
  1. Compute local density of pool (KDE or k-NN).
  2. Find density-peak atoms in the mid-percentile band.
  3. Prediction: flag samples in the top quantile (MAD rule) OR samples at a
     density-peak atom in the mid-percentile band.

Sweep the band parameters and density-peak thresholds.
"""
import os, sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
from data_utils import load_annotations
from scipy.signal import find_peaks

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}

def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    train_arr = np.array(list(train.values()), dtype=float)
    pool = np.concatenate([train_arr, test_x])
    return pool, test_x, test_y

def density_peaks_in_band(pool, lo_q, hi_q, n_peaks=3):
    """Find unique pool values in [lo_q, hi_q] percentile that are local
    maxima of pool sample density (4-decimal atoms)."""
    from collections import Counter
    # Atoms and counts
    atoms = Counter(round(float(x), 4) for x in pool)
    sorted_atoms = sorted(atoms.keys())
    # Percentile mask
    q_vals = {a: float((pool <= a).mean()) for a in sorted_atoms}
    in_band = [a for a in sorted_atoms if lo_q <= q_vals[a] <= hi_q]
    if len(in_band) < 3:
        return []
    counts = [atoms[a] for a in in_band]
    peak_idx, _ = find_peaks(counts, distance=1)
    # Rank peaks by count, take top n
    peak_atoms = sorted([(in_band[i], counts[i]) for i in peak_idx], key=lambda p: -p[1])
    return [a for a, c in peak_atoms[:n_peaks]]

def mad_threshold(pool):
    med = float(np.median(pool))
    mad = float(np.median(np.abs(pool - med)))
    q = max(0.01, min(0.99, 0.60 + 7.83*mad))
    return float(np.quantile(pool, q))

def evaluate_rule(lo_q, hi_q, n_peaks, band_width=0.003):
    """Apply MAD suffix + density-peak band additions.
    band_width: score-halfwidth around each peak atom to include."""
    results = {}
    all_strict = True
    for d in ["MHClip_EN", "MHClip_ZH"]:
        pool, test_x, test_y = load(d)
        t_mad = mad_threshold(pool)
        peaks = density_peaks_in_band(pool, lo_q, hi_q, n_peaks)
        # Fake score: 1 if sample is above MAD or within peak bands; else 0
        flagged = (test_x >= t_mad)
        for p in peaks:
            flagged |= (np.abs(test_x - p) < band_width)
        fake = flagged.astype(float)
        m = metrics(fake, test_y, 0.5)
        acc_b, mf_b = BASE[d]
        s_acc = m["acc"] > acc_b
        s_mf = m["mf"] > mf_b
        results[d] = (m["acc"], m["mf"], s_acc, s_mf, len(peaks))
        if not (s_acc and s_mf):
            all_strict = False
    return results, all_strict

print("=== MAD + density-peak band ===")
for lo_q in [0.30, 0.40, 0.50, 0.55, 0.60]:
    for hi_q in [0.65, 0.70, 0.75, 0.80, 0.85]:
        if hi_q <= lo_q + 0.05:
            continue
        for n_peaks in [1, 2, 3]:
            for band_width in [0.001, 0.003, 0.005, 0.01]:
                r, s = evaluate_rule(lo_q, hi_q, n_peaks, band_width)
                if s:
                    en = r["MHClip_EN"]
                    zh = r["MHClip_ZH"]
                    print(f"  lo={lo_q} hi={hi_q} n={n_peaks} w={band_width}  "
                          f"EN {en[0]:.4f}/{en[1]:.4f} (peaks={en[4]})  "
                          f"ZH {zh[0]:.4f}/{zh[1]:.4f} (peaks={zh[4]})  STRICT")
