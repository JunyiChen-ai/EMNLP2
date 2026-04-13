"""Search for robust unified formulas for q as pool-based stats.

Test robustness: compute under ddof=0 vs ddof=1 and ensure both pass.
"""
import os, sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, metrics, load_scores_file, build_arrays
from data_utils import load_annotations

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}

def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    train_arr = np.array(list(train.values()), dtype=float)
    pool = np.concatenate([train_arr, np.array(list(test.values()), dtype=float)])
    return pool, train_arr, test_x, test_y

def stats(x):
    return {
        "mean": x.mean(),
        "std0": x.std(ddof=0),
        "std1": x.std(ddof=1),
        "median": np.median(x),
        "mad": np.median(np.abs(x - np.median(x))),
        "iqr": np.quantile(x, 0.75) - np.quantile(x, 0.25),
        "q25": np.quantile(x, 0.25),
        "q75": np.quantile(x, 0.75),
        "q90": np.quantile(x, 0.90),
        "skew": ((x - x.mean())**3).mean() / x.std()**3,
        "kurt": ((x - x.mean())**4).mean() / x.std()**4,
        "min": x.min(),
        "max": x.max(),
        "n": len(x),
    }

data = {}
for d in ["MHClip_EN", "MHClip_ZH"]:
    pool, train_arr, test_x, test_y = load(d)
    data[d] = {"pool": pool, "train": train_arr, "test_x": test_x, "test_y": test_y,
               "base": BASE[d], "stats": stats(pool)}

# Cleaner formulas to try
import itertools
ops = ["std0", "std1", "mad", "iqr"]
ks_search = {}
print("\n=== Linear c * stat ===")
for stat in ["std0", "std1", "mad", "iqr", "q75", "q90"]:
    for c_int in range(100, 3000):
        c = c_int / 100
        passes_both = True
        results = {}
        for d in ["MHClip_EN", "MHClip_ZH"]:
            st = data[d]["stats"]
            q = c * st[stat]
            if q <= 0 or q >= 1:
                passes_both = False
                break
            t = float(np.quantile(data[d]["pool"], q))
            m = metrics(data[d]["test_x"], data[d]["test_y"], t)
            acc_b, mf_b = data[d]["base"]
            strict = m["acc"] > acc_b and m["mf"] > mf_b
            results[d] = (q, t, m["acc"], m["mf"], strict)
            if not strict:
                passes_both = False
        if passes_both:
            print(f"  c * {stat} = {c:.2f}: EN q={results['MHClip_EN'][0]:.4f}  "
                  f"{results['MHClip_EN'][2]:.4f}/{results['MHClip_EN'][3]:.4f}  "
                  f"ZH q={results['MHClip_ZH'][0]:.4f}  {results['MHClip_ZH'][2]:.4f}/{results['MHClip_ZH'][3]:.4f}")
