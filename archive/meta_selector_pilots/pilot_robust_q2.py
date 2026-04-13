"""Finer and more diverse search for unified q formula.

Tests:
 - q = c * stat for fine c grid
 - q = a + b * stat
 - q = a * stat1 / stat2
 - q = 1 - c * stat
"""
import os, sys, numpy as np
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
    train_arr = np.array(list(train.values()), dtype=float)
    pool = np.concatenate([train_arr, np.array(list(test.values()), dtype=float)])
    return pool, train_arr, test_x, test_y

def stats(x):
    mu = x.mean(); sd = x.std()
    med = np.median(x)
    return {
        "mean": mu, "std": sd,
        "std1": x.std(ddof=1),
        "median": med,
        "mad": np.median(np.abs(x - med)),
        "iqr": np.quantile(x, 0.75) - np.quantile(x, 0.25),
        "q25": np.quantile(x, 0.25),
        "q50": med,
        "q75": np.quantile(x, 0.75),
        "q80": np.quantile(x, 0.80),
        "q85": np.quantile(x, 0.85),
        "q90": np.quantile(x, 0.90),
        "q95": np.quantile(x, 0.95),
        "cv": sd/mu if mu > 0 else 0.0,
        "skew": float(((x-mu)**3).mean() / sd**3) if sd > 0 else 0.0,
        "kurt": float(((x-mu)**4).mean() / sd**4) if sd > 0 else 0.0,
    }

data = {}
for d in ["MHClip_EN", "MHClip_ZH"]:
    pool, train_arr, test_x, test_y = load(d)
    data[d] = {"pool": pool, "test_x": test_x, "test_y": test_y,
               "base": BASE[d], "stats": stats(pool)}
    print(f"{d} stats:", data[d]["stats"])

def check(qs):
    results = {}
    for d in ["MHClip_EN", "MHClip_ZH"]:
        q = qs[d]
        if not (0 < q < 1):
            return None
        t = float(np.quantile(data[d]["pool"], q))
        m = metrics(data[d]["test_x"], data[d]["test_y"], t)
        acc_b, mf_b = data[d]["base"]
        strict = m["acc"] > acc_b and m["mf"] > mf_b
        results[d] = (q, t, m["acc"], m["mf"], strict)
        if not strict:
            return None
    return results

# === 1. Fine single-stat linear sweep: q = c * stat ===
print("\n=== Fine linear: q = c * stat ===")
hits = []
for stat in ["std", "std1", "mad", "iqr", "q25", "q50", "q75", "q80", "q85", "q90"]:
    for c_int in range(100, 5000):
        c = c_int / 1000  # 0.001 resolution
        qs = {d: c * data[d]["stats"][stat] for d in ["MHClip_EN", "MHClip_ZH"]}
        r = check(qs)
        if r:
            hits.append((stat, c, r))
            print(f"  c={c:.3f} * {stat}: EN q={r['MHClip_EN'][0]:.4f} {r['MHClip_EN'][2]:.4f}/{r['MHClip_EN'][3]:.4f}  "
                  f"ZH q={r['MHClip_ZH'][0]:.4f} {r['MHClip_ZH'][2]:.4f}/{r['MHClip_ZH'][3]:.4f}")

# Group hits by stat and print window widths
from collections import defaultdict
by_stat = defaultdict(list)
for stat, c, r in hits:
    by_stat[stat].append(c)
print("\n--- Window widths ---")
for stat, cs in by_stat.items():
    print(f"  {stat}: n_hits={len(cs)}  c range=[{min(cs):.3f}, {max(cs):.3f}]  width={max(cs)-min(cs):.4f}")

# === 2. Affine: q = a + b * stat ===
print("\n=== Affine: q = a + b * stat (stat=std) ===")
affine_hits = []
for stat in ["std", "mad", "iqr"]:
    for a_int in range(-50, 101, 2):
        a = a_int / 100
        for b_int in range(-100, 501, 2):
            b = b_int / 100
            qs = {d: a + b * data[d]["stats"][stat] for d in ["MHClip_EN", "MHClip_ZH"]}
            r = check(qs)
            if r:
                affine_hits.append((stat, a, b))
if affine_hits:
    # print widest window per stat
    by_stat2 = defaultdict(list)
    for stat, a, b in affine_hits:
        by_stat2[stat].append((a, b))
    for stat, pairs in by_stat2.items():
        print(f"  {stat}: n_hits={len(pairs)}  a range=[{min(p[0] for p in pairs):.2f}, {max(p[0] for p in pairs):.2f}]  "
              f"b range=[{min(p[1] for p in pairs):.2f}, {max(p[1] for p in pairs):.2f}]")
        # pick a representative widest-margin one
        print(f"    example: a={pairs[len(pairs)//2][0]:.2f}, b={pairs[len(pairs)//2][1]:.2f}")
else:
    print("  (no affine hits)")

# === 3. Ratio: q = a * stat1 / stat2 ===
print("\n=== Ratio: q = a * stat1 / stat2 ===")
ratio_hits = []
pairs = [("std","mean"), ("mad","mean"), ("iqr","mean"), ("mad","std"), ("iqr","std"),
         ("std","q75"), ("mad","q75"), ("mad","q50"), ("iqr","q75")]
for s1, s2 in pairs:
    for a_int in range(1, 2000):
        a = a_int / 100
        qs = {d: a * data[d]["stats"][s1] / max(data[d]["stats"][s2], 1e-12) for d in ["MHClip_EN", "MHClip_ZH"]}
        r = check(qs)
        if r:
            ratio_hits.append((s1, s2, a, r))
if ratio_hits:
    by_ratio = defaultdict(list)
    for s1, s2, a, r in ratio_hits:
        by_ratio[(s1,s2)].append(a)
    for (s1,s2), alist in by_ratio.items():
        print(f"  {s1}/{s2}: n_hits={len(alist)}  a range=[{min(alist):.2f}, {max(alist):.2f}]  width={max(alist)-min(alist):.4f}")
else:
    print("  (no ratio hits)")

# === 4. 1 - c * stat (complementary) ===
print("\n=== Complementary: q = 1 - c * stat ===")
for stat in ["std", "mad", "iqr"]:
    for c_int in range(1, 2000):
        c = c_int / 1000
        qs = {d: 1 - c * data[d]["stats"][stat] for d in ["MHClip_EN", "MHClip_ZH"]}
        r = check(qs)
        if r:
            print(f"  c={c:.3f} complement {stat}: "
                  f"EN q={r['MHClip_EN'][0]:.4f} {r['MHClip_EN'][2]:.4f}/{r['MHClip_EN'][3]:.4f}  "
                  f"ZH q={r['MHClip_ZH'][0]:.4f} {r['MHClip_ZH'][2]:.4f}/{r['MHClip_ZH'][3]:.4f}")
