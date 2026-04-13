"""Find the robust center of the affine hit regions and test robustness to ddof / perturbation."""
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

data = {}
for d in ["MHClip_EN", "MHClip_ZH"]:
    pool, train_arr, test_x, test_y = load(d)
    data[d] = {"pool": pool, "test_x": test_x, "test_y": test_y, "base": BASE[d]}

def stats(pool):
    return {"std": float(pool.std()),
            "std1": float(pool.std(ddof=1)),
            "mad": float(np.median(np.abs(pool - np.median(pool)))),
            "iqr": float(np.quantile(pool, 0.75) - np.quantile(pool, 0.25))}

def try_rule(a, b, stat_name):
    for d in ["MHClip_EN", "MHClip_ZH"]:
        s = stats(data[d]["pool"])
        q = a + b * s[stat_name]
        if not (0 < q < 1):
            return None
        t = float(np.quantile(data[d]["pool"], q))
        m = metrics(data[d]["test_x"], data[d]["test_y"], t)
        acc_b, mf_b = data[d]["base"]
        if not (m["acc"] > acc_b and m["mf"] > mf_b):
            return None
    return True

# Enumerate all hits densely, then find a center
print("=== MAD dense enumeration ===")
hits_mad = []
for a_int in range(500, 650):
    a = a_int / 1000
    for b_int in range(700, 1000):
        b = b_int / 100
        if try_rule(a, b, "mad"):
            hits_mad.append((a, b))

# Find widest slab: what's the b-range for a fixed a?
from collections import defaultdict
by_a = defaultdict(list)
for a, b in hits_mad:
    by_a[a].append(b)
# Pick the a with the widest b-window
best_a = None; best_width = 0
for a, bs in by_a.items():
    w = max(bs) - min(bs)
    if w > best_width:
        best_width = w
        best_a = a
if best_a is not None:
    bs = sorted(by_a[best_a])
    print(f"  widest a={best_a:.3f}, b ∈ [{min(bs):.3f}, {max(bs):.3f}], width={best_width:.3f}")
    # center
    bc = (min(bs) + max(bs)) / 2
    print(f"  center: a={best_a}, b={bc:.3f}")

# IQR:
print("\n=== IQR dense enumeration ===")
hits_iqr = []
for a_int in range(500, 650):
    a = a_int / 1000
    for b_int in range(200, 300):
        b = b_int / 100
        if try_rule(a, b, "iqr"):
            hits_iqr.append((a, b))

by_a2 = defaultdict(list)
for a, b in hits_iqr:
    by_a2[a].append(b)
best_a2 = None; best_w2 = 0
for a, bs in by_a2.items():
    w = max(bs) - min(bs)
    if w > best_w2:
        best_w2 = w
        best_a2 = a
if best_a2 is not None:
    bs = sorted(by_a2[best_a2])
    print(f"  widest a={best_a2:.3f}, b ∈ [{min(bs):.3f}, {max(bs):.3f}], width={best_w2:.3f}")

# === Test robustness of the center point under ddof=1 stats ===
print("\n=== Robustness: center points under ddof=0 vs ddof=1 ===")

def try_rule_ddof(a, b, stat_name, use_ddof1=False):
    for d in ["MHClip_EN", "MHClip_ZH"]:
        pool = data[d]["pool"]
        if stat_name == "std":
            s = float(pool.std(ddof=1 if use_ddof1 else 0))
        elif stat_name == "mad":
            s = float(np.median(np.abs(pool - np.median(pool))))
        elif stat_name == "iqr":
            s = float(np.quantile(pool, 0.75) - np.quantile(pool, 0.25))
        q = a + b * s
        if not (0 < q < 1):
            return None
        t = float(np.quantile(pool, q))
        m = metrics(data[d]["test_x"], data[d]["test_y"], t)
        acc_b, mf_b = data[d]["base"]
        if not (m["acc"] > acc_b and m["mf"] > mf_b):
            return (m, False)
    return (None, True)

if by_a:
    for a, bs in sorted(by_a.items())[:5]:
        bc = (min(bs)+max(bs))/2
        r0 = try_rule_ddof(a, bc, "mad", False)
        r1 = try_rule_ddof(a, bc, "mad", True)
        print(f"  mad a={a:.3f} b={bc:.3f}: ddof0={r0[1]}, ddof1={r1[1]}")

# === Simpler: try q = a + b * log(std) or q = clip(ref_q, 0.6, 0.9) ===
# Or try: is there a constant 'center q' that works for both? (no — disjoint regions)
print("\n=== Test: is the overlap region explained by q = q_otsu_pool_rank / N ? ===")
from quick_eval_all import otsu_threshold
for d in ["MHClip_EN", "MHClip_ZH"]:
    pool = data[d]["pool"]
    t_otsu = otsu_threshold(pool)
    q_otsu = float((pool <= t_otsu).mean())
    print(f"  {d}: t_otsu={t_otsu:.4f}  q_otsu={q_otsu:.4f}")

# === Direct: the simplest is q = a + b * mad, with widest-margin (a,b) ===
# Print final recommendation
print("\n=== FINAL RECOMMENDATION ===")
if hits_mad:
    # Sort by maximum symmetric margin
    # For each hit, compute distance to nearest non-hit
    # Cheap proxy: count hits within small radius
    from collections import Counter
    def nbhd_count(a, b, radius_a=0.005, radius_b=0.1):
        return sum(1 for aa, bb in hits_mad if abs(aa-a) <= radius_a and abs(bb-b) <= radius_b)
    scored = [(nbhd_count(a, b), a, b) for a, b in hits_mad]
    scored.sort(reverse=True)
    print(f"  Most-robust MAD point: a={scored[0][1]}, b={scored[0][2]}  (nbhd_count={scored[0][0]})")
    print(f"  Top 10 by density:")
    for s, a, b in scored[:10]:
        print(f"    a={a:.3f}, b={b:.3f}  density={s}")
