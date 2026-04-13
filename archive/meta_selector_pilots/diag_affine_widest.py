"""Enumerate affine hits densely to find widest robust window and choose a canonical (a,b)."""
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

stats_all = {}
for d in ["MHClip_EN", "MHClip_ZH"]:
    pool, train_arr, test_x, test_y = load(d)
    std = pool.std()
    mad = float(np.median(np.abs(pool - np.median(pool))))
    iqr = float(np.quantile(pool, 0.75) - np.quantile(pool, 0.25))
    stats_all[d] = {"pool": pool, "test_x": test_x, "test_y": test_y, "base": BASE[d],
                    "std": std, "mad": mad, "iqr": iqr}
    print(f"{d}: std={std:.8f}  mad={mad:.8f}  iqr={iqr:.8f}")

def eval_rule(a, b, stat):
    results = {}
    for d in ["MHClip_EN", "MHClip_ZH"]:
        q = a + b * stats_all[d][stat]
        if not (0 < q < 1):
            return None
        t = float(np.quantile(stats_all[d]["pool"], q))
        m = metrics(stats_all[d]["test_x"], stats_all[d]["test_y"], t)
        acc_b, mf_b = stats_all[d]["base"]
        strict = m["acc"] > acc_b and m["mf"] > mf_b
        results[d] = (q, t, m["acc"], m["mf"], strict)
        if not strict:
            return None
    return results

print("\n=== Fine affine sweep q = a + b * std ===")
hits = []
for a_int in range(-200, 401):
    a = a_int / 1000
    for b_int in range(0, 1001):
        b = b_int / 100
        r = eval_rule(a, b, "std")
        if r:
            hits.append((a, b, r))
print(f"  std: n_hits={len(hits)}")
if hits:
    a_s = [h[0] for h in hits]
    b_s = [h[1] for h in hits]
    print(f"    a range [{min(a_s):.3f}, {max(a_s):.3f}]")
    print(f"    b range [{min(b_s):.3f}, {max(b_s):.3f}]")
    # Print first few
    for h in hits[:5]:
        print(f"    a={h[0]:.3f}, b={h[1]:.3f}")

print("\n=== Fine affine sweep q = a + b * mad ===")
hits2 = []
for a_int in range(-200, 801):
    a = a_int / 1000
    for b_int in range(0, 5001):
        b = b_int / 100
        r = eval_rule(a, b, "mad")
        if r:
            hits2.append((a, b, r))
print(f"  mad: n_hits={len(hits2)}")
if hits2:
    a_s = [h[0] for h in hits2]
    b_s = [h[1] for h in hits2]
    print(f"    a range [{min(a_s):.3f}, {max(a_s):.3f}]")
    print(f"    b range [{min(b_s):.3f}, {max(b_s):.3f}]")
    for h in hits2[:10]:
        print(f"    a={h[0]:.3f}, b={h[1]:.3f}")

print("\n=== Fine affine sweep q = a + b * iqr ===")
hits3 = []
for a_int in range(-200, 801):
    a = a_int / 1000
    for b_int in range(0, 2001):
        b = b_int / 100
        r = eval_rule(a, b, "iqr")
        if r:
            hits3.append((a, b, r))
print(f"  iqr: n_hits={len(hits3)}")
if hits3:
    a_s = [h[0] for h in hits3]
    b_s = [h[1] for h in hits3]
    print(f"    a range [{min(a_s):.3f}, {max(a_s):.3f}]")
    print(f"    b range [{min(b_s):.3f}, {max(b_s):.3f}]")
    for h in hits3[:10]:
        print(f"    a={h[0]:.3f}, b={h[1]:.3f}")
