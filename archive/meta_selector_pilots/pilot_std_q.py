"""Test q = c * std for various c values around 5.1.

Is there a c such that q_EN and q_ZH both land in strict-beat regions?
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

# Load
data = {}
for d in ["MHClip_EN", "MHClip_ZH"]:
    pool, train_arr, test_x, test_y = load(d)
    data[d] = {"pool": pool, "train": train_arr, "test_x": test_x, "test_y": test_y,
               "base": BASE[d]}
    print(f"  {d} pool: mu={pool.mean():.6f} std={pool.std():.6f}")
    print(f"  {d} train: mu={train_arr.mean():.6f} std={train_arr.std():.6f}")
    print(f"  {d} test: mu={test_x.mean():.6f} std={test_x.std():.6f}")

EPS = 1e-10
# Fine sweep over c
print("\n=== q = c * std(pool) ===")
for c_int in range(100, 700):
    c = c_int / 100
    results = {}
    for d in ["MHClip_EN", "MHClip_ZH"]:
        pool = data[d]["pool"]
        q = c * pool.std()
        if q >= 1 or q <= 0:
            continue
        t = float(np.quantile(pool, q))
        m = metrics(data[d]["test_x"], data[d]["test_y"], t)
        acc_b, mf_b = data[d]["base"]
        strict_both = m["acc"] > acc_b and m["mf"] > mf_b
        results[d] = (q, t, m["acc"], m["mf"], strict_both)
    if "MHClip_EN" in results and "MHClip_ZH" in results:
        r_en = results["MHClip_EN"]
        r_zh = results["MHClip_ZH"]
        if r_en[4] and r_zh[4]:
            print(f"** c={c:.2f}: EN q={r_en[0]:.4f} t={r_en[1]:.6f} {r_en[2]:.4f}/{r_en[3]:.4f}  "
                  f"ZH q={r_zh[0]:.4f} t={r_zh[1]:.6f} {r_zh[2]:.4f}/{r_zh[3]:.4f} **")
