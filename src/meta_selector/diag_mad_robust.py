"""Test robustness of q = 0.6 + 7.83 * MAD rule under bootstrap perturbation."""
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

def apply_rule(pool, a, b):
    mad = float(np.median(np.abs(pool - np.median(pool))))
    q = a + b * mad
    q = max(0.01, min(0.99, q))
    return float(np.quantile(pool, q)), q, mad

datasets = {}
for d in ["MHClip_EN", "MHClip_ZH"]:
    pool, train_arr, test_x, test_y = load(d)
    datasets[d] = (pool, test_x, test_y)

for a, b in [(0.547, 9.265), (0.6, 7.83), (0.596, 7.93)]:
    print(f"\n=== rule q = {a} + {b} * MAD ===")
    # Baseline
    for d in ["MHClip_EN", "MHClip_ZH"]:
        pool, test_x, test_y = datasets[d]
        t, q, mad = apply_rule(pool, a, b)
        m = metrics(test_x, test_y, t)
        acc_b, mf_b = BASE[d]
        print(f"  {d}: q={q:.6f}  t={t:.8f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}  "
              f"STRICT={m['acc']>acc_b and m['mf']>mf_b}")

    # Bootstrap: remove one random item from pool 50 times and recheck
    rng = np.random.default_rng(42)
    for d in ["MHClip_EN", "MHClip_ZH"]:
        pool, test_x, test_y = datasets[d]
        acc_b, mf_b = BASE[d]
        pass_count = 0
        for _ in range(50):
            idx = rng.integers(len(pool))
            pool_p = np.delete(pool, idx)
            t, q, _ = apply_rule(pool_p, a, b)
            m = metrics(test_x, test_y, t)
            if m["acc"] > acc_b and m["mf"] > mf_b:
                pass_count += 1
        print(f"  {d} leave-one-out pool robustness: {pass_count}/50 strict")
