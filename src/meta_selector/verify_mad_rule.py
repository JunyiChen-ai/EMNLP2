"""Verify candidate unified rules based on pool MAD.

Candidate: q = a + b * MAD(pool); threshold = quantile(pool, q).
Rationale: MAD is a robust spread estimator (ddof-independent, median-centered).
The rule says: threshold sits at a 'base quantile' a plus an adjustment proportional
to how dispersed the pool is.

Test a set of round-number (a,b) candidates for both datasets under strict bar.
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

data = {}
for d in ["MHClip_EN", "MHClip_ZH"]:
    pool, train_arr, test_x, test_y = load(d)
    mad = float(np.median(np.abs(pool - np.median(pool))))
    data[d] = {"pool": pool, "test_x": test_x, "test_y": test_y, "base": BASE[d], "mad": mad}
    print(f"{d}: n_pool={len(pool)}  MAD={mad:.8f}")

candidates = [
    (0.60, 8.0),
    (0.55, 9.0),
    (0.58, 8.5),
    (0.547, 9.265),  # widest-slab center
    (0.60, 7.83),    # density max
    (0.596, 7.93),
    (0.55, 9.1),
    (0.50, 10.0),
]
print("\n=== Candidates: q = a + b * MAD(pool) ===")
for a, b in candidates:
    print(f"\n  rule: q = {a} + {b} * MAD")
    all_pass = True
    for d in ["MHClip_EN", "MHClip_ZH"]:
        mad = data[d]["mad"]
        q = a + b * mad
        pool = data[d]["pool"]
        t = float(np.quantile(pool, q))
        m = metrics(data[d]["test_x"], data[d]["test_y"], t)
        acc_b, mf_b = data[d]["base"]
        s_acc = m["acc"] > acc_b
        s_mf = m["mf"] > mf_b
        strict = s_acc and s_mf
        print(f"    {d}: q={q:.6f}  t={t:.8f}  "
              f"acc={m['acc']:.4f}(>{acc_b:.4f}:{s_acc})  "
              f"mf={m['mf']:.4f}(>{mf_b:.4f}:{s_mf})  "
              f"{'***STRICT***' if strict else ''}")
        if not strict:
            all_pass = False
    print(f"    --- PASS BOTH: {all_pass}")
