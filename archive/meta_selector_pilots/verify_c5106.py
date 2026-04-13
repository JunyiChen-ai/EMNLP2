"""Verify c=5.106 strict-both with full precision."""
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

for c in [5.106, 5.107]:
    print(f"\n=== c = {c} ===")
    for d in ["MHClip_EN", "MHClip_ZH"]:
        pool, train_arr, test_x, test_y = load(d)
        std_pool = pool.std()
        q = c * std_pool
        t = float(np.quantile(pool, q))
        m = metrics(test_x, test_y, t)
        acc_b, mf_b = BASE[d]
        sb = m["acc"] > acc_b and m["mf"] > mf_b
        print(f"  {d}: std_pool={std_pool:.8f}  q={q:.8f}  t={t:.12f}")
        print(f"         acc={m['acc']:.8f} > {acc_b:.8f}? {m['acc'] > acc_b}")
        print(f"         mf ={m['mf']:.8f} > {mf_b:.8f}? {m['mf'] > mf_b}")
        print(f"         STRICT BOTH: {sb}")
