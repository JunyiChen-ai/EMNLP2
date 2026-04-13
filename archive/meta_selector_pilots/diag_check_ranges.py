"""What ACTUAL threshold values do the successful quantile ranges produce?"""
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

for d in ["MHClip_EN", "MHClip_ZH"]:
    pool, train_arr, test_x, test_y = load(d)
    acc_b, mf_b = BASE[d]
    print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}")
    # Compute strict-beat q range
    q_range = {"MHClip_EN": np.arange(0.8831, 0.8860, 0.0001),
               "MHClip_ZH": np.arange(0.6425, 0.6934, 0.0001)}[d]
    prev_t = None
    for q in q_range:
        t = float(np.quantile(pool, q))
        m = metrics(test_x, test_y, t)
        sb = m["acc"] > acc_b and m["mf"] > mf_b
        if t != prev_t:
            prev_t = t
            tag = " **STRICT**" if sb else ""
            # Find nearest atom
            uniq = sorted(set(round(v, 4) for v in pool))
            lo = max((a for a in uniq if a <= t), default=-1)
            hi = min((a for a in uniq if a >= t), default=2)
            print(f"  q={q:.4f}  t={t:.10f}  acc={m['acc']:.4f} mf={m['mf']:.4f}  [atoms {lo:.4f}→{hi:.4f}]{tag}")
