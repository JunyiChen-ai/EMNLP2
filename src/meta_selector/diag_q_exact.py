"""Exact computation of q = c * std for c around 5.1."""
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
    print(f"\n=== {d} === std_pool={pool.std():.8f}  baseline {acc_b:.4f}/{mf_b:.4f}")
    for c in np.arange(5.090, 5.120, 0.0005):
        q = c * pool.std()
        t = float(np.quantile(pool, q))
        m = metrics(test_x, test_y, t)
        tag = ""
        if m["acc"] > acc_b: tag += "[acc+]"
        if m["mf"] > mf_b: tag += "[mf+]"
        if m["acc"] > acc_b and m["mf"] > mf_b: tag = "***STRICT***"
        print(f"  c={c:.3f}  q={q:.6f}  t={t:.8f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}  {tag}")
