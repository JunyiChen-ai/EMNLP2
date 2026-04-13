"""Exhaustive threshold search: what's the best whole-atom suffix rule per dataset?

If the best possible strict-beat per dataset exists, find its threshold value.
Then check if any pool-statistic function can produce that threshold simultaneously
on both datasets.
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
    pool = np.concatenate([train_arr, test_x])
    return pool, test_x, test_y

print("=== Exhaustive threshold search (find all thresholds that strict-beat baseline) ===")
for d in ["MHClip_EN", "MHClip_ZH"]:
    pool, test_x, test_y = load(d)
    acc_b, mf_b = BASE[d]
    # For each threshold in sorted unique values plus gap midpoints
    unique = sorted(set(list(pool) + list(test_x)))
    candidate_thresholds = list(unique)
    # Add midpoints between consecutive values for potential sub-atom cuts
    for i in range(len(unique)-1):
        candidate_thresholds.append((unique[i] + unique[i+1])/2)
    candidate_thresholds = sorted(set(candidate_thresholds))

    hits = []
    for t in candidate_thresholds:
        m = metrics(test_x, test_y, t)
        if m["acc"] > acc_b and m["mf"] > mf_b:
            hits.append((t, m["acc"], m["mf"]))
    print(f"\n{d}: {len(hits)} strict-beat thresholds")
    if hits:
        print(f"  range: [{hits[0][0]:.6f}, {hits[-1][0]:.6f}]")
        # Print quantile position in pool
        for t, acc, mf in hits[:10]:
            q = float((pool <= t).mean())
            print(f"    t={t:.6f}  q={q:.4f}  acc={acc:.4f}  mf={mf:.4f}")
        if len(hits) > 10:
            print(f"    ... {len(hits)-10} more ...")
            for t, acc, mf in hits[-5:]:
                q = float((pool <= t).mean())
                print(f"    t={t:.6f}  q={q:.4f}  acc={acc:.4f}  mf={mf:.4f}")
