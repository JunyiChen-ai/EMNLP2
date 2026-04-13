"""Enumerate ALL whole-atom-boundary monotone thresholds.

If ZH has NO monotone threshold that strict-beats baseline, then ANY unified
threshold-based rule must exploit sub-atom ordering OR do non-monotone flipping.
"""
import os, sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
from data_utils import load_annotations

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}


def load(dataset):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{dataset}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(dataset)
    test_x, test_y = build_arrays(test, ann)
    train_arr = np.array(list(train.values()), dtype=float)
    pool = np.concatenate([train_arr, np.array(list(test.values()), dtype=float)])
    return pool, train_arr, test_x, test_y


for d in ["MHClip_EN", "MHClip_ZH"]:
    pool, train_arr, test_x, test_y = load(d)
    acc_b, mf_b = BASE[d]
    # 4-decimal atoms
    atoms = sorted(set(round(v, 4) for v in pool))
    print(f"\n=== {d} === baseline {acc_b:.6f}/{mf_b:.6f}")
    # Cut just below atom A means predict positive iff x >= A
    best_acc = -1
    best_mf = -1
    strict_cuts = []
    for A in atoms:
        # threshold slightly below A (e.g., midpoint with previous atom)
        t = A - 1e-6
        m = metrics(test_x, test_y, t)
        strict = m["acc"] > acc_b and m["mf"] > mf_b
        if strict:
            strict_cuts.append((A, m["acc"], m["mf"]))
        if m["acc"] > best_acc:
            best_acc = m["acc"]
        if m["mf"] > best_mf:
            best_mf = m["mf"]
    print(f"  monotone best_acc={best_acc:.4f}  best_mf={best_mf:.4f}")
    print(f"  strict cuts: {len(strict_cuts)}")
    for c in strict_cuts:
        print(f"    cut at atom {c[0]:.4f}: acc={c[1]:.4f} mf={c[2]:.4f}")
