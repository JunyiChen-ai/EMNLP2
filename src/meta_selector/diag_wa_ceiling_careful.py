"""Very careful whole-atom ceiling search for ZH. Use the actual raw test
values (no rounding, no synthesis) and enumerate every possible threshold that
falls at a gap between distinct raw test values.

Each such cut produces n_pred_pos in {0, 1, ..., N}. For each cut, compute acc
and mf. Find the max strictly beating baseline, and verify the cut lands in a
genuine inter-sample gap (not inside an FP cluster).
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
from data_utils import load_annotations


BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}

FP_CLUSTER_GAP = 1e-6  # gaps smaller than this are considered sub-cluster


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    return np.array(list(train.values()), dtype=float), test_x, test_y


for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    acc_b, mf_b = BASE[d]
    # Sort test values
    sorted_test = sorted(test_x)
    # Enumerate gaps between consecutive distinct values
    gaps = []
    for i in range(len(sorted_test) - 1):
        a = sorted_test[i]
        b = sorted_test[i + 1]
        if b - a > FP_CLUSTER_GAP:
            gaps.append((a, b, b - a))
    gaps.append((sorted_test[-1], float('inf'), float('inf')))  # max

    print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}")
    print(f"  {len(gaps)} legitimate inter-sample gaps (> {FP_CLUSTER_GAP})")

    hits = []
    best_acc = (-1, None)
    best_mf = (-1, None)
    for a, b, g in gaps:
        t = (a + b) / 2 if np.isfinite(b) else a + 1.0  # midpoint of the gap
        m = metrics(test_x, test_y, t)
        if m["acc"] > acc_b and m["mf"] > mf_b:
            hits.append((t, m["acc"], m["mf"], a, b, g))
        if m["acc"] > best_acc[0]:
            best_acc = (m["acc"], (t, m))
        if m["mf"] > best_mf[0]:
            best_mf = (m["mf"], (t, m))

    print(f"  gap-based whole-atom strict-beat hits: {len(hits)}")
    if hits:
        for h in hits[:10]:
            print(f"    t={h[0]:.6f} acc={h[1]:.4f} mf={h[2]:.4f} gap=[{h[3]:.6f},{h[4]:.6f}] g={h[5]:.2e}")
    print(f"  best_acc={best_acc[0]:.4f}  best_mf={best_mf[0]:.4f}")
