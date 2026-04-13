"""Inspect the atom-level labeling produced by the DR passing rules to
understand the mechanism. Are they suffix or non-suffix at atom level?
And what cut values would be needed for label-free selection?
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import load_scores_file, build_arrays
from data_utils import load_annotations
from sklearn.metrics import accuracy_score, f1_score

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    return np.array(list(train.values()), dtype=float), test_x, test_y


def kde_at(x, pool, h):
    return np.sum(np.exp(-0.5 * ((pool - x) / h) ** 2)) / (len(pool) * h * np.sqrt(2*np.pi))


def silverman_h(pool):
    return float(pool.std()) * (4 / (3 * len(pool))) ** (1 / 5)


for d, target_rule in [
    ("MHClip_EN", ("h2.0_log_ratio", "<", 0.0018)),
    ("MHClip_ZH", ("h2.0_log_ratio", "<", 0.0806)),
]:
    train, test_x, test_y = load(d)
    print(f"\n=== {d} === target rule: {target_rule}")
    h = silverman_h(train) * 2.0
    f_tr = np.array([kde_at(x, train, h) for x in test_x])
    f_te = np.array([kde_at(x, test_x, h) for x in test_x])
    log_r = np.log((f_tr + 1e-12) / (f_te + 1e-12))

    rounded = np.round(test_x, 4)
    atoms_vals = sorted(set(rounded))

    print(f"  {'idx':>3s} {'val':>8s} {'log_r':>10s} {'pos':>4s} {'neg':>4s} {'rule':>6s}")
    for idx, aval in enumerate(atoms_vals):
        mask = rounded == aval
        lr_val = log_r[mask].mean()  # same for all samples at same score
        pc = int(test_y[mask].sum())
        nc = int((1 - test_y[mask]).sum())
        if target_rule[1] == "<":
            in_pred = lr_val < target_rule[2]
        else:
            in_pred = lr_val > target_rule[2]
        tag = "POS" if in_pred else "NEG"
        print(f"  {idx:>3d} {aval:>8.4f} {lr_val:>10.4f} {pc:>4d} {nc:>4d} {tag:>6s}")

    # Apply rule and report metrics
    pred = (log_r < target_rule[2]).astype(int) if target_rule[1] == "<" else (log_r > target_rule[2]).astype(int)
    acc = accuracy_score(test_y, pred)
    mf = f1_score(test_y, pred, average='macro')
    print(f"  Rule result: acc={acc:.4f} mf={mf:.4f}")
    acc_b, mf_b = BASE[d]
    print(f"  Baseline:    acc={acc_b:.4f} mf={mf_b:.4f}")
    print(f"  strict-beat: {acc > acc_b and mf >= mf_b}")
