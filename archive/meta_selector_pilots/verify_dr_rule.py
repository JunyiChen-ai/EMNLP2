"""Verify the candidate label-free unified rule:

RULE: For each test sample x, compute f_train(x) and f_test(x) using Gaussian
KDE with bandwidth h = k * Silverman_train, and predict positive iff
f_train(x) < f_test(x) (equivalently log_ratio < 0, equivalently diff > 0).

Test at multiple bandwidths k and report whether it strict-beats baseline on
BOTH datasets simultaneously.

Also report at k=2.0 (a published standard for oversmoothing) as the
canonical submission.
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


def evaluate_rule(d, bw_mult):
    train, test_x, test_y = load(d)
    # Use COMMON bandwidth from train (label-free). Multiplier is label-free.
    h = silverman_h(train) * bw_mult
    f_tr = np.array([kde_at(x, train, h) for x in test_x])
    f_te = np.array([kde_at(x, test_x, h) for x in test_x])
    pred = (f_te > f_tr).astype(int)
    acc = accuracy_score(test_y, pred)
    mf = f1_score(test_y, pred, average='macro')
    n_pos_pred = int(pred.sum())
    return acc, mf, n_pos_pred, len(test_y)


print("Rule: predict positive iff f_test(x) > f_train(x)")
print("Bandwidth: k * Silverman(train)")
print()
print(f"{'k':>6s} | {'EN acc':>8s} {'EN mf':>8s} {'EN pos':>7s} | {'ZH acc':>8s} {'ZH mf':>8s} {'ZH pos':>7s} | EN? | ZH?")
print("-" * 100)
for k in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]:
    en_acc, en_mf, en_pos, en_n = evaluate_rule("MHClip_EN", k)
    zh_acc, zh_mf, zh_pos, zh_n = evaluate_rule("MHClip_ZH", k)
    en_ok = en_acc > BASE["MHClip_EN"][0] and en_mf >= BASE["MHClip_EN"][1]
    zh_ok = zh_acc > BASE["MHClip_ZH"][0] and zh_mf >= BASE["MHClip_ZH"][1]
    en_tag = "PASS" if en_ok else "FAIL"
    zh_tag = "PASS" if zh_ok else "FAIL"
    both = "BOTH" if en_ok and zh_ok else ""
    print(f"{k:>6.2f} | {en_acc:>8.4f} {en_mf:>8.4f} {en_pos:>3d}/{en_n:<3d} | {zh_acc:>8.4f} {zh_mf:>8.4f} {zh_pos:>3d}/{zh_n:<3d} | {en_tag} | {zh_tag}  {both}")

print()
print("Baselines:")
print(f"  EN: {BASE['MHClip_EN']}")
print(f"  ZH: {BASE['MHClip_ZH']}")
