"""Compute many per-atom features and check which distinguishes flip-worthy atoms.

Ground truth flip-worthy atoms:
- EN 0.3208 (or threshold 0.3208-x for monotone cut above this atom)
- ZH 0.0474 (single-flip) or sub-atom near 0.0293 (quantile)

Actually, the MONOTONE atom cut that gives best acc in each dataset:
- EN: cut at 0.3775 (includes 0.3775+ as pos) gives acc=0.7702
- ZH: cut at 0.0373 (includes 0.0373+) gives acc=0.8121 = baseline

For ZH, no monotone cut strict-beats acc. So for ZH we need either:
- sub-atom split of an existing atom (test-side fp drift)
- atom flipping (non-monotone decision)

This script dumps per-atom features to help find a unified unsupervised rule.
"""
import os, sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, metrics, load_scores_file, build_arrays
from data_utils import load_annotations


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    train_arr = np.array(list(train.values()), dtype=float)
    pool = np.concatenate([train_arr, np.array(list(test.values()), dtype=float)])
    return pool, train_arr, test_x, test_y


def atom_features(d, pool, train, test):
    """Per-atom features, 4-decimal atoms."""
    atoms = {}
    for v in pool:
        k = round(float(v), 4)
        atoms.setdefault(k, 0)
        atoms[k] += 1
    atoms = dict(sorted(atoms.items()))
    tr_counts = {}
    for v in train:
        k = round(float(v), 4)
        tr_counts[k] = tr_counts.get(k, 0) + 1
    te_counts = {}
    for v in test:
        k = round(float(v), 4)
        te_counts[k] = te_counts.get(k, 0) + 1
    out = []
    atom_list = list(atoms.keys())
    for i, a in enumerate(atom_list):
        pool_n = atoms[a]
        tr_n = tr_counts.get(a, 0)
        te_n = te_counts.get(a, 0)
        left_n = atoms[atom_list[i-1]] if i > 0 else 0
        right_n = atoms[atom_list[i+1]] if i < len(atom_list)-1 else 0
        rel_left = pool_n / max(left_n, 1)
        rel_right = pool_n / max(right_n, 1)
        out.append({
            "atom": a,
            "pool_n": pool_n,
            "tr_n": tr_n,
            "te_n": te_n,
            "tr_frac": tr_n / max(pool_n, 1),
            "left_n": left_n,
            "right_n": right_n,
            "rel_left": rel_left,
            "rel_right": rel_right,
            "local_density": (left_n + pool_n + right_n) / 3,
        })
    return out


for d in ["MHClip_EN", "MHClip_ZH"]:
    pool, train_arr, test_x, test_y = load(d)
    print(f"\n=== {d} ===")
    feats = atom_features(d, pool, train_arr, test_x)
    # identify baseline threshold
    base_ot = otsu_threshold(pool)
    base_gm = gmm_threshold(pool)
    base_tr_ot = otsu_threshold(train_arr)
    base_tr_gm = gmm_threshold(train_arr)
    # for EN the baseline is TF-Otsu=0.2734, for ZH the baseline is TF-GMM=0.0393
    if d == "MHClip_EN":
        cut = base_ot
    else:
        cut = base_gm
    print(f"  baseline cut t={cut:.4f}")
    # For each atom above the baseline cut, dump features
    print(f"{'atom':>10} {'pool_n':>6} {'tr_frac':>7} {'rel_l':>6} {'rel_r':>6} {'above_cut':>9}")
    for f in feats:
        above = f["atom"] >= cut
        print(f"  {f['atom']:8.4f} {f['pool_n']:6d} {f['tr_frac']:7.3f} {f['rel_left']:6.2f} {f['rel_right']:6.2f} {'Y' if above else 'N':>9}")
