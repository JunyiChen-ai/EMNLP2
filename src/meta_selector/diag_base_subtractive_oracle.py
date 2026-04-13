"""Oracle: what's the ceiling if we can only SUBTRACT atoms from base=POS?
(I.e., decide for each base-POS atom independently whether to keep or drop it,
and never add a base=NEG atom to the POS set.)

This gives an upper bound on all subtractive rules.
"""
import sys, numpy as np, itertools
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, load_scores_file, build_arrays
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


for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    acc_b, mf_b = BASE[d]
    n_te = len(test_x)
    print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}")

    t_base = otsu_threshold(test_x) if d == "MHClip_EN" else gmm_threshold(test_x)
    rounded_te = np.round(test_x, 4)
    atoms_vals = np.array(sorted(set(rounded_te)))
    K = len(atoms_vals)
    base_atom = (atoms_vals >= t_base)
    base_idx = np.where(base_atom)[0]
    n_base = len(base_idx)
    print(f"  |base_pos|={n_base}, enumerating {2**n_base} subsets")

    # Per-atom pos/neg counts
    pos_per_atom = np.array([
        int(((rounded_te == v) & (test_y == 1)).sum()) for v in atoms_vals
    ])
    neg_per_atom = np.array([
        int(((rounded_te == v) & (test_y == 0)).sum()) for v in atoms_vals
    ])
    total_pos = int(test_y.sum())
    print(f"  total_pos={total_pos}, base pos/neg breakdown:")
    for i in base_idx:
        print(f"    v={atoms_vals[i]:.4f} pos={pos_per_atom[i]} neg={neg_per_atom[i]}")

    # Enumerate all 2^n_base subsets of base-POS atoms to keep
    best = (0, 0, None)
    passes = []
    for mask in range(2 ** n_base):
        keep = np.zeros(K, dtype=int)
        for bi, i in enumerate(base_idx):
            if (mask >> bi) & 1:
                keep[i] = 1
        # Predict on test
        tp = int((keep * pos_per_atom).sum())
        fp = int((keep * neg_per_atom).sum())
        fn = total_pos - tp
        tn = n_te - total_pos - fp
        acc = (tp + tn) / n_te
        pp = tp + fp
        pn = tn + fn
        if pp > 0 and pn > 0:
            prec_p = tp / pp if pp > 0 else 0
            rec_p = tp / total_pos if total_pos > 0 else 0
            f1_p = 2 * prec_p * rec_p / (prec_p + rec_p) if prec_p + rec_p > 0 else 0
            prec_n = tn / pn if pn > 0 else 0
            rec_n = tn / (n_te - total_pos) if (n_te - total_pos) > 0 else 0
            f1_n = 2 * prec_n * rec_n / (prec_n + rec_n) if prec_n + rec_n > 0 else 0
            mf = (f1_p + f1_n) / 2
        else:
            mf = 0
        strict = acc > acc_b and mf > mf_b
        if strict:
            passes.append((mask, acc, mf))
        if (acc, mf) > best[:2]:
            best = (acc, mf, mask)

    print(f"  Subtractive-only oracle ceiling: acc={best[0]:.4f} mf={best[1]:.4f}")
    print(f"  # strict-both passing subsets: {len(passes)}")
    if passes:
        # Show subset with highest mf
        passes.sort(key=lambda x: (x[2], x[1]), reverse=True)
        for mask, acc, mf in passes[:5]:
            kept = []
            for bi, i in enumerate(base_idx):
                if (mask >> bi) & 1:
                    kept.append(f"{atoms_vals[i]:.4f}")
            print(f"    mask={mask:0{n_base}b} acc={acc:.4f} mf={mf:.4f} keep={kept}")
