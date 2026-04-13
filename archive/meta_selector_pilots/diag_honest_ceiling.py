"""Honest feasibility check: under no-FP-phantom assumptions, what's the best
achievable (acc, mf) on each dataset?

Method: round scores to 1e-6 precision (coarser than any plausible FP drift,
finer than any meaningful model-output distinction). Then:
  (A) Suffix ceiling: enumerate all whole-rounded-atom cuts, find best acc, best mf.
  (B) Subset ceiling: for each rounded atom, compute its test pos/neg count. The
      oracle subset rule includes every atom where pos > neg (or pos >= neg if
      broken by tie). Report acc, mf of this oracle.
  (C) Baseline: reproduce EN TF-Otsu, ZH TF-GMM numbers.

This tells us the hard upper bound on any principled label-free rule, assuming
the rule does NOT exploit sub-1e-6 FP jitter. Both baselines must be STRICTLY
EXCEEDED for Gate 2.
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


def eval_subset(atom_flags, atom_of, test_x, test_y):
    # atom_flags: dict {atom -> 0/1}
    pred = np.array([atom_flags[atom_of(v)] for v in test_x], dtype=int)
    return metrics(pred.astype(float), test_y, 0.5)


def per_atom_counts(test_x, test_y, round_to):
    atoms = {}
    for v, y in zip(test_x, test_y):
        a = round(float(v), round_to)
        if a not in atoms:
            atoms[a] = [0, 0]
        atoms[a][int(y)] += 1
    return atoms


for d in ["MHClip_EN", "MHClip_ZH"]:
    pool, test_x, test_y = load(d)
    acc_b, mf_b = BASE[d]
    print(f"\n=== {d} === baseline {acc_b:.6f}/{mf_b:.6f}  (N_test={len(test_x)})")

    for round_to in [6, 5, 4, 3]:
        atoms = per_atom_counts(test_x, test_y, round_to)
        atom_list = sorted(atoms.keys())
        # (A) Suffix ceiling: enumerate atom boundaries
        best_suf_acc = -1
        best_suf_mf = -1
        best_suf_both = -1
        for split in range(len(atom_list) + 1):
            # flag atoms [split:] as positive, [:split] as negative
            flags = {a: 0 for a in atom_list[:split]}
            flags.update({a: 1 for a in atom_list[split:]})
            def atom_of(v, rt=round_to):
                return round(float(v), rt)
            m = eval_subset(flags, atom_of, test_x, test_y)
            if m["acc"] > best_suf_acc:
                best_suf_acc = m["acc"]
            if m["mf"] > best_suf_mf:
                best_suf_mf = m["mf"]
            if m["acc"] > acc_b and m["mf"] > mf_b:
                best_suf_both = max(best_suf_both, (m["acc"] + m["mf"]))

        # (B) Oracle subset ceiling: pos>neg
        flags_sub = {a: 1 if c[1] > c[0] else 0 for a, c in atoms.items()}
        def atom_of2(v, rt=round_to):
            return round(float(v), rt)
        m_sub = eval_subset(flags_sub, atom_of2, test_x, test_y)
        # Oracle with tie: pos >= neg
        flags_sub_tie = {a: 1 if c[1] >= c[0] and c[1] > 0 else 0 for a, c in atoms.items()}
        m_sub_tie = eval_subset(flags_sub_tie, atom_of2, test_x, test_y)

        n_atoms = len(atoms)
        # Show violations: atoms with mixed pos/neg
        mixed = sum(1 for c in atoms.values() if c[0] > 0 and c[1] > 0)
        print(f"  round_to={round_to}  n_atoms={n_atoms}  mixed={mixed}")
        print(f"    suffix ceiling: best_acc={best_suf_acc:.4f}  best_mf={best_suf_mf:.4f}  any_strict_both={'Y' if best_suf_both>0 else 'N'}")
        print(f"    subset oracle (pos>neg): acc={m_sub['acc']:.4f}  mf={m_sub['mf']:.4f}")
        print(f"    subset oracle (pos>=neg&pos>0): acc={m_sub_tie['acc']:.4f}  mf={m_sub_tie['mf']:.4f}")
