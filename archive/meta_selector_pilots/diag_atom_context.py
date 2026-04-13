"""For each mixed atom, compute context features that could distinguish oracle-ADD
from oracle-DROP atoms, using TRAIN POOL ONLY (no test labels).

Features:
  - atom value
  - train_count_at_atom (how many train samples at same 4-dec atom)
  - train_count_below_atom (cumulative train mass below)
  - train_density_local (train count within +/- 0.01)
  - train_gap_above (distance to next higher train atom)
  - train_gap_below (distance to next lower train atom)
  - test_count_at_atom (test mass at atom, no labels)
  - test_isolation (for samples at atom, mean distance to 5 nearest train samples)

Then print per atom:
  - oracle label (pos > neg, pos == neg, pos < neg)
  - all features
This tells us if ANY feature separates oracle-ADD from oracle-DROP.
"""
import os, sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import load_scores_file, build_arrays
from data_utils import load_annotations
from collections import Counter


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    train_arr = np.array(list(train.values()), dtype=float)
    test_arr = np.array(list(test.values()), dtype=float)
    return train_arr, test_x, test_y, test_arr


for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y, test_all = load(d)
    print(f"\n=== {d} ===")
    # Atom counts using 4-decimal rounding
    atoms_pos = {}
    atoms_neg = {}
    atoms_all = {}
    for v, y in zip(test_x, test_y):
        a = round(float(v), 4)
        atoms_all.setdefault(a, []).append(v)
        if int(y) == 1:
            atoms_pos[a] = atoms_pos.get(a, 0) + 1
        else:
            atoms_neg[a] = atoms_neg.get(a, 0) + 1
    # Only look at mixed or non-trivial atoms
    mixed_atoms = sorted(set(atoms_pos.keys()) | set(atoms_neg.keys()))

    # Train pool 4-dec atoms
    train_atoms = Counter(round(float(v), 4) for v in train)
    train_sorted = sorted(set(round(float(v), 4) for v in train))

    print(f"  {'atom':>10s} {'p':>3s} {'n':>3s} {'lab':>3s}"
          f" {'trAt':>5s} {'trBelow':>8s} {'trDens':>7s} {'gapAb':>7s} {'gapBe':>7s}")
    rows = []
    for a in mixed_atoms:
        p = atoms_pos.get(a, 0)
        n = atoms_neg.get(a, 0)
        if p == 0 and n == 0:
            continue
        if p > n:
            lab = "+"
        elif p == n:
            lab = "="
        else:
            lab = "-"

        train_at = train_atoms.get(a, 0)
        train_below = int((train < a).sum())
        # local density: train samples within +/- 0.01
        train_local = int(((train >= a-0.01) & (train <= a+0.01)).sum())
        # gap to nearest train atom above/below
        above = [t for t in train_sorted if t > a]
        below = [t for t in train_sorted if t < a]
        gap_ab = (above[0] - a) if above else float('nan')
        gap_be = (a - below[-1]) if below else float('nan')

        rows.append((a, p, n, lab, train_at, train_below, train_local, gap_ab, gap_be))
        print(f"  {a:>10.4f} {p:>3d} {n:>3d} {lab:>3s}"
              f" {train_at:>5d} {train_below:>8d} {train_local:>7d}"
              f" {gap_ab:>7.4f} {gap_be:>7.4f}")

    # Group by label and compute means
    print(f"\n  means by oracle label:")
    for L in ["+", "-"]:
        sub = [r for r in rows if r[3] == L]
        if not sub:
            continue
        atom_m = np.mean([r[0] for r in sub])
        train_at_m = np.mean([r[4] for r in sub])
        train_local_m = np.mean([r[6] for r in sub])
        gap_ab_m = np.nanmean([r[7] for r in sub])
        gap_be_m = np.nanmean([r[8] for r in sub])
        print(f"    {L}  n={len(sub)}  atom_mean={atom_m:.4f}  train_at={train_at_m:.2f}"
              f"  local={train_local_m:.2f}  gap_ab={gap_ab_m:.4f}  gap_be={gap_be_m:.4f}")
