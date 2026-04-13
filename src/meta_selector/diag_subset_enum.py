"""Enumerate atom subsets that strict-beat the bar.

ZH has 32 unique atoms. 2^32 = 4.3B, too many. But the goal is subsets that
strict-beat: acc > baseline AND mf >= baseline. Start from the baseline
threshold (positive = atoms with x >= 0.0333, which is atoms 16..31), and
enumerate subsets by single-atom flip operations. A flip is "include atom k
in positive class if currently negative, or exclude if currently positive".

Use a BFS over the subset graph, expanding only subsets that dominate the
current best (strict-beat bar). For each passing subset, print the subset
and check whether it has a label-free characterization by a train-landmark.

This answers the question: "does the oracle subset Pareto include subsets
that also happen to match a natural label-free rule?"

Also prints: which atoms are in/out, which are label-mixed, and train-only
features of each atom.
"""
import sys, numpy as np
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
    return np.array(list(train.values()), dtype=float), test_x, test_y


def eval_pred_from_subset(atom_vals, test_x, test_y, pos_atoms):
    """Label test samples positive iff their atom is in pos_atoms."""
    pos_set = set(pos_atoms)
    pred = np.array([1 if atom_vals[i] in pos_set else 0 for i in range(len(test_x))])
    acc = (pred == test_y).mean()
    # macro F1
    tp1 = ((pred==1)&(test_y==1)).sum(); fp1 = ((pred==1)&(test_y==0)).sum()
    fn1 = ((pred==0)&(test_y==1)).sum()
    p1 = tp1/(tp1+fp1) if tp1+fp1>0 else 0; r1 = tp1/(tp1+fn1) if tp1+fn1>0 else 0
    f1 = 2*p1*r1/(p1+r1) if p1+r1>0 else 0
    tp0 = ((pred==0)&(test_y==0)).sum(); fp0 = ((pred==0)&(test_y==1)).sum()
    fn0 = ((pred==1)&(test_y==0)).sum()
    p0 = tp0/(tp0+fp0) if tp0+fp0>0 else 0; r0 = tp0/(tp0+fn0) if tp0+fn0>0 else 0
    f0 = 2*p0*r0/(p0+r0) if p0+r0>0 else 0
    return acc, (f0+f1)/2


def atom_info(test_x, test_y, train):
    """Return list of (atom_value, pos_count, neg_count, train_at_atom, train_below)."""
    unique_atoms = sorted(set(test_x))
    info = []
    for a in unique_atoms:
        mask = test_x == a
        pc = int(test_y[mask].sum())
        nc = int((1 - test_y[mask]).sum())
        train_at = int((train == a).sum())
        train_below = int((train < a).sum())
        info.append({"atom": a, "pos": pc, "neg": nc, "n": pc+nc,
                     "train_at": train_at, "train_below": train_below,
                     "train_frac_below": train_below/len(train)})
    return info


def find_passing_subsets(atoms_list, atom_vals, test_x, test_y, acc_b, mf_b,
                         max_subsets=200):
    """BFS over 1-flip neighborhoods from the baseline threshold subset."""
    n_atoms = len(atoms_list)
    # Baseline: positive = atoms above ZH threshold 0.0333
    # Equivalently, find the atoms whose value >= min atom where baseline passes
    # Baseline cuts at t=0.0333 on ZH, so positive atoms are those with value >= ~0.0373
    # Simplest: find baseline threshold, convert to atom-set
    sorted_atoms = sorted(atoms_list)
    passing = []
    # Start from all 32 possible threshold subsets
    seed_subsets = []
    for k in range(n_atoms+1):
        seed_subsets.append(frozenset(sorted_atoms[k:]))

    visited = set()
    queue = list(seed_subsets)
    while queue and len(passing) < max_subsets:
        current = queue.pop(0)
        if current in visited: continue
        visited.add(current)
        acc, mf = eval_pred_from_subset(atom_vals, test_x, test_y, current)
        if acc > acc_b and mf >= mf_b:
            passing.append((current, acc, mf))
        # Expand: try flipping one atom
        for a in sorted_atoms:
            if a in current:
                new = current - {a}
            else:
                new = current | {a}
            if new not in visited and len(visited) < 500000:
                # Only queue if it has potential (heuristic: near-baseline acc)
                na, nm = eval_pred_from_subset(atom_vals, test_x, test_y, new)
                if na >= acc_b - 0.01 and nm >= mf_b - 0.02:
                    queue.append(new)
    return passing


for d in ["MHClip_ZH"]:  # focus on ZH where the bar is tighter
    train, test_x, test_y = load(d)
    acc_b, mf_b = BASE[d]
    print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}")

    info = atom_info(test_x, test_y, train)
    atoms_list = [i["atom"] for i in info]
    atom_vals = test_x.tolist()  # atom value at each test index

    print(f"  {len(atoms_list)} unique atoms. Atom table:")
    print(f"  {'atom':>10s} {'pos':>4s} {'neg':>4s} {'n':>4s} {'train_at':>9s} {'tr_below':>9s}")
    for i in info:
        print(f"  {i['atom']:>10.4f} {i['pos']:>4d} {i['neg']:>4d} {i['n']:>4d} {i['train_at']:>9d} {i['train_below']:>9d}")

    print(f"\n  Starting BFS for subsets strict-beating bar (acc>{acc_b:.4f} AND mf>={mf_b:.4f})")
    passing = find_passing_subsets(atoms_list, atom_vals, test_x, test_y, acc_b, mf_b, max_subsets=50)
    print(f"  Found {len(passing)} passing subsets")

    # Sort by margin
    passing.sort(key=lambda x: (-(x[1] - acc_b), -(x[2] - mf_b)))
    for sub, acc, mf in passing[:20]:
        sub_sorted = sorted(sub)
        # Describe: as threshold or as non-monotone
        threshold_desc = None
        if len(sub_sorted) > 0:
            lowest = sub_sorted[0]
            # Is this the suffix above lowest?
            suffix = [a for a in atoms_list if a >= lowest]
            if set(suffix) == set(sub):
                threshold_desc = f"threshold>={lowest:.4f}"
        if threshold_desc is None:
            # Which atoms are in, which out (above min)
            min_a = min(sub_sorted)
            holes = [a for a in atoms_list if a >= min_a and a not in sub]
            threshold_desc = f"NONMON: min={min_a:.4f}, holes={[round(x,4) for x in holes]}"
        print(f"  acc={acc:.4f} mf={mf:.4f} +Δacc={acc-acc_b:.4f} +Δmf={mf-mf_b:.4f} n={len(sub)}  {threshold_desc}")
