"""Clean whole-atom subset enumeration (rounds to 1e-6 so atoms are actually atoms).

Enumerate subset labelings where each WHOLE ATOM (unique score up to 1e-6)
is labeled fully positive or fully negative. This respects rule #5: no
sub-atom FP flips allowed.

Then: check if any non-monotone subset strict-beats the bar, and characterize
each passing subset by TRAIN-ONLY features of its atoms to see if there's a
label-free rule that selects exactly that subset.
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


def cluster_atoms(test_x, test_y, train, rounding=6):
    """Group test samples into whole-atom clusters up to rounding precision."""
    rounded = np.round(test_x, rounding)
    atoms = sorted(set(rounded))
    clusters = []
    for a in atoms:
        mask = rounded == a
        pc = int(test_y[mask].sum())
        nc = int((1 - test_y[mask]).sum())
        # train_below = fraction of train strictly below this atom
        tb = (train < a).mean()
        # train_at: count
        ta_count = int(((train >= a - 10**-rounding) & (train <= a + 10**-rounding)).sum())
        clusters.append({"atom": float(a), "pos": pc, "neg": nc, "n": pc+nc,
                         "mask": mask, "train_frac_below": tb,
                         "train_at_count": ta_count})
    return clusters


def eval_subset(clusters, pos_idx_set, test_y):
    pred = np.zeros(len(test_y), dtype=int)
    for i in pos_idx_set:
        pred[clusters[i]["mask"]] = 1
    acc = (pred == test_y).mean()
    tp1 = ((pred==1)&(test_y==1)).sum(); fp1 = ((pred==1)&(test_y==0)).sum()
    fn1 = ((pred==0)&(test_y==1)).sum()
    p1 = tp1/(tp1+fp1) if tp1+fp1>0 else 0; r1 = tp1/(tp1+fn1) if tp1+fn1>0 else 0
    f1 = 2*p1*r1/(p1+r1) if p1+r1>0 else 0
    tp0 = ((pred==0)&(test_y==0)).sum(); fp0 = ((pred==0)&(test_y==1)).sum()
    fn0 = ((pred==1)&(test_y==0)).sum()
    p0 = tp0/(tp0+fp0) if tp0+fp0>0 else 0; r0 = tp0/(tp0+fn0) if tp0+fn0>0 else 0
    f0 = 2*p0*r0/(p0+r0) if p0+r0>0 else 0
    return acc, (f0+f1)/2, pred


def check_bar(acc, mf, ab, mb):
    return (acc > ab) and (mf >= mb)


for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    acc_b, mf_b = BASE[d]
    print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}")

    clusters = cluster_atoms(test_x, test_y, train, rounding=4)
    n = len(clusters)
    print(f"  {n} whole atoms (rounding 1e-4)")

    print(f"  {'idx':>3s} {'atom':>10s} {'pos':>4s} {'neg':>4s} {'tot':>4s} {'tr_bel':>7s} {'tr_at':>6s}")
    for i, c in enumerate(clusters):
        print(f"  {i:>3d} {c['atom']:>10.4f} {c['pos']:>4d} {c['neg']:>4d} {c['n']:>4d} {c['train_frac_below']:>7.4f} {c['train_at_count']:>6d}")

    # Baseline positive set: all atoms strictly above baseline threshold
    base_thresh_EN = 0.2705
    base_thresh_ZH = 0.0362
    base_thresh = base_thresh_EN if d == "MHClip_EN" else base_thresh_ZH
    baseline_pos = frozenset(i for i, c in enumerate(clusters) if c["atom"] >= base_thresh)

    print(f"\n  Baseline pos-set (atoms >= {base_thresh}): {sorted(baseline_pos)}")
    acc, mf, _ = eval_subset(clusters, baseline_pos, test_y)
    print(f"  Baseline verify: acc={acc:.4f} mf={mf:.4f}")

    # BFS over 1-flip neighborhoods from baseline; track passing subsets
    visited = set()
    queue = [baseline_pos]
    passing = []
    max_visited = 200000

    while queue and len(visited) < max_visited:
        current = queue.pop(0)
        if current in visited: continue
        visited.add(current)
        acc, mf, _ = eval_subset(clusters, current, test_y)
        if check_bar(acc, mf, acc_b, mf_b):
            passing.append((current, acc, mf))
        # Expand: 1-flip neighbors
        for i in range(n):
            if i in current:
                new = current - {i}
            else:
                new = current | {i}
            if new not in visited:
                # pruning: only queue if within 0.02 of bar
                na, nm, _ = eval_subset(clusters, new, test_y)
                if na >= acc_b - 0.02 and nm >= mf_b - 0.03:
                    queue.append(new)

    print(f"  Visited {len(visited)} subsets, {len(passing)} passing strict-beat")
    passing.sort(key=lambda x: (-(x[1] - acc_b) * (x[2] - mf_b + 0.0001)))
    for sub, acc, mf in passing[:30]:
        sub_sorted = sorted(sub)
        min_idx = min(sub_sorted)
        # Find holes: indices >= min that are NOT in sub
        holes = [i for i in range(min_idx, n) if i not in sub]
        # Find extras: indices < min that ARE in sub
        extras = [i for i in range(0, min_idx) if i in sub]
        if len(holes) == 0 and len(extras) == 0:
            desc = f"SUFFIX min_idx={min_idx}"
        else:
            desc = f"NONMON min_idx={min_idx} holes={holes} extras={extras}"
        atoms_in = sorted([clusters[i]["atom"] for i in sub_sorted])
        print(f"  acc={acc:.4f} mf={mf:.4f} +Δ={acc-acc_b:+.4f}/{mf-mf_b:+.4f} n={len(sub)}  {desc}")
