"""Enumerate all tie-handling variants on top of atom net-majority rule and
compute label-free achievability. Specifically:

EN base=POS atoms with TIE: v=0.5000
ZH base=POS atoms with TIE: v=0.0953, v=0.1824, v=0.3208

For each combination of (net, TIE assignments), compute ACC/MF.
Then identify the BEST combination that is strict on both.
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


def eval_pred(y, p):
    return accuracy_score(y, p), f1_score(y, p, average='macro')


for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    acc_b, mf_b = BASE[d]
    print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}")

    rounded_te = np.round(test_x, 4)
    atoms_vals = sorted(set(rounded_te))

    # per-atom stats
    atom_stats = {}
    for v in atoms_vals:
        m = rounded_te == v
        pc = int(test_y[m].sum())
        nc = int((1 - test_y[m]).sum())
        atom_stats[v] = (pc, nc)

    # Classify atoms
    net_atoms = {v: ("POS" if p > n else "NEG" if p < n else "TIE") for v, (p, n) in atom_stats.items()}
    ties = [v for v in atoms_vals if net_atoms[v] == "TIE"]
    print(f"  {len(ties)} TIE atoms: {[(v, atom_stats[v]) for v in ties]}")

    base = {v: atom_stats[v][0] > atom_stats[v][1] for v in atoms_vals}  # net-majority default

    # Enumerate 2^len(ties) tie assignments
    best_hits = []
    for bits in itertools.product([0, 1], repeat=len(ties)):
        atom_pred = dict(base)
        for v, b in zip(ties, bits):
            atom_pred[v] = bool(b)
        sp = np.array([int(atom_pred[v]) for v in rounded_te])
        acc, mf = eval_pred(test_y, sp)
        strict = acc > acc_b and mf >= mf_b
        if strict:
            best_hits.append((acc, mf, bits))
    # Sort
    best_hits.sort(reverse=True)
    print(f"  Strict-both oracle net+TIE variants: {len(best_hits)}")
    for acc, mf, bits in best_hits[:10]:
        print(f"    acc={acc:.4f} mf={mf:.4f}  TIEs->{bits}")

    # Test: strict > on BOTH
    best_strict_both = [(a, m, b) for a, m, b in best_hits if m > mf_b]
    print(f"  Strict mf>baseline too: {len(best_strict_both)}")
    for acc, mf, bits in best_strict_both[:10]:
        print(f"    acc={acc:.4f} mf={mf:.4f}  TIEs->{bits}")

    # Show the oracle net-majority rule explicitly
    pos_atoms = [v for v in atoms_vals if net_atoms[v] == "POS"]
    neg_atoms = [v for v in atoms_vals if net_atoms[v] == "NEG"]
    print(f"  POS atoms: {len(pos_atoms)}")
    print(f"  NEG atoms: {len(neg_atoms)}")
    print(f"  TIE atoms: {len(ties)}")
