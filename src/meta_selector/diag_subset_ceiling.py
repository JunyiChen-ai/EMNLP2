"""Oracle subset ceiling: what's the best possible metric using ANY subset of atoms?

For each atom: assign it label 1 iff its test positive count > test negative count.
This is the best-possible non-suffix subset rule (uses test labels as oracle).
"""
import os, sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from collections import defaultdict
from quick_eval_all import metrics, load_scores_file, build_arrays
from data_utils import load_annotations

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}

def analyze(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    train_arr = np.array(list(train.values()), dtype=float)
    pool = np.concatenate([train_arr, np.array(list(test.values()), dtype=float)])

    # Per-atom test counts
    atom_pos = defaultdict(int)
    atom_neg = defaultdict(int)
    for x, y in zip(test_x, test_y):
        a = round(float(x), 4)
        if y == 1:
            atom_pos[a] += 1
        else:
            atom_neg[a] += 1

    atoms = sorted(set(atom_pos.keys()) | set(atom_neg.keys()))

    # Oracle subset rule #1: flag atoms where pos > neg
    def eval_subset(flag_fn):
        flagged_atoms = set(a for a in atoms if flag_fn(a))
        tp = sum(atom_pos[a] for a in flagged_atoms)
        fp = sum(atom_neg[a] for a in flagged_atoms)
        fn = sum(atom_pos[a] for a in atoms if a not in flagged_atoms)
        tn = sum(atom_neg[a] for a in atoms if a not in flagged_atoms)
        n = tp+fp+fn+tn
        acc = (tp+tn)/n
        prec_p = tp/(tp+fp) if (tp+fp)>0 else 0
        rec_p = tp/(tp+fn) if (tp+fn)>0 else 0
        f1_p = 2*prec_p*rec_p/(prec_p+rec_p) if (prec_p+rec_p)>0 else 0
        prec_n = tn/(tn+fn) if (tn+fn)>0 else 0
        rec_n = tn/(tn+fp) if (tn+fp)>0 else 0
        f1_n = 2*prec_n*rec_n/(prec_n+rec_n) if (prec_n+rec_n)>0 else 0
        return acc, (f1_p+f1_n)/2

    acc_b, mf_b = BASE[d]

    # 1. Pos > neg oracle
    acc1, mf1 = eval_subset(lambda a: atom_pos[a] > atom_neg[a])
    print(f"\n=== {d} ===  baseline {acc_b:.4f}/{mf_b:.4f}")
    print(f"  subset (pos>neg oracle):     acc={acc1:.4f}  mf={mf1:.4f}")

    # 2. Pos >= neg
    acc2, mf2 = eval_subset(lambda a: atom_pos[a] >= atom_neg[a] and atom_pos[a] > 0)
    print(f"  subset (pos>=neg & pos>0):   acc={acc2:.4f}  mf={mf2:.4f}")
    selected = sorted([a for a in atoms if atom_pos[a] >= atom_neg[a] and atom_pos[a] > 0])
    print(f"    selected atoms: {selected}")
    suffix_t = sorted(atoms)[len(atoms)//2] if atoms else 0
    # which of the selected atoms are BELOW MAD threshold?
    mad = float(np.median(np.abs(pool - np.median(pool))))
    q = 0.6 + 7.83 * mad
    t_mad = float(np.quantile(pool, q))
    non_suffix = [a for a in selected if a < t_mad]
    missed_suffix = [a for a in atoms if a >= t_mad and a not in selected]
    print(f"    non-suffix selections (oracle wants, MAD misses): {non_suffix}")
    print(f"    MAD-flags-oracle-doesnt (oracle wants to drop): {missed_suffix}")

    # 3. Pos rate > 0.5
    acc3, mf3 = eval_subset(lambda a: (atom_pos[a]/(atom_pos[a]+atom_neg[a]+1e-9)) > 0.5)
    print(f"  subset (rate > 0.5):         acc={acc3:.4f}  mf={mf3:.4f}")

    # 4. Pos rate >= 0.5
    acc4, mf4 = eval_subset(lambda a: (atom_pos[a]/(atom_pos[a]+atom_neg[a]+1e-9)) >= 0.5 and atom_pos[a] > 0)
    print(f"  subset (rate >= 0.5):        acc={acc4:.4f}  mf={mf4:.4f}")

    # 5. Best suffix (all atoms >= threshold)
    print(f"  --- best suffix ---")
    best_suffix = (0, 0, 0)
    for i in range(len(atoms)+1):
        flagged = set(atoms[i:])
        acc, mf = eval_subset(lambda a: a in flagged)
        if (acc, mf) > (best_suffix[1], best_suffix[2]):
            best_suffix = (atoms[i] if i < len(atoms) else 1.0, acc, mf)
    print(f"  best suffix threshold={best_suffix[0]:.4f}  acc={best_suffix[1]:.4f}  mf={best_suffix[2]:.4f}")

    # 6. What does the MAD rule flag?
    mad = float(np.median(np.abs(pool - np.median(pool))))
    q = 0.6 + 7.83 * mad
    t_mad = float(np.quantile(pool, q))
    flagged_mad = set(a for a in atoms if a >= t_mad)
    acc_mad, mf_mad = eval_subset(lambda a: a >= t_mad)
    print(f"  MAD rule: t={t_mad:.4f}  flagged atoms={sorted(flagged_mad)}")
    print(f"  MAD result: acc={acc_mad:.4f}  mf={mf_mad:.4f}")

for d in ["MHClip_EN", "MHClip_ZH"]:
    analyze(d)
