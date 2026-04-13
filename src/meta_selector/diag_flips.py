"""Single-atom flip search from baseline monotone cut.

Starting from baseline preds (score >= t*), try flipping each atom's decision and
see which flips (or combinations of 2-3 flips) produce strict-both wins.
This tells us whether strict-beat is even possible via LOCAL non-monotone moves.
"""
import json
import os
import sys
import numpy as np
from itertools import combinations

sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
from data_utils import load_annotations

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}
BASE_T = {"MHClip_EN": 0.2734, "MHClip_ZH": 0.0362}


def load(dataset):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{dataset}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(dataset)
    test_x, test_y = build_arrays(test, ann)
    return np.round(test_x, 4), test_y, test_x


def eval_preds(preds, y):
    tp = int(((preds==1)&(y==1)).sum()); fp = int(((preds==1)&(y==0)).sum())
    fn = int(((preds==0)&(y==1)).sum()); tn = int(((preds==0)&(y==0)).sum())
    n = len(y); acc = (tp+tn)/n
    p_pos = tp/(tp+fp) if (tp+fp)>0 else 0
    r_pos = tp/(tp+fn) if (tp+fn)>0 else 0
    f_pos = 2*p_pos*r_pos/(p_pos+r_pos) if (p_pos+r_pos)>0 else 0
    p_neg = tn/(tn+fn) if (tn+fn)>0 else 0
    r_neg = tn/(tn+fp) if (tn+fp)>0 else 0
    f_neg = 2*p_neg*r_neg/(p_neg+r_neg) if (p_neg+r_neg)>0 else 0
    return acc, (f_pos+f_neg)/2


def main():
    for dataset in ["MHClip_EN", "MHClip_ZH"]:
        acc_b, mf_b = BASE[dataset]
        test_atoms, y, test_x_raw = load(dataset)
        t = BASE_T[dataset]
        # Starting preds: threshold at t
        base_preds = (test_x_raw >= t).astype(int)
        base_acc, base_mf = eval_preds(base_preds, y)

        print(f"\n=== {dataset} ===  base at t={t}: acc={base_acc:.4f}  mf={base_mf:.4f}  (target {acc_b:.4f}/{mf_b:.4f})")

        uniques = sorted(set(test_atoms.tolist()))
        # Single flip: for each atom, flip its prediction
        flip_results = []
        for a in uniques:
            mask = (test_atoms == a)
            new_preds = base_preds.copy()
            new_preds[mask] = 1 - new_preds[mask]
            acc, mf = eval_preds(new_preds, y)
            flip_results.append({"a": float(a), "acc": acc, "mf": mf, "n": int(mask.sum()),
                                 "new_label": int(new_preds[mask][0]) if mask.sum()>0 else -1})

        # Strict-both single flips
        wins1 = [r for r in flip_results if r["acc"] > acc_b and r["mf"] > mf_b]
        print(f"  single-flip strict-both wins: {len(wins1)}")
        for r in wins1[:12]:
            print(f"    flip a={r['a']:.4f} n={r['n']} -> new_label={r['new_label']}  acc={r['acc']:.4f}  mf={r['mf']:.4f}")

        # Two-flip combinations
        print(f"  --- two-flip combinations ---")
        wins2 = []
        for i in range(len(uniques)):
            for j in range(i+1, len(uniques)):
                a1 = uniques[i]; a2 = uniques[j]
                mask1 = (test_atoms == a1); mask2 = (test_atoms == a2)
                new_preds = base_preds.copy()
                new_preds[mask1] = 1 - new_preds[mask1]
                new_preds[mask2] = 1 - new_preds[mask2]
                acc, mf = eval_preds(new_preds, y)
                if acc > acc_b and mf > mf_b:
                    wins2.append({"a1": float(a1), "a2": float(a2), "acc": acc, "mf": mf,
                                  "n1": int(mask1.sum()), "n2": int(mask2.sum())})
        print(f"  two-flip strict-both wins: {len(wins2)}")
        for r in wins2[:20]:
            print(f"    flip a={r['a1']:.4f}(n={r['n1']}) + a={r['a2']:.4f}(n={r['n2']})  acc={r['acc']:.4f}  mf={r['mf']:.4f}")


if __name__ == "__main__":
    main()
