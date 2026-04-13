"""Non-monotone atom-subset oracle ceiling.

For each (4-decimal) atom in the test pool, compute its conditional positive rate.
Sort atoms by positive-rate and find the best threshold on this reordered axis.
This is the unsupervised upper bound assuming you could perfectly reorder atoms by
evidential strength (which you cannot without labels, but gives an optimistic ceiling).
Also report the fully-unconstrained subset-selection oracle (pick any subset of atoms).
"""
import json
import os
import sys
import numpy as np
from itertools import combinations

sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
from data_utils import load_annotations

BASE = {"MHClip_EN": (0.7640, 0.6532), "MHClip_ZH": (0.8121, 0.7871)}

def main():
    for dataset in ["MHClip_EN", "MHClip_ZH"]:
        acc_b, mf_b = BASE[dataset]
        base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{dataset}"
        train = load_scores_file(f"{base_d}/train_binary.jsonl")
        test = load_scores_file(f"{base_d}/test_binary.jsonl")
        ann = load_annotations(dataset)
        test_x, test_y = build_arrays(test, ann)

        # 4-decimal rounding
        test_atoms = np.round(test_x, 4)
        unique_atoms = sorted(set(test_atoms.tolist()))
        # Conditional positive rate for each atom
        atom_stats = []
        for a in unique_atoms:
            mask = (test_atoms == a)
            n = int(mask.sum())
            pos = int(test_y[mask].sum())
            rate = pos / n if n else 0.0
            atom_stats.append({"atom": a, "n": n, "pos": pos, "rate": rate})

        # Sort atoms by rate descending (non-monotone oracle on score, but monotone on rate)
        sorted_by_rate = sorted(atom_stats, key=lambda r: -r["rate"])

        # Sweep: pick top-k atoms as positive
        print(f"\n=== {dataset} ===  baseline {acc_b:.4f}/{mf_b:.4f}  n_test={len(test_x)} n_pos={int(test_y.sum())}")
        best = {"acc": 0.0, "mf": 0.0, "k": 0}
        cum_pos = 0
        cum_n = 0
        for k in range(1, len(sorted_by_rate) + 1):
            selected = set(s["atom"] for s in sorted_by_rate[:k])
            preds = np.array([1 if a in selected else 0 for a in test_atoms], dtype=int)
            tp = int(((preds==1)&(test_y==1)).sum())
            fp = int(((preds==1)&(test_y==0)).sum())
            fn = int(((preds==0)&(test_y==1)).sum())
            tn = int(((preds==0)&(test_y==0)).sum())
            acc = (tp+tn)/len(test_y)
            p_pos = tp/(tp+fp) if (tp+fp)>0 else 0
            r_pos = tp/(tp+fn) if (tp+fn)>0 else 0
            f_pos = 2*p_pos*r_pos/(p_pos+r_pos) if (p_pos+r_pos)>0 else 0
            p_neg = tn/(tn+fn) if (tn+fn)>0 else 0
            r_neg = tn/(tn+fp) if (tn+fp)>0 else 0
            f_neg = 2*p_neg*r_neg/(p_neg+r_neg) if (p_neg+r_neg)>0 else 0
            mf = (f_pos+f_neg)/2
            if acc > acc_b and mf > mf_b:
                print(f"  ORACLE k={k}  acc={acc:.4f}  mf={mf:.4f}  atoms={sorted(list(selected))[:5]}...")
            if acc > best["acc"] or (acc == best["acc"] and mf > best["mf"]):
                best = {"acc": acc, "mf": mf, "k": k}
        print(f"  non-monotone oracle: best acc={best['acc']:.4f}  mf={best['mf']:.4f}  k={best['k']}")

        # Contiguous (monotone-threshold) oracle for comparison
        by_score = sorted(atom_stats, key=lambda r: r["atom"])
        best_mono = {"acc": 0.0, "mf": 0.0, "k": 0}
        for k in range(len(by_score) + 1):
            # Predict positive if atom >= by_score[len-k]
            selected = set(r["atom"] for r in by_score[len(by_score)-k:])
            preds = np.array([1 if a in selected else 0 for a in test_atoms], dtype=int)
            tp = int(((preds==1)&(test_y==1)).sum()); fp = int(((preds==1)&(test_y==0)).sum())
            fn = int(((preds==0)&(test_y==1)).sum()); tn = int(((preds==0)&(test_y==0)).sum())
            acc = (tp+tn)/len(test_y)
            p_pos = tp/(tp+fp) if (tp+fp)>0 else 0
            r_pos = tp/(tp+fn) if (tp+fn)>0 else 0
            f_pos = 2*p_pos*r_pos/(p_pos+r_pos) if (p_pos+r_pos)>0 else 0
            p_neg = tn/(tn+fn) if (tn+fn)>0 else 0
            r_neg = tn/(tn+fp) if (tn+fp)>0 else 0
            f_neg = 2*p_neg*r_neg/(p_neg+r_neg) if (p_neg+r_neg)>0 else 0
            mf = (f_pos+f_neg)/2
            if acc > best_mono["acc"]:
                best_mono = {"acc": acc, "mf": mf, "k": k}
        print(f"  monotone-threshold oracle: best acc={best_mono['acc']:.4f}  mf={best_mono['mf']:.4f}  k={best_mono['k']}")

        # Show per-atom rates
        print(f"  atoms sorted by score (with rates):")
        for r in by_score:
            if r["n"] > 0:
                print(f"    atom={r['atom']:.4f}  n={r['n']:3d}  pos={r['pos']:2d}  rate={r['rate']:.3f}")


if __name__ == "__main__":
    main()
