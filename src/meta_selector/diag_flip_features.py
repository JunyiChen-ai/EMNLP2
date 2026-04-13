"""For each atom, compute all candidate unsupervised features, then see which features
best separate "should-flip" from "should-not-flip" atoms.

should-flip: single-atom flip from baseline gives strict-both (or at least improvement)
should-not-flip: single-atom flip hurts both metrics
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
from data_utils import load_annotations

BASE = {"MHClip_EN": (0.7640, 0.6532), "MHClip_ZH": (0.8121, 0.7871)}
BASE_T = {"MHClip_EN": 0.2734, "MHClip_ZH": 0.0362}


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
        base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{dataset}"
        train = load_scores_file(f"{base_d}/train_binary.jsonl")
        test = load_scores_file(f"{base_d}/test_binary.jsonl")
        ann = load_annotations(dataset)
        test_x, test_y = build_arrays(test, ann)
        train_arr = np.array(list(train.values()), dtype=float)
        test_arr = np.array(list(test.values()), dtype=float)
        pool = np.concatenate([train_arr, test_arr])

        test_a = np.round(test_x, 4)
        train_a = np.round(train_arr, 4)
        pool_a = np.round(pool, 4)
        t = BASE_T[dataset]
        base_preds = (test_x >= t).astype(int)
        base_acc, base_mf = eval_preds(base_preds, y=test_y)

        uniques = sorted(set(pool_a.tolist()))
        print(f"\n=== {dataset} ===  base {base_acc:.4f}/{base_mf:.4f}")
        print(f"  {'atom':>8} {'above':>5} {'n_pool':>6} {'n_train':>7} {'n_test':>6} {'nbhd':>5} {'gap_l':>8} {'gap_r':>8}  flip_acc  flip_mf  verdict")
        for a in uniques:
            # features
            n_pool = int((pool_a == a).sum())
            n_train = int((train_a == a).sum())
            n_test = int((test_a == a).sum())
            nbhd = int(((pool_a != a) & (np.abs(pool_a - a) < 0.05)).sum())
            # Gap to nearest lower/higher atom
            lo = [x for x in uniques if x < a]
            hi = [x for x in uniques if x > a]
            gap_l = (a - lo[-1]) if lo else 0.0
            gap_r = (hi[0] - a) if hi else 0.0

            above = int(a >= t)
            # flip this atom's test samples
            mask = (test_a == a)
            if mask.sum() == 0:
                continue
            new_preds = base_preds.copy()
            new_preds[mask] = 1 - new_preds[mask]
            fa, fm = eval_preds(new_preds, test_y)
            verdict = ""
            if fa > acc_b and fm > mf_b: verdict = "STRICT-BOTH"
            elif fa > base_acc or fm > base_mf: verdict = "partial"
            elif fa < base_acc and fm < base_mf: verdict = "hurts"
            print(f"  {a:>8.4f} {above:>5} {n_pool:>6} {n_train:>7} {n_test:>6} {nbhd:>5} {gap_l:>8.4f} {gap_r:>8.4f}  {fa:.4f}   {fm:.4f}  {verdict}")


if __name__ == "__main__":
    main()
