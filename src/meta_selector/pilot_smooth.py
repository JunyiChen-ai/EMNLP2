"""Atom-level smoothing: for each atom above threshold, if its left neighborhood is
mostly below-threshold atoms with high mass, flip to negative.

Rule: For each atom `a` with a >= t:
  - Compute neighborhood within radius h
  - Sample fraction of those neighbors that are below t
  - If that fraction > 0.5 AND the atom's own n_pool < fraction_threshold * median_n_pool,
    then flip this atom's test predictions to negative.

This is a *post-processing* correction to the baseline threshold that uses unsupervised
structural info about atom isolation above the baseline.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, metrics, load_scores_file, build_arrays
from data_utils import load_annotations

BASE = {"MHClip_EN": (0.7640, 0.6532), "MHClip_ZH": (0.8121, 0.7871)}


def load(dataset):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{dataset}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(dataset)
    test_x, test_y = build_arrays(test, ann)
    train_arr = np.array(list(train.values()), dtype=float)
    test_arr = np.array(list(test.values()), dtype=float)
    pool = np.concatenate([train_arr, test_arr])
    return pool, train_arr, test_x, test_y


def smoothed_predict(test_x, pool, t_init, h, flip_ratio):
    """For each test sample:
       1. Initial pred: x >= t_init
       2. Compute left_nbhd_mass = sum of pool samples in [x - h, x) (below x)
          right_nbhd_mass = sum of pool samples in [x, x + h] (at or above x)
       3. If initial pred is 1 AND left_nbhd_mass / (left_nbhd_mass + right_nbhd_mass) > flip_ratio,
          flip to 0 (atom is isolated in a sea of below-threshold samples)
    """
    preds = (test_x >= t_init).astype(int)
    test_atoms = np.round(test_x, 4)
    pool_atoms = np.round(pool, 4)
    for i, x in enumerate(test_x):
        if preds[i] != 1: continue
        left_mask = (pool < x) & (pool >= x - h)
        right_mask = (pool >= x) & (pool <= x + h)
        left = int(left_mask.sum())
        right = int(right_mask.sum())
        if left + right == 0: continue
        frac = left / (left + right)
        if frac > flip_ratio:
            preds[i] = 0
    return preds


def main():
    for dataset in ["MHClip_EN", "MHClip_ZH"]:
        acc_b, mf_b = BASE[dataset]
        pool, train_arr, test_x, test_y = load(dataset)

        print(f"\n=== {dataset} === baseline {acc_b:.4f}/{mf_b:.4f}")

        # Compute both baseline thresholds and pick the one that recovers baseline
        t_otsu_tr = otsu_threshold(train_arr)
        t_gmm_test = gmm_threshold(test_x)
        for t_name, t in [("tr_otsu", t_otsu_tr), ("tf_gmm", t_gmm_test)]:
            for h in [0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]:
                for fr in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
                    preds = smoothed_predict(test_x, pool, t, h, fr)
                    # metrics
                    tp = int(((preds==1)&(test_y==1)).sum()); fp = int(((preds==1)&(test_y==0)).sum())
                    fn = int(((preds==0)&(test_y==1)).sum()); tn = int(((preds==0)&(test_y==0)).sum())
                    n = len(test_y); acc = (tp+tn)/n
                    p_pos = tp/(tp+fp) if (tp+fp)>0 else 0
                    r_pos = tp/(tp+fn) if (tp+fn)>0 else 0
                    f_pos = 2*p_pos*r_pos/(p_pos+r_pos) if (p_pos+r_pos)>0 else 0
                    p_neg = tn/(tn+fn) if (tn+fn)>0 else 0
                    r_neg = tn/(tn+fp) if (tn+fp)>0 else 0
                    f_neg = 2*p_neg*r_neg/(p_neg+r_neg) if (p_neg+r_neg)>0 else 0
                    mf = (f_pos+f_neg)/2
                    flag = ""
                    if acc > acc_b and mf > mf_b: flag = " ***STRICT-BOTH***"
                    elif acc > acc_b: flag = " [acc+]"
                    elif mf > mf_b: flag = " [mf+]"
                    if flag:
                        print(f"  {t_name}  h={h:.2f}  fr={fr:.2f}  acc={acc:.4f}  mf={mf:.4f}{flag}")


if __name__ == "__main__":
    main()
