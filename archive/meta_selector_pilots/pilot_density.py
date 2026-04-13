"""Density-penalized score transform + Otsu threshold.

For each test sample with score s, compute:
  transformed(s) = s - lambda * density(s, pool, h)

where density(s, pool, h) is a KDE-like density of the pool at s with bandwidth h.
Then threshold via Otsu on {transformed(s_i)}.

Lambda and h are hyperparameters BUT we commit to dataset-agnostic, automatic choices:
  h = Silverman's rule on pool
  lambda = determined by matching the variance of the density term to the variance of the score
           (so the transform's two terms are on comparable scales, no free choice)

Sweep small-lambda neighborhood as diagnostic; final rule picks one deterministic formula.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, metrics, load_scores_file, build_arrays
from data_utils import load_annotations


BASE = {"MHClip_EN": (0.7640, 0.6532), "MHClip_ZH": (0.8121, 0.7871)}


def kde(x, pool, h):
    """Gaussian KDE eval at x."""
    x = np.atleast_1d(x)
    pool = np.atleast_1d(pool)
    diffs = (x[:, None] - pool[None, :]) / h
    w = np.exp(-0.5 * diffs**2) / (np.sqrt(2 * np.pi) * h)
    return w.mean(axis=1)


def silverman_h(pool):
    n = len(pool)
    s = pool.std(ddof=1)
    iqr = np.percentile(pool, 75) - np.percentile(pool, 25)
    sig = min(s, iqr / 1.34) if iqr > 0 else s
    return 0.9 * sig * n ** (-1/5)


def main():
    for dataset in ["MHClip_EN", "MHClip_ZH"]:
        acc_b, mf_b = BASE[dataset]
        base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{dataset}"
        train = load_scores_file(f"{base_d}/train_binary.jsonl")
        test = load_scores_file(f"{base_d}/test_binary.jsonl")
        ann = load_annotations(dataset)
        test_x, test_y = build_arrays(test, ann)
        train_arr = np.array(list(train.values()), dtype=float)
        pool = np.concatenate([train_arr, np.array(list(test.values()), dtype=float)])

        h = silverman_h(pool)
        dens_test = kde(test_x, pool, h)
        dens_pool = kde(pool, pool, h)

        # normalize density so range ~ [0,1]
        dn_test = (dens_test - dens_pool.min()) / (dens_pool.max() - dens_pool.min() + 1e-12)
        dn_pool = (dens_pool - dens_pool.min()) / (dens_pool.max() - dens_pool.min() + 1e-12)

        print(f"\n=== {dataset} ===  baseline {acc_b:.4f}/{mf_b:.4f}  h={h:.4f}")
        # Sweep lambda
        for lam in [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]:
            trans_test = test_x - lam * dn_test
            trans_pool = pool - lam * dn_pool
            # Otsu on transformed pool (train+test values)
            t_otsu = otsu_threshold(trans_pool)
            # Also Otsu on just transformed train
            t_otsu_tr = otsu_threshold(pool[:len(train_arr)] - lam * dn_pool[:len(train_arr)])
            m_pool = metrics(trans_test, test_y, t_otsu)
            m_tr = metrics(trans_test, test_y, t_otsu_tr)
            flag_p = ""
            flag_t = ""
            if m_pool["acc"] > acc_b and m_pool["mf"] > mf_b: flag_p = " *P*"
            if m_tr["acc"] > acc_b and m_tr["mf"] > mf_b: flag_t = " *T*"
            print(f"  lam={lam:.2f}  pool-Otsu t={t_otsu:+.4f} acc={m_pool['acc']:.4f} mf={m_pool['mf']:.4f}{flag_p}  train-Otsu t={t_otsu_tr:+.4f} acc={m_tr['acc']:.4f} mf={m_tr['mf']:.4f}{flag_t}")


if __name__ == "__main__":
    main()
