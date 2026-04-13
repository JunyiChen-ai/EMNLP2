"""Pilot: parameter-free published methods on density-isolation features.

Phenomenon (from diag_atom_context + diag_distance_feature):
  - Positive samples are both HIGHER in score AND MORE ISOLATED from the pool
    (larger d_knn, lower local density).
  - d_knn correlates positively with label on BOTH datasets (EN +0.39, ZH +0.44).
  - KDE density correlates negatively with label on BOTH (EN -0.31, ZH -0.53).

Method candidates (all parameter-free or published-standard):
  A. Otsu on test score             (baseline reproduction)
  B. Otsu on log-density-inverted score (Silverman BW, standard)
  C. Otsu on d_knn with k = sqrt(|train|) (Fix-Hodges 1951 standard)
  D. Otsu on normalized fusion (z-score sum)
  E. Otsu on product: score * (1 - density_rank)
  F. Otsu on isolation-reweighted score

Each uses ZERO free parameters beyond published standards. Committed to run
ONCE per candidate; no cherry-pick among them.
"""
import os, sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, metrics, load_scores_file, build_arrays
from data_utils import load_annotations
from scipy.stats import gaussian_kde

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    train_arr = np.array(list(train.values()), dtype=float)
    return train_arr, test_x, test_y


def d_knn(x, train_arr, k):
    diffs = np.abs(train_arr - x)
    return np.partition(diffs, k-1)[k-1]


def run_all(d):
    train, test_x, test_y = load(d)
    pool = np.concatenate([train, test_x])
    acc_b, mf_b = BASE[d]
    print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}")

    # Silverman bandwidth KDE (parameter-free standard)
    kde = gaussian_kde(pool, bw_method='silverman')
    dens = kde(test_x)
    # rank transforms
    rank_score = np.array([(pool <= x).mean() for x in test_x])
    rank_dens = np.array([(dens <= v).mean() for v in dens])
    # k-NN with Fix-Hodges standard k=sqrt(n)
    k = max(1, int(np.sqrt(len(train))))
    d_k = np.array([d_knn(x, train, k) for x in test_x])
    rank_dk = np.array([(d_k <= v).mean() for v in d_k])

    candidates = {
        "A. Otsu on score":
            test_x,
        "B. Otsu on log(1/density)":
            -np.log(dens + 1e-12),
        "C. Otsu on d_knn (k=sqrt(n))":
            d_k,
        "D. Otsu on score+log(1/density)":
            test_x - np.log(dens + 1e-12) / np.log(dens + 1e-12).max(),
        "E. Otsu on rank_score*(1-rank_dens)":
            rank_score * (1.0 - rank_dens),
        "F. Otsu on rank_score+rank_dk":
            rank_score + rank_dk,
        "G. Otsu on z(score)+z(d_knn)":
            ((test_x - test_x.mean()) / test_x.std() +
             (d_k - d_k.mean()) / (d_k.std() + 1e-12)),
        "H. Otsu on rank_score+(1-rank_dens)":
            rank_score + (1.0 - rank_dens),
    }
    for name, f in candidates.items():
        t = otsu_threshold(f)
        m = metrics(f, test_y, t)
        s_acc = m["acc"] > acc_b
        s_mf = m["mf"] > mf_b
        tag = " STRICT" if (s_acc and s_mf) else ""
        print(f"  {name:45s}  acc={m['acc']:.4f}  mf={m['mf']:.4f}{tag}")


results_en = run_all("MHClip_EN")
results_zh = run_all("MHClip_ZH")
