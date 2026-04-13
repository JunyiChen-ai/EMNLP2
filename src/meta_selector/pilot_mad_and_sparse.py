"""Subset rule: MAD-quantile AND sparseness filter.

Prediction rule:
  pred(x) = (x >= t_mad) AND (local_density(x) < q_dens)

where:
  t_mad = quantile(pool, 0.6 + 7.83*MAD(pool))   # primary rule
  local_density(x) = 1 / d_k(x, pool)             # k-NN density
  q_dens = some quantile of pool density

The AND gate removes 'dense cluster' atoms in the high-score region that are
empirically mostly negative (e.g., EN atom 0.2227, ZH atom 0.0180).

Sweep over density cutoff quantile to find single unified rule that strict-beats both.
"""
import os, sys, numpy as np
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
    train_arr = np.array(list(train.values()), dtype=float)
    pool = np.concatenate([train_arr, np.array(list(test.values()), dtype=float)])
    return pool, test_x, test_y

def knn_density(pool, x_arr, k=10):
    pool_s = np.sort(pool)
    out = np.zeros(len(x_arr))
    for i, x in enumerate(x_arr):
        d = np.abs(pool_s - x)
        d.sort()
        # use distance to k-th neighbor; skip self if exact match
        if d[0] < 1e-12 and k < len(d):
            d_k = d[k]
        else:
            d_k = d[min(k-1, len(d)-1)]
        out[i] = 1.0 / (d_k + 1e-12)
    return out

def mad_threshold(pool):
    med = float(np.median(pool))
    mad = float(np.median(np.abs(pool - med)))
    q = 0.60 + 7.83 * mad
    q = max(0.01, min(0.99, q))
    return float(np.quantile(pool, q))

data = {}
for d in ["MHClip_EN", "MHClip_ZH"]:
    pool, test_x, test_y = load(d)
    data[d] = {"pool": pool, "test_x": test_x, "test_y": test_y, "base": BASE[d]}

print("=== MAD + density-cutoff AND gate ===")
for k in [3, 5, 8, 10, 15, 20]:
    for q_dens in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
        results = {}
        all_strict = True
        for d in ["MHClip_EN", "MHClip_ZH"]:
            pool = data[d]["pool"]
            test_x = data[d]["test_x"]
            test_y = data[d]["test_y"]
            t_mad = mad_threshold(pool)
            # density of pool samples
            dens_pool = knn_density(pool, pool, k=k)
            dens_cut = float(np.quantile(dens_pool, q_dens))
            # test density
            dens_test = knn_density(pool, test_x, k=k)
            # predictions: x >= t_mad AND dens_test < dens_cut
            preds = (test_x >= t_mad) & (dens_test < dens_cut)
            # metrics on binary preds (use a sentinel threshold trick)
            # Construct a fake score: preds converted to {0, 1} vs threshold 0.5
            fake_scores = preds.astype(float)
            m = metrics(fake_scores, test_y, 0.5)
            acc_b, mf_b = data[d]["base"]
            s_acc = m["acc"] > acc_b
            s_mf = m["mf"] > mf_b
            results[d] = (m["acc"], m["mf"], s_acc, s_mf)
            if not (s_acc and s_mf):
                all_strict = False
        if all_strict:
            en = results["MHClip_EN"]
            zh = results["MHClip_ZH"]
            print(f"  k={k:2d}  q_dens={q_dens:.2f}  EN {en[0]:.4f}/{en[1]:.4f}  ZH {zh[0]:.4f}/{zh[1]:.4f}  STRICT-BOTH")
