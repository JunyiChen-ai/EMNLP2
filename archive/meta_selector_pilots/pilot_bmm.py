"""Beta mixture model threshold.

Fit 2-component Beta mixture to pool (train+test) scores (clipped to (eps, 1-eps)).
Threshold = posterior-0.5 crossover. Also try on train-only.
"""
import json
import os
import sys
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import beta as beta_dist

sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
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


def fit_beta_method_of_moments(x):
    """Fit Beta(a,b) by method of moments."""
    m = np.mean(x)
    v = np.var(x, ddof=0)
    if v <= 0 or m <= 0 or m >= 1:
        return 1.0, 1.0
    k = (m*(1-m))/v - 1
    if k <= 0:
        return 1.0, 1.0
    a = m * k
    b = (1 - m) * k
    return max(a, 0.01), max(b, 0.01)


def bmm_em(x, n_iter=200, tol=1e-6):
    eps = 1e-4
    x = np.clip(x, eps, 1 - eps)
    n = len(x)
    # Init: split at median
    med = np.median(x)
    low = x[x < med]; high = x[x >= med]
    a0, b0 = fit_beta_method_of_moments(low)
    a1, b1 = fit_beta_method_of_moments(high)
    w0 = len(low) / n; w1 = len(high) / n
    log_lik_prev = -np.inf
    for it in range(n_iter):
        log_p0 = np.log(max(w0, 1e-12)) + beta_dist.logpdf(x, a0, b0)
        log_p1 = np.log(max(w1, 1e-12)) + beta_dist.logpdf(x, a1, b1)
        lmax = np.maximum(log_p0, log_p1)
        log_sum = lmax + np.log(np.exp(log_p0-lmax) + np.exp(log_p1-lmax))
        r0 = np.exp(log_p0 - log_sum)
        r1 = np.exp(log_p1 - log_sum)
        w0 = r0.sum() / n; w1 = r1.sum() / n
        # Weighted MoM for each component
        m0 = (r0*x).sum() / r0.sum(); v0 = (r0*(x-m0)**2).sum() / r0.sum()
        m1 = (r1*x).sum() / r1.sum(); v1 = (r1*(x-m1)**2).sum() / r1.sum()
        if v0 > 0 and 0 < m0 < 1:
            k0 = (m0*(1-m0))/v0 - 1
            if k0 > 0:
                a0 = max(m0*k0, 0.01); b0 = max((1-m0)*k0, 0.01)
        if v1 > 0 and 0 < m1 < 1:
            k1 = (m1*(1-m1))/v1 - 1
            if k1 > 0:
                a1 = max(m1*k1, 0.01); b1 = max((1-m1)*k1, 0.01)
        ll = log_sum.sum()
        if abs(ll - log_lik_prev) < tol: break
        log_lik_prev = ll
    return {"w0": w0, "w1": w1, "a0": a0, "b0": b0, "a1": a1, "b1": b1}


def bmm_threshold(params):
    """Find x where posterior-1 >= 0.5."""
    w0, w1 = params["w0"], params["w1"]
    a0, b0 = params["a0"], params["b0"]
    a1, b1 = params["a1"], params["b1"]
    xs = np.linspace(1e-4, 1-1e-4, 4000)
    p0 = w0 * beta_dist.pdf(xs, a0, b0)
    p1 = w1 * beta_dist.pdf(xs, a1, b1)
    post = p1 / (p0 + p1 + 1e-12)
    # Ensure component 1 is the high mean
    m0 = a0 / (a0 + b0); m1 = a1 / (a1 + b1)
    if m0 > m1:
        post = 1 - post
    idx = np.where(post >= 0.5)[0]
    if len(idx) == 0:
        return 0.5
    return xs[idx[0]]


def main():
    for dataset in ["MHClip_EN", "MHClip_ZH"]:
        acc_b, mf_b = BASE[dataset]
        pool, train_arr, test_x, test_y = load(dataset)

        print(f"\n=== {dataset} === baseline {acc_b:.4f}/{mf_b:.4f}")
        for name, x in [("pool", pool), ("train", train_arr), ("test", test_x)]:
            try:
                p = bmm_em(x)
                t = bmm_threshold(p)
                m = metrics(test_x, test_y, t)
                flag = ""
                if m["acc"] > acc_b and m["mf"] > mf_b: flag = " ***STRICT-BOTH***"
                print(f"  BMM-{name:<5} w0={p['w0']:.3f} a0={p['a0']:.2f} b0={p['b0']:.2f} a1={p['a1']:.2f} b1={p['b1']:.2f}  t={t:.4f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}{flag}")
            except Exception as e:
                print(f"  BMM-{name}: ERR {e}")


if __name__ == "__main__":
    main()
