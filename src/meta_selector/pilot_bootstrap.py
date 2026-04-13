"""Bootstrap Otsu ensemble: fit Otsu on N subsamples of the pool, then use the resulting
per-sample majority-vote as the label.

This is non-monotone in a subtle way: the same test sample gets a 'positive' vote from
bootstrap replicates whose threshold is <= sample score, and 'negative' otherwise. Since
all replicate thresholds are scalars, the majority vote IS monotone in the sample score —
if X's majority vote is 1, any Y > X also has majority vote >= 1.

So this doesn't help. What MIGHT help: *per-sample* leave-one-out bootstrap where the
sample's own membership affects the threshold fit, so "marginal" samples are inconsistent
across replicates. Still monotone.

Let me instead try: **threshold the score using a PAIRWISE comparison rule** — for each
test sample x, compute rank-based features over the pool. Since all monotone transforms
are equivalent, no help.

Pivot: Implement the **confusion-matrix self-consistency iteration** (direction 5 from
literature notes). Fit Otsu → get pseudo-labels on pool → compute pseudo-class-conditional
densities → re-threshold to Bayes-optimal under estimated priors → iterate.

This CAN converge to a different fixed point than single-pass Otsu because each iteration
re-estimates the class means and covariances.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, metrics, load_scores_file, build_arrays
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


def bayes_threshold(pool, pseudo_labels):
    """Given pool with pseudo-labels, compute the Bayes-optimal threshold assuming
    Gaussian class-conditionals."""
    pos = pool[pseudo_labels == 1]
    neg = pool[pseudo_labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.mean(pool)
    pi = len(pos) / len(pool)
    mu1, s1 = pos.mean(), pos.std() + 1e-6
    mu0, s0 = neg.mean(), neg.std() + 1e-6
    # Discriminant threshold
    xs = np.linspace(pool.min(), pool.max(), 5000)
    log_p1 = np.log(pi) - 0.5*np.log(2*np.pi*s1**2) - 0.5*((xs-mu1)/s1)**2
    log_p0 = np.log(1-pi) - 0.5*np.log(2*np.pi*s0**2) - 0.5*((xs-mu0)/s0)**2
    diff = log_p1 - log_p0
    # threshold where diff crosses 0
    cross = np.where(diff[:-1] * diff[1:] < 0)[0]
    if len(cross) == 0:
        return (mu0 + mu1) / 2
    # Pick the crossing closest to (mu0+mu1)/2 above mu0
    mid = (mu0 + mu1) / 2
    best = cross[np.argmin(np.abs(xs[cross] - mid))]
    return xs[best]


def iterate(pool, test_x, test_y, acc_b, mf_b, max_iter=20, init="otsu"):
    if init == "otsu":
        t = otsu_threshold(pool)
    else:
        t = np.median(pool)
    history = []
    for i in range(max_iter):
        pseudo = (pool >= t).astype(int)
        t_new = bayes_threshold(pool, pseudo)
        m = metrics(test_x, test_y, t_new)
        history.append({"iter": i, "t": t_new, "acc": m["acc"], "mf": m["mf"], "n_pos": int(pseudo.sum())})
        if abs(t_new - t) < 1e-6:
            break
        t = t_new
    return history


def main():
    for dataset in ["MHClip_EN", "MHClip_ZH"]:
        acc_b, mf_b = BASE[dataset]
        pool, train_arr, test_x, test_y = load(dataset)
        print(f"\n=== {dataset} === baseline {acc_b:.4f}/{mf_b:.4f}")

        for init in ["otsu", "median"]:
            hist = iterate(pool, test_x, test_y, acc_b, mf_b, init=init)
            print(f"  init={init}")
            for h in hist:
                flag = ""
                if h["acc"] > acc_b and h["mf"] > mf_b: flag = " ***STRICT-BOTH***"
                elif h["acc"] > acc_b: flag = " [acc+]"
                elif h["mf"] > mf_b: flag = " [mf+]"
                print(f"    iter={h['iter']:2d}  t={h['t']:.4f}  acc={h['acc']:.4f}  mf={h['mf']:.4f}  n_pos={h['n_pos']}{flag}")


if __name__ == "__main__":
    main()
