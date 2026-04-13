"""EM-refined 2-component mixture threshold.

Initialize with Otsu (k-means on histogram minimizes intra-variance), then run
EM to convergence to find MLE 2-component mixture parameters, then compute
Bayes-optimal decision boundary.

Published method: Fraley-Raftery 1998 "How many clusters? Which clustering
method?" and McLachlan-Peel 2000 "Finite Mixture Models". Standard parameters.
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, metrics, load_scores_file, build_arrays
from data_utils import load_annotations
from sklearn.mixture import GaussianMixture

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    return np.array(list(train.values()), dtype=float), test_x, test_y


def em_threshold(scores, init_thresh=None):
    """Fit 2-component GMM, return decision boundary."""
    s = np.asarray(scores, dtype=float).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
    if init_thresh is not None:
        # Initialize with labels from Otsu threshold
        labels = (s.flatten() >= init_thresh).astype(int)
        if labels.sum() > 0 and (1 - labels).sum() > 0:
            means = np.array([s[labels == 0].mean(), s[labels == 1].mean()]).reshape(-1, 1)
            gmm.means_init = means
    gmm.fit(s)
    mu0, mu1 = gmm.means_.flatten()
    v0, v1 = gmm.covariances_.flatten()
    p0, p1 = gmm.weights_
    # Bayes decision boundary: p0*N(x|mu0,v0) = p1*N(x|mu1,v1)
    # Log form: log(p0) - 0.5*log(v0) - (x-mu0)^2/(2*v0) = log(p1) - 0.5*log(v1) - (x-mu1)^2/(2*v1)
    # Quadratic in x
    a = 1/(2*v1) - 1/(2*v0)
    b = mu0/v0 - mu1/v1
    c = mu1**2/(2*v1) - mu0**2/(2*v0) + 0.5*np.log(v1/v0) + np.log(p0/p1)
    if abs(a) < 1e-12:
        # Linear: bx + c = 0
        t = -c / b
    else:
        disc = b**2 - 4*a*c
        if disc < 0:
            t = (mu0 + mu1) / 2
        else:
            r1 = (-b + np.sqrt(disc)) / (2*a)
            r2 = (-b - np.sqrt(disc)) / (2*a)
            # Pick root between mu0 and mu1
            mu_lo, mu_hi = min(mu0, mu1), max(mu0, mu1)
            if mu_lo <= r1 <= mu_hi:
                t = r1
            elif mu_lo <= r2 <= mu_hi:
                t = r2
            else:
                t = (mu0 + mu1) / 2
    return float(t)


for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    pool = np.concatenate([train, test_x])
    acc_b, mf_b = BASE[d]
    print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}")

    # EM on test (TF)
    t_em = em_threshold(test_x)
    m = metrics(test_x, test_y, t_em)
    print(f"  EM (TF): t={t_em:.4f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}  "
          f"{'STRICT' if m['acc']>acc_b and m['mf']>mf_b else ''}")

    # EM on pool
    t_em = em_threshold(pool)
    m = metrics(test_x, test_y, t_em)
    print(f"  EM (pool): t={t_em:.4f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}  "
          f"{'STRICT' if m['acc']>acc_b and m['mf']>mf_b else ''}")

    # EM on test, initialized from pool-Otsu
    t_init = otsu_threshold(pool)
    t_em = em_threshold(test_x, init_thresh=t_init)
    m = metrics(test_x, test_y, t_em)
    print(f"  EM (TF, init=pool-Otsu): t={t_em:.4f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}  "
          f"{'STRICT' if m['acc']>acc_b and m['mf']>mf_b else ''}")

    # EM on pool, initialized from test-Otsu
    t_init = otsu_threshold(test_x)
    t_em = em_threshold(pool, init_thresh=t_init)
    m = metrics(test_x, test_y, t_em)
    print(f"  EM (pool, init=TF-Otsu): t={t_em:.4f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}  "
          f"{'STRICT' if m['acc']>acc_b and m['mf']>mf_b else ''}")
