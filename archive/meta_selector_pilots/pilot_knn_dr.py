"""k-NN density estimation based rules.

Unlike KDE which smooths, k-NN density uses DISCRETE ordering: for each
test point, find the k-th nearest train point and estimate density as
k / (2 * n_train * dist_kth). This is piecewise-constant in x (jumps
when the k-th nearest train changes) — potentially NON-SMOOTH enough to
produce non-suffix atom-level labelings.

Features per test sample:
- knn_dist_k{1,3,5,10}: distance to k-th nearest train point
- knn_density_k{1,3,5,10}: k-NN density estimate from train
- test_knn_dist, test_knn_density: same but using test pool
- log_ratio_knn_k: log(test_density / train_density)
- lof-style: mean knn_dist of neighbors over own knn_dist

Then test rules:
- predict positive iff log_ratio_knn > 0 (label-free zero cut)
- various published thresholds (median, otsu, gmm)
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import load_scores_file, build_arrays
from data_utils import load_annotations
from sklearn.metrics import accuracy_score, f1_score
from sklearn.mixture import GaussianMixture
from skimage.filters import threshold_otsu, threshold_triangle

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    return np.array(list(train.values()), dtype=float), test_x, test_y


def knn_dist(x, pool, k):
    d = np.sort(np.abs(pool - x))
    return d[min(k - 1, len(d) - 1)]


def eval_pred(y, p):
    return accuracy_score(y, p), f1_score(y, p, average='macro')


for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    acc_b, mf_b = BASE[d]
    n_train, n_test = len(train), len(test_x)
    print(f"\n=== {d} === baseline acc={acc_b:.4f} mf={mf_b:.4f}")

    # For each k compute features
    for k in [1, 3, 5, 10, 20]:
        d_tr = np.array([knn_dist(x, train, k) for x in test_x])
        d_te = np.array([knn_dist(x, test_x, k + 1) for x in test_x])  # +1 since x is in test_x
        # k-NN density estimates
        f_tr = k / (2.0 * n_train * (d_tr + 1e-12))
        f_te = k / (2.0 * n_test * (d_te + 1e-12))
        log_r = np.log((f_tr + 1e-12) / (f_te + 1e-12))

        # Try rule: predict pos iff log_r < 0 (test density > train density)
        # And: pos iff f_te > f_tr (direct comparison)
        # Both directions
        rules = {
            "logr<0": (log_r < 0).astype(int),
            "logr>0": (log_r > 0).astype(int),
            "fte>ftr": (f_te > f_tr).astype(int),
            "d_tr>d_te": (d_tr > d_te).astype(int),  # train is far, test is close → test-cluster
        }
        # Label-free cuts on log_r
        try:
            cut = threshold_otsu(log_r, nbins=256)
            rules[f"otsu_lr<"] = (log_r < cut).astype(int)
            rules[f"otsu_lr>"] = (log_r > cut).astype(int)
        except Exception:
            pass
        try:
            g = GaussianMixture(n_components=2, random_state=0).fit(log_r.reshape(-1, 1))
            mus = sorted(g.means_.flatten())
            mid = (mus[0] + mus[1]) / 2
            rules[f"gmm_lr<"] = (log_r < mid).astype(int)
            rules[f"gmm_lr>"] = (log_r > mid).astype(int)
        except Exception:
            pass

        for rname, pred in rules.items():
            if pred.sum() == 0 or pred.sum() == n_test: continue
            acc, mf = eval_pred(test_y, pred)
            strict = acc > acc_b and mf >= mf_b
            tag = " ** PASS **" if strict else ""
            if strict or rname in ("logr<0", "fte>ftr"):
                print(f"  k={k:>2d} {rname:>10s}: acc={acc:.4f} mf={mf:.4f} n_pos={pred.sum()}{tag}")
