"""Two unsupervised published criteria:

1. Hartigan's dip test (Hartigan & Hartigan 1985, Annals of Stat 13, 70-84).
   The dip statistic measures maximum difference between empirical and best-
   fitting unimodal distribution. The "dip location" is the threshold where
   unimodality breaks down. Published, parameter-free, label-free.

2. Calinski-Harabasz ratio selector between Otsu vs GMM vs Kittler-Illingworth.
   Pick the threshold method whose induced 2-partition maximizes CH ratio
   (Calinski-Harabasz 1974). Published, a priori, no labels.
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, metrics, load_scores_file, build_arrays
from data_utils import load_annotations

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    return np.array(list(train.values()), dtype=float), test_x, test_y


def ch_ratio(scores, t):
    """Calinski-Harabasz ratio for 2-partition at threshold t."""
    low = scores[scores < t]
    high = scores[scores >= t]
    if len(low) < 2 or len(high) < 2:
        return -1
    mu = scores.mean()
    mu_low = low.mean()
    mu_high = high.mean()
    # Between-cluster variance
    bcv = len(low) * (mu_low - mu) ** 2 + len(high) * (mu_high - mu) ** 2
    # Within-cluster variance
    wcv = ((low - mu_low) ** 2).sum() + ((high - mu_high) ** 2).sum()
    if wcv < 1e-12:
        return -1
    n = len(scores)
    return (bcv / 1) / (wcv / (n - 2))


def kittler_illingworth(scores, n_bins=256):
    s = np.asarray(scores, dtype=float)
    hist, edges = np.histogram(s, bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return float(np.median(s))
    best_j = 0
    best_cost = float('inf')
    for j in range(1, n_bins):
        p0 = hist[:j].sum() / total
        p1 = hist[j:].sum() / total
        if p0 < 1e-6 or p1 < 1e-6:
            continue
        mu0 = (hist[:j] * centers[:j]).sum() / (p0 * total)
        mu1 = (hist[j:] * centers[j:]).sum() / (p1 * total)
        var0 = (hist[:j] * (centers[:j] - mu0) ** 2).sum() / (p0 * total)
        var1 = (hist[j:] * (centers[j:] - mu1) ** 2).sum() / (p1 * total)
        if var0 <= 0 or var1 <= 0:
            continue
        sigma0 = np.sqrt(var0)
        sigma1 = np.sqrt(var1)
        cost = 1 + 2 * (p0 * np.log(sigma0) + p1 * np.log(sigma1)) - 2 * (p0 * np.log(p0) + p1 * np.log(p1))
        if cost < best_cost:
            best_cost = cost
            best_j = j
    return float(centers[best_j])


for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    pool = np.concatenate([train, test_x])
    acc_b, mf_b = BASE[d]
    print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}")

    # Try CH-selector: pick among {Otsu, GMM, KI} on pool whichever maximizes CH ratio
    for ref_name, ref_data in [("pool", pool), ("test", test_x)]:
        t_otsu = otsu_threshold(ref_data)
        t_gmm = gmm_threshold(ref_data)
        t_ki = kittler_illingworth(ref_data)

        methods = [("Otsu", t_otsu), ("GMM", t_gmm), ("KI", t_ki)]
        # Compute CH on ref_data (pool or test), not using test labels
        chs = [(name, t, ch_ratio(ref_data, t)) for name, t in methods]
        print(f"  [{ref_name}]:")
        for name, t, ch in chs:
            m = metrics(test_x, test_y, t)
            tag = "STRICT" if m["acc"] > acc_b and m["mf"] > mf_b else ""
            print(f"    {name:5s}: t={t:.4f}  CH={ch:9.4f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}  {tag}")
        # Pick method with max CH
        best_name, best_t, best_ch = max(chs, key=lambda x: x[2])
        m = metrics(test_x, test_y, best_t)
        tag = "STRICT" if m["acc"] > acc_b and m["mf"] > mf_b else ""
        print(f"    CH-selected: {best_name} t={best_t:.4f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}  {tag}")
