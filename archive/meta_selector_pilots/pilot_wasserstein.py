"""Wasserstein-distance-maximizing threshold.

For each candidate cut t, split pool into {below t} and {above t}, compute
Wasserstein-1 distance between the two empirical distributions after
normalizing each to unit total mass, and pick the t that maximizes it.

Different from BCV: BCV uses mean-squared between-cluster distance,
Wasserstein captures the full distributional gap. Parameter-free, published.

Also: Jensen-Shannon divergence threshold — same scan, different criterion.
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, metrics, load_scores_file, build_arrays
from data_utils import load_annotations
from scipy.stats import wasserstein_distance

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    return np.array(list(train.values()), dtype=float), test_x, test_y


def wasserstein_threshold(scores):
    """Scan candidate thresholds at every inter-sample gap. Pick the one
    maximizing Wasserstein-1 distance between {below} and {above}.
    """
    sorted_s = np.sort(scores)
    best_t = None
    best_w = -1
    for i in range(1, len(sorted_s)):
        if sorted_s[i] - sorted_s[i-1] < 1e-9:
            continue
        t = (sorted_s[i] + sorted_s[i-1]) / 2
        below = scores[scores < t]
        above = scores[scores >= t]
        if len(below) < 2 or len(above) < 2:
            continue
        w = wasserstein_distance(below, above)
        if w > best_w:
            best_w = w
            best_t = t
    return best_t


def js_threshold(scores, n_bins=100):
    """Jensen-Shannon divergence threshold via histogram."""
    from scipy.spatial.distance import jensenshannon
    hist, edges = np.histogram(scores, bins=n_bins, density=False)
    centers = (edges[:-1] + edges[1:]) / 2
    best_t = None
    best_js = -1
    for j in range(5, n_bins - 5):
        p_below = hist[:j].astype(float)
        p_above = hist[j:].astype(float)
        if p_below.sum() < 5 or p_above.sum() < 5:
            continue
        p_below = p_below / p_below.sum()
        p_above = p_above / p_above.sum()
        # Pad to same length for JS
        max_len = max(len(p_below), len(p_above))
        pb = np.concatenate([p_below, np.zeros(max_len - len(p_below))])
        pa = np.concatenate([np.zeros(max_len - len(p_above)), p_above])
        js = jensenshannon(pb, pa) ** 2
        if js > best_js:
            best_js = js
            best_t = centers[j]
    return best_t if best_t is not None else float(np.median(scores))


for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    pool = np.concatenate([train, test_x])
    acc_b, mf_b = BASE[d]
    print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}")

    for ref_name, ref in [("pool", pool), ("test", test_x)]:
        t_w = wasserstein_threshold(ref)
        m = metrics(test_x, test_y, t_w)
        tag = "STRICT" if m["acc"] > acc_b and m["mf"] > mf_b else ""
        print(f"  Wasserstein-max[{ref_name}]: t={t_w:.4f}  acc={m['acc']:.4f} mf={m['mf']:.4f}  {tag}")

        t_js = js_threshold(ref)
        m = metrics(test_x, test_y, t_js)
        tag = "STRICT" if m["acc"] > acc_b and m["mf"] > mf_b else ""
        print(f"  JS-max[{ref_name}]:          t={t_js:.4f}  acc={m['acc']:.4f} mf={m['mf']:.4f}  {tag}")
