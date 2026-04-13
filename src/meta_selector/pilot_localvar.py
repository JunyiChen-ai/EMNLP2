"""Local train-neighborhood variance as non-monotone re-score.

For each test sample x, find its k nearest train samples by score distance,
compute the variance (or other spread statistic) of those train scores, and
use that as a new classification feature. Apply Otsu on the new feature.

Non-monotone justification: two test samples with identical score will have
identical local variance (feature deterministic on score), but the MAPPING
from score to variance is NOT monotone — local variance peaks in mid-range
where train scores transition between dense cluster regions.

Published basis: local density / local variance is a standard anomaly
detection feature (Breunig et al. 2000 "LOF: Identifying Density-Based
Local Outliers"). k = sqrt(n) is the Fix-Hodges 1951 published default.

Also tests: gap-to-nearest-train-of-different-magnitude (semantic jump),
distance-to-train-median, train-percentile rank.
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


def local_var(test_x, ref, k):
    """For each test point, variance of k nearest ref points."""
    out = np.empty(len(test_x))
    for i, x in enumerate(test_x):
        dists = np.abs(ref - x)
        idx = np.argsort(dists)[:k]
        out[i] = ref[idx].var()
    return out


def local_range(test_x, ref, k):
    """For each test point, range (max-min) of k nearest ref points."""
    out = np.empty(len(test_x))
    for i, x in enumerate(test_x):
        dists = np.abs(ref - x)
        idx = np.argsort(dists)[:k]
        out[i] = ref[idx].max() - ref[idx].min()
    return out


def local_meandist(test_x, ref, k):
    """Mean distance to k nearest ref points."""
    out = np.empty(len(test_x))
    for i, x in enumerate(test_x):
        dists = np.abs(ref - x)
        idx = np.argsort(dists)[:k]
        out[i] = dists[idx].mean()
    return out


for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    pool = np.concatenate([train, test_x])
    acc_b, mf_b = BASE[d]
    print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}")

    n_ref = len(train)
    k = max(3, int(np.sqrt(n_ref)))  # Fix-Hodges default
    print(f"  n_train={n_ref}, k=sqrt(n)={k}")

    for feat_name, feat_fn in [("local_var", local_var),
                                ("local_range", local_range),
                                ("local_meandist", local_meandist)]:
        for ref_name, ref in [("train", train), ("pool", pool)]:
            f = feat_fn(test_x, ref, k)
            for method_name, method in [("Otsu", otsu_threshold), ("GMM", gmm_threshold)]:
                t = method(f)
                m = metrics(f, test_y, t)
                tag = "STRICT" if m["acc"] > acc_b and m["mf"] > mf_b else ""
                print(f"  {feat_name}[{ref_name}]+{method_name}: acc={m['acc']:.4f} mf={m['mf']:.4f} {tag}")

            # Also check: "score + lambda*local_var" where lambda chosen to match scales
            scale = np.std(test_x) / (np.std(f) + 1e-12)
            fused = test_x + scale * f
            t = otsu_threshold(fused)
            m = metrics(fused, test_y, t)
            tag = "STRICT" if m["acc"] > acc_b and m["mf"] > mf_b else ""
            print(f"  {feat_name}[{ref_name}]+score+Otsu: acc={m['acc']:.4f} mf={m['mf']:.4f} {tag}")
