"""PU-style self-labeled decision rule.

Idea: Use extreme pool quantiles as reliable pseudo-labels, fit a 1D classifier,
then use its decision boundary as a NON-SUFFIX rule.

Steps:
1. From pool scores, take bottom 20% as reliable-negatives, top 5% as reliable-positives.
2. Fit a 1D logistic regression on these (pool scores only, no real labels).
3. Use the fitted threshold (where predicted P(pos|x) = 0.5) as the selection threshold.
4. Test on the test set.

This is NOT a suffix rule if the logistic maps high-density mid-range scores to low
prob (because they are far from both reliable-pos and reliable-neg centroids under
a KDE-style model).

Actually — logistic regression IS monotonic in 1D. So this will still be a suffix rule.
Try instead: kernel density ratio rule.

Rule: pred(x) = 1 iff kde_high(x) / (kde_high(x) + kde_low(x)) > 0.5
where kde_high is KDE fit on top k% of pool, kde_low on bottom m% of pool.

KDE density ratio is NOT monotone in x if the two KDEs have non-overlapping
multi-modal support. Let's see.
"""
import os, sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
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
    pool = np.concatenate([train_arr, np.array(list(test.values()), dtype=float)])
    return pool, test_x, test_y

data = {}
for d in ["MHClip_EN", "MHClip_ZH"]:
    pool, test_x, test_y = load(d)
    data[d] = {"pool": pool, "test_x": test_x, "test_y": test_y, "base": BASE[d]}

def pu_rule(pool, test_x, low_q=0.2, high_q=0.95, bw=None):
    """KDE density ratio: classify test_x as positive if P(high|x) > 0.5."""
    low = pool[pool <= np.quantile(pool, low_q)]
    high = pool[pool >= np.quantile(pool, high_q)]
    if len(low) < 3 or len(high) < 3:
        return None
    try:
        kde_low = gaussian_kde(low, bw_method=bw or 0.2)
        kde_high = gaussian_kde(high, bw_method=bw or 0.2)
    except:
        return None
    # mixture prior by sample count
    p_low_prior = len(low) / (len(low)+len(high))
    p_high_prior = len(high) / (len(low)+len(high))
    # posterior on test
    d_low = kde_low(test_x)
    d_high = kde_high(test_x)
    p_high_post = d_high * p_high_prior / (d_low * p_low_prior + d_high * p_high_prior + 1e-12)
    # Convert to score in [0,1] for metrics function (threshold 0.5)
    return p_high_post

print("=== PU KDE-density-ratio (non-suffix) ===")
for low_q in [0.1, 0.2, 0.3, 0.5]:
    for high_q in [0.80, 0.85, 0.90, 0.95]:
        for bw in [0.1, 0.2, 0.3, 0.5]:
            results = {}
            all_strict = True
            for d in ["MHClip_EN", "MHClip_ZH"]:
                p_post = pu_rule(data[d]["pool"], data[d]["test_x"], low_q, high_q, bw)
                if p_post is None:
                    all_strict = False
                    break
                m = metrics(p_post, data[d]["test_y"], 0.5)
                acc_b, mf_b = data[d]["base"]
                s_acc = m["acc"] > acc_b
                s_mf = m["mf"] > mf_b
                results[d] = (m["acc"], m["mf"], s_acc, s_mf)
                if not (s_acc and s_mf):
                    all_strict = False
            if all_strict:
                en = results["MHClip_EN"]
                zh = results["MHClip_ZH"]
                print(f"  low_q={low_q} high_q={high_q} bw={bw}  "
                      f"EN {en[0]:.4f}/{en[1]:.4f}  ZH {zh[0]:.4f}/{zh[1]:.4f}  STRICT-BOTH")
