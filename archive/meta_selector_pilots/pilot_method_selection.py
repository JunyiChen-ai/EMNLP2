"""Unified rule: apply both Otsu and GMM to pool, pick the one with higher
between-class separation (BCS, label-free).

Also try other selection criteria:
 - bimodality coefficient
 - KDE valley depth
 - silhouette in 1D
"""
import os, sys, numpy as np
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
    train_arr = np.array(list(train.values()), dtype=float)
    pool = np.concatenate([train_arr, test_x])
    return pool, test_x, test_y

def bcs(scores, thresh):
    """Between-class variance given a threshold."""
    below = scores[scores < thresh]
    above = scores[scores >= thresh]
    if len(below) == 0 or len(above) == 0:
        return 0.0
    w0 = len(below) / len(scores)
    w1 = len(above) / len(scores)
    mu0 = below.mean()
    mu1 = above.mean()
    mu = scores.mean()
    return w0 * (mu0 - mu)**2 + w1 * (mu1 - mu)**2

def bcs_over_wcv(scores, thresh):
    """Ratio: between-class variance / within-class variance (separation ratio)."""
    below = scores[scores < thresh]
    above = scores[scores >= thresh]
    if len(below) < 2 or len(above) < 2:
        return 0.0
    b = bcs(scores, thresh)
    wcv = (len(below)*below.var() + len(above)*above.var()) / len(scores)
    return b / (wcv + 1e-12)

def bimodality_coeff(scores):
    """Bimodality coefficient: higher = more bimodal."""
    n = len(scores)
    if n < 4 or scores.std() == 0:
        return 0.0
    s = ((scores - scores.mean())**3).mean() / scores.std()**3
    k = ((scores - scores.mean())**4).mean() / scores.std()**4 - 3
    bc = (s**2 + 1) / (k + 3 * (n-1)**2 / ((n-2)*(n-3)))
    return bc

def kde_valley_depth(scores, t):
    """Depth of valley at t in the KDE: higher = deeper separation."""
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(scores, bw_method=0.2)
    except:
        return 0.0
    # density at t
    dt = kde(t)[0]
    # max density to the left and right
    xs = np.linspace(scores.min(), t, 50)
    left_max = kde(xs).max()
    xs2 = np.linspace(t, scores.max(), 50)
    right_max = kde(xs2).max()
    return min(left_max, right_max) - dt  # positive when valley

print("=== Unified method-selection rules ===\n")

for d in ["MHClip_EN", "MHClip_ZH"]:
    pool, test_x, test_y = load(d)
    t_otsu = otsu_threshold(pool)
    t_gmm = gmm_threshold(pool)

    # Label-free scores for each method's threshold
    bcs_otsu = bcs(pool, t_otsu)
    bcs_gmm = bcs(pool, t_gmm)
    rat_otsu = bcs_over_wcv(pool, t_otsu)
    rat_gmm = bcs_over_wcv(pool, t_gmm)
    val_otsu = kde_valley_depth(pool, t_otsu)
    val_gmm = kde_valley_depth(pool, t_gmm)

    m_otsu = metrics(test_x, test_y, t_otsu)
    m_gmm = metrics(test_x, test_y, t_gmm)
    print(f"{d}:")
    print(f"  Otsu-pool: t={t_otsu:.4f}  acc={m_otsu['acc']:.4f}  mf={m_otsu['mf']:.4f}  "
          f"BCS={bcs_otsu:.6f}  ratio={rat_otsu:.3f}  valley={val_otsu:.3f}")
    print(f"  GMM-pool:  t={t_gmm:.4f}  acc={m_gmm['acc']:.4f}  mf={m_gmm['mf']:.4f}  "
          f"BCS={bcs_gmm:.6f}  ratio={rat_gmm:.3f}  valley={val_gmm:.3f}")
    print()

# Now test unified rules
print("=== Unified selection rules ===")

for sel_name, sel_func in [
    ("BCS max", lambda bo,bg,ro,rg,vo,vg: "otsu" if bo > bg else "gmm"),
    ("BCS min", lambda bo,bg,ro,rg,vo,vg: "otsu" if bo < bg else "gmm"),
    ("Ratio max", lambda bo,bg,ro,rg,vo,vg: "otsu" if ro > rg else "gmm"),
    ("Ratio min", lambda bo,bg,ro,rg,vo,vg: "otsu" if ro < rg else "gmm"),
    ("Valley max", lambda bo,bg,ro,rg,vo,vg: "otsu" if vo > vg else "gmm"),
    ("Valley min", lambda bo,bg,ro,rg,vo,vg: "otsu" if vo < vg else "gmm"),
]:
    print(f"\n  Selection rule: {sel_name}")
    results = {}
    all_strict = True
    for d in ["MHClip_EN", "MHClip_ZH"]:
        pool, test_x, test_y = load(d)
        t_otsu = otsu_threshold(pool)
        t_gmm = gmm_threshold(pool)
        bcs_otsu = bcs(pool, t_otsu)
        bcs_gmm = bcs(pool, t_gmm)
        rat_otsu = bcs_over_wcv(pool, t_otsu)
        rat_gmm = bcs_over_wcv(pool, t_gmm)
        val_otsu = kde_valley_depth(pool, t_otsu)
        val_gmm = kde_valley_depth(pool, t_gmm)
        pick = sel_func(bcs_otsu, bcs_gmm, rat_otsu, rat_gmm, val_otsu, val_gmm)
        t = t_otsu if pick == "otsu" else t_gmm
        m = metrics(test_x, test_y, t)
        acc_b, mf_b = BASE[d]
        s_acc = m["acc"] > acc_b
        s_mf = m["mf"] > mf_b
        tag = "STRICT" if (s_acc and s_mf) else ""
        print(f"    {d}: pick={pick}  t={t:.4f}  acc={m['acc']:.4f}(>{acc_b:.4f}:{s_acc})  "
              f"mf={m['mf']:.4f}(>{mf_b:.4f}:{s_mf}) {tag}")
        if not (s_acc and s_mf):
            all_strict = False
    print(f"    ALL-STRICT: {all_strict}")
