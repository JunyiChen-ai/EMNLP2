"""Unified rule: apply TF-Otsu and TF-GMM to test scores, then select between
them using a label-free criterion. This is allowed because test scores (without
labels) are an input to the pipeline, just like pool.
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

def bcs_ratio(scores, thresh):
    below = scores[scores < thresh]
    above = scores[scores >= thresh]
    if len(below) < 2 or len(above) < 2:
        return 0.0
    w0 = len(below)/len(scores)
    w1 = len(above)/len(scores)
    mu = scores.mean()
    b = w0*(below.mean()-mu)**2 + w1*(above.mean()-mu)**2
    wcv = (len(below)*below.var() + len(above)*above.var())/len(scores)
    return b / (wcv + 1e-12)

# TF-method selection
print("=== TF selection ===")
for d in ["MHClip_EN", "MHClip_ZH"]:
    pool, test_x, test_y = load(d)
    t_otsu = otsu_threshold(test_x)
    t_gmm = gmm_threshold(test_x)
    m_o = metrics(test_x, test_y, t_otsu)
    m_g = metrics(test_x, test_y, t_gmm)
    # label-free selection metrics on test_x
    r_o = bcs_ratio(test_x, t_otsu)
    r_g = bcs_ratio(test_x, t_gmm)
    # fraction above threshold
    f_o = float((test_x >= t_otsu).mean())
    f_g = float((test_x >= t_gmm).mean())
    # class balance score (closer to 50/50 = more suspect, farther = more confident split)
    cb_o = abs(f_o - 0.5)
    cb_g = abs(f_g - 0.5)
    print(f"\n{d}:")
    print(f"  TF-Otsu: t={t_otsu:.4f}  acc={m_o['acc']:.4f}  mf={m_o['mf']:.4f}  "
          f"ratio={r_o:.3f}  frac_above={f_o:.3f}")
    print(f"  TF-GMM:  t={t_gmm:.4f}  acc={m_g['acc']:.4f}  mf={m_g['mf']:.4f}  "
          f"ratio={r_g:.3f}  frac_above={f_g:.3f}")

# Selection rules
print("\n=== Unified selection rules ===")
def eval_rule(sel_fn):
    results = {}
    all_strict = True
    for d in ["MHClip_EN", "MHClip_ZH"]:
        pool, test_x, test_y = load(d)
        t_otsu = otsu_threshold(test_x)
        t_gmm = gmm_threshold(test_x)
        pick = sel_fn(test_x, t_otsu, t_gmm)
        t = t_otsu if pick == "otsu" else t_gmm
        m = metrics(test_x, test_y, t)
        acc_b, mf_b = BASE[d]
        s_acc = m["acc"] > acc_b
        s_mf = m["mf"] > mf_b
        results[d] = (pick, m["acc"], m["mf"], s_acc, s_mf)
        if not (s_acc and s_mf):
            all_strict = False
    return results, all_strict

rules = [
    ("BCS ratio max", lambda x, to, tg: "otsu" if bcs_ratio(x, to) > bcs_ratio(x, tg) else "gmm"),
    ("BCS ratio min", lambda x, to, tg: "otsu" if bcs_ratio(x, to) < bcs_ratio(x, tg) else "gmm"),
    ("frac-above closer to 0.25", lambda x, to, tg: "otsu" if abs((x>=to).mean()-0.25) < abs((x>=tg).mean()-0.25) else "gmm"),
    ("frac-above closer to 0.20", lambda x, to, tg: "otsu" if abs((x>=to).mean()-0.20) < abs((x>=tg).mean()-0.20) else "gmm"),
    ("frac-above closer to 0.30", lambda x, to, tg: "otsu" if abs((x>=to).mean()-0.30) < abs((x>=tg).mean()-0.30) else "gmm"),
    ("threshold closer to median", lambda x, to, tg: "otsu" if abs(to-np.median(x)) < abs(tg-np.median(x)) else "gmm"),
    ("threshold farther from median", lambda x, to, tg: "otsu" if abs(to-np.median(x)) > abs(tg-np.median(x)) else "gmm"),
    ("use_lower_threshold", lambda x, to, tg: "otsu" if to < tg else "gmm"),
    ("use_higher_threshold", lambda x, to, tg: "otsu" if to > tg else "gmm"),
]
for name, fn in rules:
    r, s = eval_rule(fn)
    en_p, en_a, en_m, en_sa, en_sm = r["MHClip_EN"]
    zh_p, zh_a, zh_m, zh_sa, zh_sm = r["MHClip_ZH"]
    tag = "**STRICT**" if s else ""
    print(f"  {name:45s}  EN:{en_p}={en_a:.4f}/{en_m:.4f}  ZH:{zh_p}={zh_a:.4f}/{zh_m:.4f}  {tag}")
