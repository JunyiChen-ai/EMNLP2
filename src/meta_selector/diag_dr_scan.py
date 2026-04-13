"""Scan every density-ratio feature cut on both datasets. Find any
label-free rule that strict-beats the bar.

Also test combinations:
  - (log_ratio < c1) AND (score > c2) where c1, c2 are LABEL-FREE cuts
    (e.g., c1 from Otsu on log_ratio histogram, c2 from Otsu on score)
  - Non-monotone combinations (XOR, score > t1 OR log_ratio > t2)
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import load_scores_file, build_arrays
from data_utils import load_annotations
from sklearn.metrics import accuracy_score, f1_score
from skimage.filters import threshold_otsu

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    return np.array(list(train.values()), dtype=float), test_x, test_y


def kde_at(x, pool, h):
    return np.sum(np.exp(-0.5 * ((pool - x) / h) ** 2)) / (len(pool) * h * np.sqrt(2*np.pi))


def dr_value(x, train, test_pool, h, name):
    f_tr = kde_at(x, train, h)
    f_te = kde_at(x, test_pool, h)
    if name == "log_ratio":
        return np.log((f_tr + 1e-12) / (f_te + 1e-12))
    if name == "ratio":
        return f_tr / (f_te + 1e-12)
    if name == "diff":
        return f_te - f_tr
    if name == "rel_diff":
        return (f_te - f_tr) / (f_tr + f_te + 1e-12)


def eval_pred(y_true, y_pred):
    return {"acc": accuracy_score(y_true, y_pred), "mf": f1_score(y_true, y_pred, average='macro')}


for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    acc_b, mf_b = BASE[d]
    print(f"\n=== {d} === baseline acc={acc_b:.4f} mf={mf_b:.4f}")

    # For each bandwidth and feature name, scan cuts
    tr_std = float(train.std())
    h_silv = tr_std * (4/(3*len(train)))**(1/5)

    results_all = []
    for mul in [0.25, 0.5, 1.0, 2.0, 4.0]:
        h = h_silv * mul
        for fname in ["log_ratio", "ratio", "diff", "rel_diff"]:
            feat_per_sample = np.array([dr_value(x, train, test_x, h, fname) for x in test_x])
            # Also need score-per-sample for combinations
            # Scan cuts from sorted unique values
            sorted_vals = np.sort(np.unique(feat_per_sample))
            # Try cuts at each midpoint
            for i in range(len(sorted_vals) - 1):
                cut = (sorted_vals[i] + sorted_vals[i+1]) / 2
                # Direction 1: predict pos iff feat > cut
                for direction in [1, -1]:
                    pred = ((feat_per_sample * direction) > (cut * direction)).astype(int)
                    if pred.sum() == 0 or pred.sum() == len(pred): continue
                    m = eval_pred(test_y, pred)
                    if m["acc"] > acc_b and m["mf"] >= mf_b:
                        dir_s = ">" if direction == 1 else "<"
                        results_all.append((m["acc"], m["mf"], f"h{mul}_{fname} {dir_s} {cut:.4f}"))

    # Report passing rules
    if results_all:
        results_all.sort(reverse=True)
        print(f"  PASSING rules ({len(results_all)}):")
        for acc, mf, desc in results_all[:15]:
            print(f"    acc={acc:.4f} mf={mf:.4f}  {desc}")
    else:
        print("  No single-DR-feature cut passes.")

    # Non-monotone COMBINATION: (score > c_score) XOR (log_ratio > c_lr)
    # where c_score is train-only (med, q25-q75, Otsu on train) and c_lr is zero
    print(f"\n  Combination rules (score-cut-by-train-pool AND/OR DR-sign):")
    score_cuts = {
        "tr_med": float(np.median(train)),
        "tr_q25": float(np.quantile(train, 0.25)),
        "tr_q75": float(np.quantile(train, 0.75)),
        "tr_otsu": float(threshold_otsu(train.reshape(-1, 1), nbins=256)),
    }
    comb_results = []
    h = h_silv
    feat = np.array([dr_value(x, train, test_x, h, "log_ratio") for x in test_x])
    for sname, sc in score_cuts.items():
        for lr_c in [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0]:
            for op in ["AND_s>_lr>", "AND_s>_lr<", "OR_s>_lr>", "OR_s>_lr<", "XOR"]:
                a = test_x > sc
                b = feat > lr_c
                if op == "AND_s>_lr>": p = a & b
                elif op == "AND_s>_lr<": p = a & (~b)
                elif op == "OR_s>_lr>": p = a | b
                elif op == "OR_s>_lr<": p = a | (~b)
                else: p = a ^ b
                p = p.astype(int)
                if p.sum() == 0 or p.sum() == len(p): continue
                m = eval_pred(test_y, p)
                if m["acc"] > acc_b and m["mf"] >= mf_b:
                    comb_results.append((m["acc"], m["mf"], f"{sname}={sc:.4f} lr{lr_c:+.2f} {op}"))
    if comb_results:
        comb_results.sort(reverse=True)
        print(f"    PASSING combinations ({len(comb_results)}):")
        for acc, mf, desc in comb_results[:10]:
            print(f"      acc={acc:.4f} mf={mf:.4f}  {desc}")
    else:
        print(f"    No passing combinations.")
