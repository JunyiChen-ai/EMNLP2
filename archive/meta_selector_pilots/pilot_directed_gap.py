"""Directed gap features: non-monotone-in-score atom-level scoring.

For each test atom x:
  gap_up(x)   = min(t in sorted(train_unique) : t > x) - x      (0 if none)
  gap_dn(x)   = x - max(t in sorted(train_unique) : t < x)      (0 if none)
  asymmetry(x) = gap_up(x) - gap_dn(x)
  log_ratio(x) = log((gap_up + eps) / (gap_dn + eps))

These are global-topology features that are NOT monotone in x — an atom
between dense train clusters has small gap_up and small gap_dn; an atom at a
tail boundary has large gap_up and small gap_dn. By construction, gap_up
jumps up whenever x crosses a train atom.

Tests: use each feature + Otsu/GMM as a threshold method; also combine with
raw score.

Bar: acc STRICT > baseline AND mf >= baseline.
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


def directed_gaps(test_x, ref):
    u = np.sort(np.unique(ref))
    gap_up = np.empty(len(test_x))
    gap_dn = np.empty(len(test_x))
    for i, x in enumerate(test_x):
        above = u[u > x]
        below = u[u < x]
        gap_up[i] = (above[0] - x) if len(above) > 0 else 0.0
        gap_dn[i] = (x - below[-1]) if len(below) > 0 else 0.0
    return gap_up, gap_dn


def check_bar(acc, mf, ab, mb):
    return (acc > ab) and (mf >= mb)


for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    pool = np.concatenate([train, test_x])
    acc_b, mf_b = BASE[d]
    print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}")

    for ref_name, ref in [("train", train), ("pool", pool)]:
        gu, gd = directed_gaps(test_x, ref)
        asym = gu - gd
        log_ratio = np.log((gu + 1e-6) / (gd + 1e-6))

        # Higher asymmetry means more gap above than below = isolated-on-right = likely positive tail
        # Lower asym = denser above = mid cluster = likely negative
        for feat_name, feat in [("gap_up", gu), ("gap_dn", gd), ("asym", asym), ("log_ratio", log_ratio),
                                ("neg_gap_up", -gu), ("neg_gap_dn", -gd), ("neg_asym", -asym)]:
            for method_name, method in [("Otsu", otsu_threshold), ("GMM", gmm_threshold)]:
                try:
                    t = method(feat)
                    m = metrics(feat, test_y, t)
                    tag = "PASS" if check_bar(m["acc"], m["mf"], acc_b, mf_b) else ""
                    print(f"  {feat_name:12s}[{ref_name}]+{method_name}: acc={m['acc']:.4f} mf={m['mf']:.4f} {tag}")
                except Exception as e:
                    pass
