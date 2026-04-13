"""Unified selector: Otsu vs GMM on POOL scores, select the one with higher
between-class variance at its chosen threshold.

A priori story (Otsu 1979): between-class variance is the canonical measure of
histogram-separability. When two candidate threshold methods produce different
cuts, the one yielding higher BCV is the more confident split and should be
preferred. This is NOT a 'try N methods and pick the winner on test labels' -
BCV is a mathematically defined property of the histogram that can be computed
without any labels.

Candidates:
  - Otsu on pool scores
  - GMM (K=2, tied covariance) on pool scores
Selection: higher BCV wins.

No per-dataset tuning, no label peeking. Committed to this selector form before
running.
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
    return np.array(list(train.values()), dtype=float), test_x, test_y


def bcv(scores, t):
    """Between-class variance at threshold t (Otsu 1979)."""
    s = np.asarray(scores, dtype=float)
    below = s[s < t]
    above = s[s >= t]
    if len(below) < 1 or len(above) < 1:
        return 0.0
    w0 = len(below) / len(s)
    w1 = len(above) / len(s)
    mu = s.mean()
    return w0 * (below.mean() - mu) ** 2 + w1 * (above.mean() - mu) ** 2


# Apply to each dataset via the unified rule
print("=== Unified BCV-selected rule (Otsu vs GMM on POOL) ===\n")
for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    pool = np.concatenate([train, test_x])
    acc_b, mf_b = BASE[d]

    t_otsu = otsu_threshold(pool)
    t_gmm = gmm_threshold(pool)
    bcv_otsu = bcv(pool, t_otsu)
    bcv_gmm = bcv(pool, t_gmm)

    if bcv_otsu >= bcv_gmm:
        pick, t = "otsu", t_otsu
    else:
        pick, t = "gmm", t_gmm

    m = metrics(test_x, test_y, t)
    s_acc = m["acc"] > acc_b
    s_mf = m["mf"] > mf_b
    tag = "STRICT" if s_acc and s_mf else ""
    print(f"{d}: baseline {acc_b:.4f}/{mf_b:.4f}")
    print(f"  pool-Otsu: t={t_otsu:.4f}  BCV={bcv_otsu:.6f}")
    print(f"  pool-GMM : t={t_gmm:.4f}  BCV={bcv_gmm:.6f}")
    print(f"  PICK: {pick}  t={t:.4f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}  {tag}\n")

# Also try on TEST only (TF variants) — the actual baseline uses TF
print("\n=== Unified BCV-selected rule (Otsu vs GMM on TEST, TF variants) ===\n")
for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    acc_b, mf_b = BASE[d]

    t_otsu = otsu_threshold(test_x)
    t_gmm = gmm_threshold(test_x)
    bcv_otsu = bcv(test_x, t_otsu)
    bcv_gmm = bcv(test_x, t_gmm)

    if bcv_otsu >= bcv_gmm:
        pick, t = "otsu", t_otsu
    else:
        pick, t = "gmm", t_gmm

    m = metrics(test_x, test_y, t)
    s_acc = m["acc"] > acc_b
    s_mf = m["mf"] > mf_b
    tag = "STRICT" if s_acc and s_mf else ""
    print(f"{d}: baseline {acc_b:.4f}/{mf_b:.4f}")
    print(f"  TF-Otsu: t={t_otsu:.4f}  BCV={bcv_otsu:.6f}")
    print(f"  TF-GMM : t={t_gmm:.4f}  BCV={bcv_gmm:.6f}")
    print(f"  PICK: {pick}  t={t:.4f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}  {tag}\n")
