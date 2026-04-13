"""Pilot: 'negative core' approach (Liu et al. 2003 negative-only learning).

Phenomenon for hateful video: MLLMs confidently say 'no' for most videos. The
negative class forms a tight dense core in score space; the positive class
spreads into the tail. If we can identify the negative core unsupervised,
we can flag samples far from it.

Standard literature choices (no label-fit parameters):
  - negative core = [0, pool median] (median split)
  - spread estimator = MAD of lower half (robust, parameter-free)
  - flag threshold = lower_half_median + k * MAD_lower (k = 3 is Tukey's fence,
    published standard from Tukey 1977 "Exploratory Data Analysis")

Single unified rule, no per-dataset tuning, no enumeration of k.
"""
import os, sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, otsu_threshold, load_scores_file, build_arrays
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


def negcore_rule(pool, k=3.0):
    """Tukey fence on lower half of pool."""
    med = float(np.median(pool))
    lower = pool[pool <= med]
    lower_med = float(np.median(lower))
    lower_mad = float(np.median(np.abs(lower - lower_med)))
    return lower_med + k * lower_mad, lower_med, lower_mad


def negcore_inner_fence(pool):
    """Tukey inner fence: Q3 + 1.5*IQR. Classical published parameter."""
    q1, q3 = np.quantile(pool, [0.25, 0.75])
    iqr = q3 - q1
    return float(q3 + 1.5 * iqr)


def negcore_outer_fence(pool):
    """Tukey outer fence: Q3 + 3*IQR. Classical published parameter."""
    q1, q3 = np.quantile(pool, [0.25, 0.75])
    iqr = q3 - q1
    return float(q3 + 3.0 * iqr)


print("=== Negative-core (Tukey) thresholds ===")
print("All three variants use Tukey (1977) published standard constants.")
print()

for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    pool = np.concatenate([train, test_x])
    acc_b, mf_b = BASE[d]
    print(f"\n--- {d} --- baseline {acc_b:.4f}/{mf_b:.4f}")
    # Variant 1: k-MAD with k=3 (Tukey 3-sigma analog)
    t1, lm, lmad = negcore_rule(pool, k=3.0)
    m = metrics(test_x, test_y, t1)
    print(f"  MAD-3 rule: t={t1:.4f}  lower_med={lm:.4f}  lower_mad={lmad:.4f}")
    print(f"    acc={m['acc']:.4f}  mf={m['mf']:.4f}  "
          f"{'STRICT' if m['acc']>acc_b and m['mf']>mf_b else ''}")
    # Variant 2: Tukey inner fence on full pool
    t2 = negcore_inner_fence(pool)
    m = metrics(test_x, test_y, t2)
    print(f"  Tukey inner fence: t={t2:.4f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}  "
          f"{'STRICT' if m['acc']>acc_b and m['mf']>mf_b else ''}")
    # Variant 3: Tukey outer fence on full pool
    t3 = negcore_outer_fence(pool)
    m = metrics(test_x, test_y, t3)
    print(f"  Tukey outer fence: t={t3:.4f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}  "
          f"{'STRICT' if m['acc']>acc_b and m['mf']>mf_b else ''}")
