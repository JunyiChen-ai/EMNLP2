"""Isolation Forest score on train-pool (Liu, Ting, Zhou 2008).

Published method (sklearn default params: n_estimators=100, contamination='auto',
max_samples='auto'). The iForest anomaly score is non-monotone in raw value
because it depends on recursive random partitioning of the train pool — a
score value near a dense train cluster gets MORE paths than a score value in
a sparse gap. This is genuinely non-monotone in raw score at test time.

Rationale for hateful video: hateful videos are anomalous relative to the
train-pool's dominant non-hateful core. Isolation Forest is designed exactly
for this setting and is a standard published anomaly detector with no
hate-specific tuning. The label-free signal is: fit iForest on train-pool
scores alone, compute isolation score for each test score, then threshold
the isolation score.
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, metrics, load_scores_file, build_arrays
from data_utils import load_annotations
from sklearn.ensemble import IsolationForest

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    return np.array(list(train.values()), dtype=float), test_x, test_y


for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    pool = np.concatenate([train, test_x])
    acc_b, mf_b = BASE[d]
    print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}")

    for name, fit_data in [("train", train), ("pool", pool)]:
        iforest = IsolationForest(n_estimators=100, random_state=42,
                                  contamination='auto', max_samples='auto')
        iforest.fit(fit_data.reshape(-1, 1))
        # Higher = more anomalous = more likely positive
        # sklearn returns score_samples where HIGHER = more normal
        # We want: higher anomaly -> larger positive score
        iso_scores = -iforest.score_samples(test_x.reshape(-1, 1))

        # Apply Otsu and GMM on iso_scores
        t_otsu = otsu_threshold(iso_scores)
        m = metrics(iso_scores, test_y, t_otsu)
        tag = "STRICT" if m["acc"] > acc_b and m["mf"] > mf_b else ""
        print(f"  iForest[{name}] + Otsu: acc={m['acc']:.4f}  mf={m['mf']:.4f}  {tag}")

        t_gmm = gmm_threshold(iso_scores)
        m = metrics(iso_scores, test_y, t_gmm)
        tag = "STRICT" if m["acc"] > acc_b and m["mf"] > mf_b else ""
        print(f"  iForest[{name}] + GMM:  acc={m['acc']:.4f}  mf={m['mf']:.4f}  {tag}")

        # Also check what fraction of atoms flip order
        order_orig = np.argsort(test_x)
        order_iso = np.argsort(iso_scores)
        flipped = np.sum(order_orig != order_iso)
        print(f"    order changes: {flipped}/{len(test_x)}")
