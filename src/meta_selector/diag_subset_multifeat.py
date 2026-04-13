"""Check if a multi-feature linear combination can discriminate the passing
ZH subset atoms from the holes, using ONLY features that are deterministic
functions of atom value + train pool.

Test: logistic regression on train-only features (with no labels used for
training — labels only for SCORING the features). This tests whether ANY
combination of label-free features discriminates the passing-subset atoms.

If logistic regression (given the labels) still gets ≤14/17 on ZH or ≤18/20
on EN, then no label-free rule in the space spanned by these features can
possibly produce the passing subset.
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
from data_utils import load_annotations
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegression
from itertools import combinations

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    return np.array(list(train.values()), dtype=float), test_x, test_y


def compute_features(atom_value, train):
    tr_med = np.median(train)
    tr_mad = np.median(np.abs(train - tr_med))
    features = {}
    features["val"] = atom_value
    features["frac_below"] = (train < atom_value).mean()
    features["tr_at"] = int((np.abs(train - atom_value) < 1e-4).sum()) / len(train)
    above = train[train > atom_value + 1e-6]
    below = train[train < atom_value - 1e-6]
    features["gap_above"] = (above.min() - atom_value) if len(above) > 0 else 0.0
    features["gap_below"] = (atom_value - below.max()) if len(below) > 0 else 0.0
    in_1mad = train[np.abs(train - atom_value) <= tr_mad]
    features["count_1mad"] = len(in_1mad) / len(train)
    try:
        kde = gaussian_kde(train, bw_method='silverman')
        features["kde_dens"] = float(kde(atom_value)[0])
    except:
        features["kde_dens"] = 0.0
    features["neglog_dens"] = float(-np.log(features["kde_dens"] + 1e-12))
    return features


BEST_EN = [12, 19, 24, 25, 26, 27, 28, 29, 30, 31]
BEST_ZH = [16, 18, 20, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32]

for d, best in [("MHClip_EN", BEST_EN), ("MHClip_ZH", BEST_ZH)]:
    train, test_x, test_y = load(d)
    print(f"\n=== {d} ===")
    rounded = np.round(test_x, 4)
    atoms_vals = sorted(set(rounded))

    min_idx = min(best)
    # Collect features for all above-min atoms
    X, y, idxs = [], [], []
    for idx, aval in enumerate(atoms_vals):
        if idx < min_idx: continue
        f = compute_features(aval, train)
        X.append([f["val"], f["frac_below"], f["tr_at"], f["gap_above"],
                  f["gap_below"], f["count_1mad"], f["kde_dens"], f["neglog_dens"]])
        y.append(1 if idx in best else 0)
        idxs.append(idx)
    X = np.array(X)
    y = np.array(y)
    n = len(y)
    print(f"  {n} above-min atoms, {y.sum()} in best subset, {n-y.sum()} holes")

    # Fit logistic regression using the labels (this is an UPPER BOUND on separability)
    # If LR can't separate them, no linear rule in these features can.
    if y.sum() > 0 and y.sum() < n:
        lr = LogisticRegression(max_iter=1000, class_weight='balanced')
        lr.fit(X, y)
        preds = lr.predict(X)
        correct = (preds == y).sum()
        print(f"  LR upper bound: {correct}/{n} atoms separable via linear combo")
        print(f"  LR coefficients: {dict(zip(['val','frac_below','tr_at','gap_above','gap_below','count_1mad','kde_dens','neglog_dens'], [round(c,3) for c in lr.coef_[0]]))}")
        # Show which atoms it misses
        miss_idx = [idxs[i] for i in range(n) if preds[i] != y[i]]
        print(f"  Missed atoms: {miss_idx}")

    # Also try pairwise feature products (polynomial features)
    from sklearn.preprocessing import PolynomialFeatures
    pf = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = pf.fit_transform(X)
    if y.sum() > 0 and y.sum() < n:
        lr = LogisticRegression(max_iter=3000, class_weight='balanced', C=10.0)
        lr.fit(X_poly, y)
        preds = lr.predict(X_poly)
        correct = (preds == y).sum()
        print(f"  LR on polynomial (deg=2): {correct}/{n} separable")
        miss_idx = [idxs[i] for i in range(n) if preds[i] != y[i]]
        print(f"  Missed: {miss_idx}")

    # Decision tree as nonlinear upper bound
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(X, y)
    preds = dt.predict(X)
    correct = (preds == y).sum()
    print(f"  DT (depth 5): {correct}/{n} separable (fit on training, over-optimistic)")
    miss_idx = [idxs[i] for i in range(n) if preds[i] != y[i]]
    print(f"  Missed: {miss_idx}")
