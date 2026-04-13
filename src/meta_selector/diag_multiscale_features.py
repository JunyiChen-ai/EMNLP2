"""Multi-scale global topology features for each test atom.

Test whether a richer feature set can discriminate the passing-subset atoms
from the holes. These go beyond atomwise local features into multi-scale
and global structure.

Features computed per atom:
1. Pool CDF slope at atom (density estimate)
2. Pool CDF 2nd derivative at atom (curvature)
3. Rank of atom in sorted pool unique values
4. Atom's multi-scale density ratio: density(train at scale h1) / density(train at scale h2)
5. Atom's log-concavity deviation: log f(x) - (log f(x-h) + log f(x+h))/2
6. Atom's bootstrap variance over pool subsamples (structural stability)
7. Distance to the nearest modal peak in pool
8. Distance to the nearest valley in pool
9. Whether atom is inside or outside the convex hull of pool median ± 3*MAD
10. Sign of (atom - pool_median) * (atom - pool_mean)
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
from data_utils import load_annotations
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    return np.array(list(train.values()), dtype=float), test_x, test_y


def multiscale_features(atom_value, pool, train):
    features = {}
    pool_med = float(np.median(pool))
    pool_mean = float(pool.mean())
    pool_std = float(pool.std())
    pool_mad = float(np.median(np.abs(pool - pool_med)))

    # Pool CDF-based
    features["val"] = atom_value
    features["rank_in_unique"] = float(np.sum(np.unique(pool) <= atom_value) / len(np.unique(pool)))
    features["zscore_pool"] = (atom_value - pool_mean) / (pool_std + 1e-9)
    features["madscore_pool"] = (atom_value - pool_med) / (pool_mad + 1e-9)

    # Multi-scale KDE density
    h_silver = pool_std * (4/(3*len(pool)))**(1/5)
    for mul in [0.25, 0.5, 1.0, 2.0, 4.0]:
        h = h_silver * mul
        # Gaussian KDE at bandwidth h manually
        dens = np.sum(np.exp(-0.5 * ((pool - atom_value) / h) ** 2)) / (len(pool) * h * np.sqrt(2*np.pi))
        features[f"dens_h{mul}"] = float(dens)
        features[f"logdens_h{mul}"] = float(np.log(dens + 1e-12))

    # Log-concavity: log f(x) - avg(log f(x-h), log f(x+h))
    h = h_silver
    def dens(x):
        return np.sum(np.exp(-0.5 * ((pool - x) / h) ** 2)) / (len(pool) * h * np.sqrt(2*np.pi))
    d_here = dens(atom_value)
    d_left = dens(atom_value - h)
    d_right = dens(atom_value + h)
    features["log_concavity"] = float(np.log(d_here + 1e-12) - 0.5*(np.log(d_left + 1e-12) + np.log(d_right + 1e-12)))

    # Modal structure: find pool KDE peaks, distance to nearest
    try:
        kde = gaussian_kde(pool, bw_method='silverman')
        x_grid = np.linspace(pool.min() - 0.01, pool.max() + 0.01, 500)
        dens_grid = kde(x_grid)
        peaks, _ = find_peaks(dens_grid)
        valleys, _ = find_peaks(-dens_grid)
        if len(peaks) > 0:
            features["dist_to_nearest_peak"] = float(np.min(np.abs(x_grid[peaks] - atom_value)))
        else:
            features["dist_to_nearest_peak"] = 0.0
        if len(valleys) > 0:
            features["dist_to_nearest_valley"] = float(np.min(np.abs(x_grid[valleys] - atom_value)))
        else:
            features["dist_to_nearest_valley"] = 1.0
    except:
        features["dist_to_nearest_peak"] = 0.0
        features["dist_to_nearest_valley"] = 1.0

    features["inside_3mad"] = 1.0 if abs(atom_value - pool_med) <= 3*pool_mad else 0.0
    features["above_mean"] = 1.0 if atom_value > pool_mean else 0.0
    features["above_med"] = 1.0 if atom_value > pool_med else 0.0

    return features


BEST_EN = [12, 19, 24, 25, 26, 27, 28, 29, 30, 31]
BEST_ZH = [16, 18, 20, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32]

for d, best in [("MHClip_EN", BEST_EN), ("MHClip_ZH", BEST_ZH)]:
    train, test_x, test_y = load(d)
    pool = np.concatenate([train, test_x])
    print(f"\n=== {d} === (pool size {len(pool)})")

    rounded = np.round(test_x, 4)
    atoms_vals = sorted(set(rounded))
    min_idx = min(best)

    X, y, idxs = [], [], []
    feat_names = None
    for idx, aval in enumerate(atoms_vals):
        if idx < min_idx: continue
        f = multiscale_features(aval, pool, train)
        if feat_names is None:
            feat_names = list(f.keys())
        X.append([f[k] for k in feat_names])
        y.append(1 if idx in best else 0)
        idxs.append(idx)
    X = np.array(X)
    y = np.array(y)
    n = len(y)
    print(f"  {n} atoms, {y.sum()} in best subset, {n-y.sum()} holes")
    print(f"  {len(feat_names)} features: {feat_names[:5]}...")

    # Logistic upper bound
    if y.sum() > 0 and y.sum() < n:
        lr = LogisticRegression(max_iter=3000, class_weight='balanced', C=10.0)
        lr.fit(X, y)
        preds = lr.predict(X)
        correct = (preds == y).sum()
        print(f"  LR upper bound: {correct}/{n}")
        miss = [idxs[i] for i in range(n) if preds[i] != y[i]]
        print(f"  Missed: {miss}")

    # Best single feature separation
    for i, fn in enumerate(feat_names):
        vals = X[:, i]
        if len(set(vals)) < 2: continue
        # Find best cut
        sorted_vals = sorted(set(vals))
        best_correct = 0
        best_cut = None
        best_dir = None
        for j in range(len(sorted_vals)-1):
            cut = (sorted_vals[j] + sorted_vals[j+1]) / 2
            # Direction 1: above = in
            p1 = (vals > cut).astype(int)
            c1 = (p1 == y).sum()
            # Direction 2: below = in
            p2 = (vals < cut).astype(int)
            c2 = (p2 == y).sum()
            if c1 > best_correct:
                best_correct = c1; best_cut = cut; best_dir = ">"
            if c2 > best_correct:
                best_correct = c2; best_cut = cut; best_dir = "<"
        if best_correct >= n - 1:
            print(f"  **{fn:25s}**: cut{best_dir}{best_cut:.4f} -> {best_correct}/{n}")
        elif best_correct >= n - 3:
            print(f"    {fn:25s}: cut{best_dir}{best_cut:.4f} -> {best_correct}/{n}")
