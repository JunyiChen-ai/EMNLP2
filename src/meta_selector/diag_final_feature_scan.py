"""Final feature scan: combine ALL features I have and do an
exhaustive LR + decision-tree + random-forest probe on passing
subsets. Also test: for each feature, can it discriminate the
specific ZH passing atoms {16, 18, 22, 25} from holes {17, 19, 21, 24}
(the 4 non-tie discriminating decisions)?
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
from data_utils import load_annotations
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    return np.array(list(train.values()), dtype=float), test_x, test_y


def all_features(atom_value, pool, train):
    pool_med = float(np.median(pool))
    pool_mean = float(pool.mean())
    pool_std = float(pool.std())
    pool_mad = float(np.median(np.abs(pool - pool_med)))
    tr_med = float(np.median(train))
    tr_std = float(train.std())
    tr_mad = float(np.median(np.abs(train - tr_med)))
    features = {}
    features["val"] = atom_value
    features["frac_below_train"] = (train < atom_value).mean()
    features["frac_below_pool"] = (pool < atom_value).mean()
    features["z_pool"] = (atom_value - pool_mean) / (pool_std + 1e-9)
    features["z_train"] = (atom_value - train.mean()) / (tr_std + 1e-9)
    features["mad_pool"] = (atom_value - pool_med) / (pool_mad + 1e-9)
    features["mad_train"] = (atom_value - tr_med) / (tr_mad + 1e-9)
    # Directed gaps on both train and pool
    for name, ref in [("train", train), ("pool", pool)]:
        above = ref[ref > atom_value + 1e-6]
        below = ref[ref < atom_value - 1e-6]
        gu = (above.min() - atom_value) if len(above) > 0 else 0.0
        gd = (atom_value - below.max()) if len(below) > 0 else 0.0
        features[f"gap_up_{name}"] = gu
        features[f"gap_dn_{name}"] = gd
        features[f"log_asym_{name}"] = np.log((gu + 1e-6) / (gd + 1e-6))
        features[f"gap_sum_{name}"] = gu + gd
    # Counts within MAD windows
    features["cnt_1mad_tr"] = (np.abs(train - atom_value) <= tr_mad).mean()
    features["cnt_2mad_tr"] = (np.abs(train - atom_value) <= 2*tr_mad).mean()
    features["cnt_1mad_pool"] = (np.abs(pool - atom_value) <= pool_mad).mean()
    features["cnt_2mad_pool"] = (np.abs(pool - atom_value) <= 2*pool_mad).mean()
    # Multi-scale KDE densities
    h = pool_std * (4/(3*len(pool)))**(1/5)
    for mul in [0.25, 0.5, 1.0, 2.0, 4.0]:
        hh = h * mul
        dens = np.sum(np.exp(-0.5 * ((pool - atom_value) / hh) ** 2)) / (len(pool) * hh * np.sqrt(2*np.pi))
        features[f"dens_h{mul}"] = float(dens)
    # log-concavity
    d_here = np.sum(np.exp(-0.5 * ((pool - atom_value) / h) ** 2)) / (len(pool) * h * np.sqrt(2*np.pi))
    d_left = np.sum(np.exp(-0.5 * ((pool - (atom_value - h)) / h) ** 2)) / (len(pool) * h * np.sqrt(2*np.pi))
    d_right = np.sum(np.exp(-0.5 * ((pool - (atom_value + h)) / h) ** 2)) / (len(pool) * h * np.sqrt(2*np.pi))
    features["logconcav"] = float(np.log(d_here + 1e-12) - 0.5*(np.log(d_left + 1e-12) + np.log(d_right + 1e-12)))
    try:
        kde = gaussian_kde(pool, bw_method='silverman')
        x_grid = np.linspace(pool.min() - 0.01, pool.max() + 0.01, 500)
        dens_grid = kde(x_grid)
        peaks, _ = find_peaks(dens_grid)
        valleys, _ = find_peaks(-dens_grid)
        features["dist_peak"] = float(np.min(np.abs(x_grid[peaks] - atom_value))) if len(peaks) > 0 else 0.0
        features["dist_valley"] = float(np.min(np.abs(x_grid[valleys] - atom_value))) if len(valleys) > 0 else 0.0
    except:
        features["dist_peak"] = 0.0
        features["dist_valley"] = 0.0
    return features


BEST_EN = [12, 19, 24, 25, 26, 27, 28, 29, 30, 31]
BEST_ZH = [16, 18, 20, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32]
DISC_ZH_IN = [16, 18, 22, 25]  # non-tie IN
DISC_ZH_OUT = [17, 19, 21, 24]  # non-tie OUT

for d, best, disc_in, disc_out in [
    ("MHClip_EN", BEST_EN, [19, 24, 25, 27, 28, 29, 30, 31], [13, 14, 15, 16, 17, 18, 20, 21, 22, 23]),
    ("MHClip_ZH", BEST_ZH, DISC_ZH_IN, DISC_ZH_OUT),
]:
    train, test_x, test_y = load(d)
    pool = np.concatenate([train, test_x])
    rounded = np.round(test_x, 4)
    atoms_vals = sorted(set(rounded))
    print(f"\n=== {d} ===")

    min_idx = min(best)
    X, y, idxs = [], [], []
    for idx, aval in enumerate(atoms_vals):
        if idx < min_idx: continue
        f = all_features(aval, pool, train)
        X.append(list(f.values()))
        y.append(1 if idx in best else 0)
        idxs.append(idx)
    feat_names = list(all_features(atoms_vals[0], pool, train).keys())
    X = np.array(X); y = np.array(y)
    n = len(y)
    print(f"  {len(feat_names)} features, {n} atoms, {y.sum()} IN / {n-y.sum()} OUT")

    # LR
    lr = LogisticRegression(max_iter=5000, class_weight='balanced', C=100.0)
    lr.fit(X, y)
    preds = lr.predict(X)
    correct = (preds == y).sum()
    print(f"  LR upper bound (all {len(feat_names)} features): {correct}/{n}")
    miss = [idxs[i] for i in range(n) if preds[i] != y[i]]
    print(f"  LR Missed: {miss}")

    # RF
    rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
    rf.fit(X, y)
    preds = rf.predict(X)
    correct = (preds == y).sum()
    print(f"  RF (depth 3) upper bound: {correct}/{n}")
    miss = [idxs[i] for i in range(n) if preds[i] != y[i]]
    print(f"  RF Missed: {miss}")

    # Focus on discriminating pairs: 4 IN vs 4 OUT (non-tie)
    print(f"\n  Discriminating atoms (non-tie passing-IN vs non-tie OUT):")
    print(f"  {'idx':>3s} {'label':>5s}  " + " ".join([f"{n[:10]:>10s}" for n in feat_names[:10]]))
    for idx, aval in enumerate(atoms_vals):
        if idx not in disc_in and idx not in disc_out: continue
        f = all_features(aval, pool, train)
        tag = "IN" if idx in disc_in else "OUT"
        row = " ".join([f"{f[k]:>10.4f}" for k in feat_names[:10]])
        print(f"  {idx:>3d} {tag:>5s}  {row}")
    # Second half of features
    print(f"  {'idx':>3s} {'label':>5s}  " + " ".join([f"{n[:10]:>10s}" for n in feat_names[10:20]]))
    for idx, aval in enumerate(atoms_vals):
        if idx not in disc_in and idx not in disc_out: continue
        f = all_features(aval, pool, train)
        tag = "IN" if idx in disc_in else "OUT"
        row = " ".join([f"{f[k]:>10.4f}" for k in feat_names[10:20]])
        print(f"  {idx:>3d} {tag:>5s}  {row}")
    print(f"  {'idx':>3s} {'label':>5s}  " + " ".join([f"{n[:10]:>10s}" for n in feat_names[20:]]))
    for idx, aval in enumerate(atoms_vals):
        if idx not in disc_in and idx not in disc_out: continue
        f = all_features(aval, pool, train)
        tag = "IN" if idx in disc_in else "OUT"
        row = " ".join([f"{f[k]:>10.4f}" for k in feat_names[20:]])
        print(f"  {idx:>3d} {tag:>5s}  {row}")
