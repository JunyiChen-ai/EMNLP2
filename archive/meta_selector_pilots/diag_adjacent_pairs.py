"""The barrier: can ANY label-free feature f(atom_value, pool) separate
adjacent-score atoms where one is net-positive and the next is net-negative?

ZH passing subset has:
  idx 16 (+3, IN) val=0.0373
  idx 17 (-2, OUT) val=0.0474  <- must be OUT despite being adjacent
  idx 18 (+5, IN) val=0.0601
  idx 19 (-1, OUT) val=0.0759  <- must be OUT
  idx 20 (=0, IN)  val=0.0953
  idx 21 (-1, OUT) val=0.1192
  idx 22 (+2, IN)  val=0.1480
  idx 23 (=0, IN)  val=0.1824
  idx 24 (-1, OUT) val=0.2227

The pattern alternates IN/OUT across adjacent atoms. No monotone
f(atom_value, pool) can do this — any feature that depends only on atom
value and pool (no labels) gives a MONOTONE (in score) or near-monotone
ordering of atoms. A 'zigzag' IN/OUT/IN/OUT/IN/OUT pattern requires a
feature with 4+ sign changes in x, which pool-based features do not have.

Verify: for every feature we've computed, print its sign of derivative
at each atom. If no feature has the needed 4 sign changes,
no rule over that feature space can produce the alternating pattern.
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
from data_utils import load_annotations
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}


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
    tr_mad = float(np.median(np.abs(train - tr_med)))
    features = {}
    features["val"] = atom_value
    features["frac_below"] = (train < atom_value).mean()
    features["frac_below_pool"] = (pool < atom_value).mean()
    features["rank_unique"] = float(np.sum(np.unique(pool) <= atom_value) / len(np.unique(pool)))
    features["z_pool"] = (atom_value - pool_mean) / (pool_std + 1e-9)
    features["mad_pool"] = (atom_value - pool_med) / (pool_mad + 1e-9)
    above = train[train > atom_value + 1e-6]
    below = train[train < atom_value - 1e-6]
    features["gap_up"] = (above.min() - atom_value) if len(above) > 0 else 0.0
    features["gap_dn"] = (atom_value - below.max()) if len(below) > 0 else 0.0
    features["log_asym"] = np.log((features["gap_up"] + 1e-6) / (features["gap_dn"] + 1e-6))
    in_1mad = train[np.abs(train - atom_value) <= tr_mad]
    features["cnt_1mad"] = len(in_1mad) / len(train)
    h = pool_std * (4/(3*len(pool)))**(1/5)
    for mul in [0.5, 1.0, 2.0, 4.0]:
        hh = h * mul
        dens = np.sum(np.exp(-0.5 * ((pool - atom_value) / hh) ** 2)) / (len(pool) * hh * np.sqrt(2*np.pi))
        features[f"dens_h{mul}"] = float(dens)
    # log-concavity
    def dens_at(x, hh):
        return np.sum(np.exp(-0.5 * ((pool - x) / hh) ** 2)) / (len(pool) * hh * np.sqrt(2*np.pi))
    d_here = dens_at(atom_value, h)
    d_left = dens_at(atom_value - h, h)
    d_right = dens_at(atom_value + h, h)
    features["logconcav"] = float(np.log(d_here + 1e-12) - 0.5*(np.log(d_left + 1e-12) + np.log(d_right + 1e-12)))
    try:
        kde = gaussian_kde(pool, bw_method='silverman')
        x_grid = np.linspace(pool.min() - 0.01, pool.max() + 0.01, 500)
        dens_grid = kde(x_grid)
        peaks, _ = find_peaks(dens_grid)
        features["dist_peak"] = float(np.min(np.abs(x_grid[peaks] - atom_value))) if len(peaks) > 0 else 0.0
    except:
        features["dist_peak"] = 0.0
    return features


for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    pool = np.concatenate([train, test_x])
    rounded = np.round(test_x, 4)
    atoms_vals = sorted(set(rounded))
    print(f"\n=== {d} ===")

    # Compute feature matrix
    all_feats = []
    for aval in atoms_vals:
        all_feats.append(all_features(aval, pool, train))
    feat_names = list(all_feats[0].keys())
    F = np.array([[af[k] for k in feat_names] for af in all_feats])

    # For each feature, count sign changes in the sequence (F[i+1] - F[i])
    print(f"  Feature sign-change counts (higher = more non-monotone):")
    for fi, fn in enumerate(feat_names):
        col = F[:, fi]
        diffs = np.diff(col)
        signs = np.sign(diffs)
        # Count sign changes
        sc = int(np.sum(np.abs(np.diff(signs[signs != 0]))) // 2)
        print(f"    {fn:15s} range=[{col.min():.4f}, {col.max():.4f}] sign_changes={sc}")

    # For each atom, compute label net
    print(f"\n  Atom net labels:")
    for idx, aval in enumerate(atoms_vals):
        mask = rounded == aval
        pc = int(test_y[mask].sum())
        nc = int((1 - test_y[mask]).sum())
        net = pc - nc
        print(f"    idx {idx:2d} val={aval:.4f} net={net:+d}")
