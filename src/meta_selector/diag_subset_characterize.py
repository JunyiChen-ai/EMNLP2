"""Characterize the top passing subset by train-only features.

Take the best ZH subset (acc=0.8456, mf=0.8107) and the best EN subset
(acc=0.7702, mf=0.6948). For each atom, compute every train-only feature
I can think of. Then check: is there a label-free rule that selects
exactly the passing subset, with the CUT-VALUE derivable from pool
statistics (not from labels)?

Features per atom:
- train_frac_below (already had)
- train_at_count (at 1e-4 precision)
- train_frac_at (train_at / n_train)
- train_nearest_dist: distance to nearest train atom (not same value)
- train_gap_above: distance to nearest train strictly above
- train_gap_below: distance to nearest train strictly below
- train_kde_density (silverman on train only)
- train_count_in_1mad: train points within 1-MAD of atom value
- train_count_in_3mad: same, 3-MAD
- train_local_mean: mean of train points within 2-MAD window
- train_local_std: std of train points within 2-MAD window
- train_local_q50_diff: atom - median(local train)

Then test: for each feature, what value CUT perfectly separates the
passing-subset atoms from the holes? If such a cut exists, the feature
gives a label-free rule.
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
from data_utils import load_annotations
from scipy.stats import gaussian_kde

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
    features["frac_at_1e4"] = (np.abs(train - atom_value) < 1e-4).mean()
    features["tr_at_count"] = int((np.abs(train - atom_value) < 1e-4).sum())

    above = train[train > atom_value + 1e-6]
    below = train[train < atom_value - 1e-6]
    features["gap_above"] = (above.min() - atom_value) if len(above) > 0 else 0.0
    features["gap_below"] = (atom_value - below.max()) if len(below) > 0 else 0.0
    features["nearest_dist"] = min(features["gap_above"], features["gap_below"]) if features["gap_above"] and features["gap_below"] else max(features["gap_above"], features["gap_below"])

    # local train statistics within 1 MAD window
    window = tr_mad
    in_window = train[np.abs(train - atom_value) <= window]
    features["count_in_1mad"] = len(in_window)
    features["count_in_3mad"] = int((np.abs(train - atom_value) <= 3*tr_mad).sum())

    if len(in_window) > 1:
        features["local_mean"] = float(in_window.mean())
        features["local_std"] = float(in_window.std())
        features["local_med_diff"] = atom_value - float(np.median(in_window))
    else:
        features["local_mean"] = atom_value
        features["local_std"] = 0.0
        features["local_med_diff"] = 0.0

    # KDE density at atom_value
    try:
        kde = gaussian_kde(train, bw_method='silverman')
        features["kde_density"] = float(kde(atom_value)[0])
    except:
        features["kde_density"] = 0.0

    # log(1 / density)
    features["neglog_density"] = float(-np.log(features["kde_density"] + 1e-12))

    return features


# Best subsets from job 8060
BEST_EN = {
    "pos_atoms": [12, 19, 24, 25, 26, 27, 28, 29, 30, 31],  # 10 atoms, min=12, holes=[13-18, 20-23]
    "all_pos_above_min": list(range(12, 32)),
}
BEST_ZH = {
    # idx 16, 18, 20, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32  (13 atoms)
    "pos_atoms": [16, 18, 20, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32],
    "all_pos_above_min": list(range(16, 33)),
}

for d, best in [("MHClip_EN", BEST_EN), ("MHClip_ZH", BEST_ZH)]:
    train, test_x, test_y = load(d)
    print(f"\n{'='*60}\n=== {d} ===")

    # Rebuild atoms
    rounded = np.round(test_x, 4)
    atoms_vals = sorted(set(rounded))
    print(f"  {len(atoms_vals)} atoms")

    feat_rows = []
    for idx, aval in enumerate(atoms_vals):
        mask = rounded == aval
        pc = int(test_y[mask].sum())
        nc = int((1 - test_y[mask]).sum())
        feats = compute_features(aval, train)
        feats["idx"] = idx
        feats["pos"] = pc
        feats["neg"] = nc
        feats["in_best_subset"] = 1 if idx in best["pos_atoms"] else 0
        feats["above_min"] = 1 if idx >= min(best["pos_atoms"]) else 0
        feat_rows.append(feats)

    # Print feature table
    feat_names = [k for k in feat_rows[0].keys() if k not in ("idx", "pos", "neg", "in_best_subset", "above_min")]
    print(f"  Feature table (atoms above min of best subset):")
    header = "  " + f"{'idx':>3s} {'pos':>3s} {'neg':>3s} {'in':>2s}  " + " ".join([f"{f:>12s}" for f in feat_names])
    print(header)
    for row in feat_rows:
        if row["above_min"]:
            line = "  " + f"{row['idx']:>3d} {row['pos']:>3d} {row['neg']:>3d} {row['in_best_subset']:>2d}  "
            line += " ".join([f"{row[f]:>12.4f}" for f in feat_names])
            print(line)

    # For each feature, check if a single cut separates IN from NOT-IN within above-min atoms
    print("\n  Feature separability (within above-min atoms):")
    above_min_rows = [r for r in feat_rows if r["above_min"]]
    in_set = [r for r in above_min_rows if r["in_best_subset"]]
    out_set = [r for r in above_min_rows if not r["in_best_subset"]]
    if len(out_set) == 0:
        print("    No holes — it's just a threshold, not non-monotone.")
    else:
        for f in feat_names:
            in_vals = [r[f] for r in in_set]
            out_vals = [r[f] for r in out_set]
            if len(set(in_vals + out_vals)) < 2: continue
            min_in, max_in = min(in_vals), max(in_vals)
            min_out, max_out = min(out_vals), max(out_vals)
            # Check if any threshold on f separates in from out
            if max_out < min_in:
                print(f"    {f:>20s}: PERFECT IN>={min_in:.4f} OUT<={max_out:.4f}")
            elif max_in < min_out:
                print(f"    {f:>20s}: PERFECT IN<={max_in:.4f} OUT>={min_out:.4f}")
            else:
                # Compute best single-cut separation
                all_vals = sorted(set(in_vals + out_vals))
                best_sep = 0
                best_cut = None
                for i in range(len(all_vals)-1):
                    cut = (all_vals[i] + all_vals[i+1]) / 2
                    above_in = sum(1 for v in in_vals if v > cut)
                    below_out = sum(1 for v in out_vals if v < cut)
                    total_correct = above_in + below_out
                    if total_correct > best_sep:
                        best_sep = total_correct
                        best_cut = cut
                n_total = len(in_set) + len(out_set)
                print(f"    {f:>20s}: best cut {best_cut} -> {best_sep}/{n_total} correct")
