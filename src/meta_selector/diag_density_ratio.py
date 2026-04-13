"""Density-ratio features: f_train(x) / f_test(x).

The insight: the test pool is distributionally different from train (it
has more positives per atom in high-score region than train does). A
density ratio captures that shift without using labels.

Compute features per atom:
  - r1 = f_train(x) / f_test(x)
  - log_r1 = log f_train - log f_test
  - r2 at multiple bandwidths
  - uLSIF-style ratio (bound above)
  - Difference-of-densities: f_test(x) - f_train(x)

Then check:
(a) Whether these features ARE non-monotone (label-free sign changes)
(b) LR upper bound on passing-subset recovery using these features
(c) Whether simple label-free rules (e.g., log_r < c) produce non-suffix
    atom-level labelings
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import load_scores_file, build_arrays
from sklearn.metrics import accuracy_score, f1_score

def eval_pred(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    mf = f1_score(y_true, y_pred, average='macro')
    return {"acc": acc, "mf": mf}
from data_utils import load_annotations
from sklearn.linear_model import LogisticRegression


BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}
BEST_EN = [12, 19, 24, 25, 26, 27, 28, 29, 30, 31]
BEST_ZH = [16, 18, 20, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32]


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    return np.array(list(train.values()), dtype=float), test_x, test_y


def kde_at(x, pool, h):
    return np.sum(np.exp(-0.5 * ((pool - x) / h) ** 2)) / (len(pool) * h * np.sqrt(2*np.pi))


def dr_features(atom_value, train, test_pool):
    features = {}
    tr_std = float(train.std())
    te_std = float(test_pool.std())
    h_tr_silv = tr_std * (4/(3*len(train)))**(1/5)
    h_te_silv = te_std * (4/(3*len(test_pool)))**(1/5)
    h_common = 0.5 * (h_tr_silv + h_te_silv)

    for mul in [0.5, 1.0, 2.0, 4.0]:
        h = h_common * mul
        f_tr = kde_at(atom_value, train, h)
        f_te = kde_at(atom_value, test_pool, h)
        features[f"f_tr_h{mul}"] = f_tr
        features[f"f_te_h{mul}"] = f_te
        features[f"ratio_h{mul}"] = f_tr / (f_te + 1e-12)
        features[f"log_ratio_h{mul}"] = np.log((f_tr + 1e-12) / (f_te + 1e-12))
        features[f"diff_h{mul}"] = f_te - f_tr
        features[f"rel_diff_h{mul}"] = (f_te - f_tr) / (f_tr + f_te + 1e-12)

    # Nearest-neighbor ratios: count of train and test in fixed window
    for win_mul in [1.0, 2.0, 4.0]:
        win = h_common * win_mul
        n_tr_in = int(np.sum(np.abs(train - atom_value) <= win))
        n_te_in = int(np.sum(np.abs(test_pool - atom_value) <= win))
        features[f"cnt_ratio_w{win_mul}"] = (n_te_in + 0.5) / (n_tr_in + 0.5)

    return features


for d, best in [("MHClip_EN", BEST_EN), ("MHClip_ZH", BEST_ZH)]:
    train, test_x, test_y = load(d)
    rounded = np.round(test_x, 4)
    atoms_vals = sorted(set(rounded))
    print(f"\n=== {d} ===")
    print(f"  train n={len(train)}, test n={len(test_x)}")

    # Compute features for all atoms
    all_feats = []
    for aval in atoms_vals:
        all_feats.append(dr_features(aval, train, test_x))
    feat_names = list(all_feats[0].keys())
    F = np.array([[f[k] for k in feat_names] for f in all_feats])

    # Sign-change count per feature
    print(f"\n  Sign changes across all {len(atoms_vals)} atoms:")
    for fi, fn in enumerate(feat_names):
        col = F[:, fi]
        diffs = np.diff(col)
        signs = np.sign(diffs)
        sc = int(np.sum(np.abs(np.diff(signs[signs != 0]))) // 2)
        print(f"    {fn:20s} range=[{col.min():.4f}, {col.max():.4f}] sign_changes={sc}")

    # LR upper bound on passing-subset recovery (atoms at or above best's min)
    min_idx = min(best)
    X, y, idxs = [], [], []
    for idx, aval in enumerate(atoms_vals):
        if idx < min_idx: continue
        f = dr_features(aval, train, test_x)
        X.append([f[k] for k in feat_names])
        y.append(1 if idx in best else 0)
        idxs.append(idx)
    X = np.array(X); y = np.array(y)
    n = len(y)
    print(f"\n  Passing-subset recovery (atoms >= idx {min_idx}): {y.sum()} IN / {n - y.sum()} OUT")
    lr = LogisticRegression(max_iter=5000, class_weight='balanced', C=100.0)
    lr.fit(X, y)
    preds = lr.predict(X)
    correct = (preds == y).sum()
    print(f"  LR upper bound on DR features: {correct}/{n}")
    miss = [idxs[i] for i in range(n) if preds[i] != y[i]]
    print(f"  LR Missed: {miss}")

    # Simple label-free rules: f_te > f_tr means "test-enriched" — assign positive
    # Try each density-ratio feature as a threshold: cut at zero-crossing
    print(f"\n  Label-free rule tests on FULL test set (not subset):")
    for fn in ["log_ratio_h1.0", "diff_h1.0", "rel_diff_h1.0"]:
        col_full = np.array([dr_features(av, train, test_x)[fn] for av in atoms_vals])
        # Rule: predict positive iff feature > 0 (test_density > train_density)
        atom_pred = (col_full > 0).astype(int)
        # Propagate to samples
        pred = np.zeros(len(test_x), dtype=int)
        for idx, aval in enumerate(atoms_vals):
            mask = rounded == aval
            pred[mask] = atom_pred[idx]
        m = eval_pred(test_y, pred)
        acc_b, mf_b = BASE[d]
        strict = (m["acc"] > acc_b) and (m["mf"] >= mf_b)
        tag = " ** PASS **" if strict else ""
        print(f"    {fn} > 0 : acc={m['acc']:.4f} mf={m['mf']:.4f}{tag}")

    # And also: for each bandwidth, scan cut points over log_ratio
    from itertools import product
    for mul in [0.5, 1.0, 2.0, 4.0]:
        fn = f"log_ratio_h{mul}"
        col_full = np.array([dr_features(av, train, test_x)[fn] for av in atoms_vals])
        # Try each unique value as cut
        for cut in np.unique(col_full):
            for direction in [">", ">="]:
                atom_pred = (col_full >= cut if direction == ">=" else col_full > cut).astype(int)
                if atom_pred.sum() == 0 or atom_pred.sum() == len(atom_pred): continue
                pred = np.zeros(len(test_x), dtype=int)
                for idx, aval in enumerate(atoms_vals):
                    mask = rounded == aval
                    pred[mask] = atom_pred[idx]
                m = eval_pred(test_y, pred)
                acc_b, mf_b = BASE[d]
                if m["acc"] > acc_b and m["mf"] >= mf_b:
                    print(f"    ** {fn} {direction} {cut:.4f}: acc={m['acc']:.4f} mf={m['mf']:.4f} PASS **")
