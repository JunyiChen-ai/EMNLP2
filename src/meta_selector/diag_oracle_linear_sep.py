"""Oracle linear-separability probe on the BASE=POS atoms only.

This is a diagnostic, NOT a rule. Question: among atoms where baseline
predicts POS, is there any LINEAR combination of label-free features that
separates the "should-stay-POS" atoms from the "should-flip-to-NEG" atoms
on ZH (and symmetrically for EN's base-flips)?

If the LR upper bound cannot achieve perfect separation using label-free
features, then no linear non-suffix label-free rule can exist. This gives
a hard ceiling.

Features per atom (all label-free):
  - rounded value v
  - log v
  - test_count, train_count, log test_count, log train_count
  - test_count/train_count ratio (Laplace)
  - atom index in sorted order
  - test CDF, train CDF, gap
  - distance to nearest lower/higher atom
  - train density at v (KDE h=silverman and multiples)
  - test density at v
  - density ratio at multiple bandwidths
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, load_scores_file, build_arrays
from data_utils import load_annotations
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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


def ecdf_at(vals, q):
    return np.searchsorted(np.sort(vals), q, side='right') / len(vals)


for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    n_tr, n_te = len(train), len(test_x)
    t_base = otsu_threshold(test_x) if d == "MHClip_EN" else gmm_threshold(test_x)
    print(f"\n=== {d} === baseline t={t_base:.4f}")

    rounded_te = np.round(test_x, 4)
    rounded_tr = np.round(train, 4)
    atoms_vals = sorted(set(rounded_te))

    # Compute per-atom features
    feats = []
    atom_pos, atom_neg, atom_base = [], [], []
    # KDE train at multiple BW
    bw_base = max(1e-4, float(np.std(train)) * 1.06 * (n_tr ** -0.2))
    bws = [bw_base * m for m in [0.25, 0.5, 1.0, 2.0, 4.0]]
    train_kdes = []
    for bw in bws:
        kde = gaussian_kde(train, bw_method=bw / np.std(train))
        train_kdes.append(kde)
    test_kdes = []
    for bw in bws:
        kde = gaussian_kde(test_x, bw_method=bw / np.std(test_x))
        test_kdes.append(kde)

    for i, v in enumerate(atoms_vals):
        tc = int((rounded_te == v).sum())
        trc = int((rounded_tr == v).sum())
        mask = rounded_te == v
        pc = int(test_y[mask].sum())
        nc = int((1 - test_y[mask]).sum())
        base_is_pos = int(test_x[mask][0] >= t_base)
        fte = ecdf_at(test_x, v)
        ftr = ecdf_at(train, v)
        # distances
        if i == 0:
            d_lo = v - atoms_vals[0] + 1e-6
        else:
            d_lo = v - atoms_vals[i-1]
        if i == len(atoms_vals) - 1:
            d_hi = atoms_vals[-1] - v + 1e-6
        else:
            d_hi = atoms_vals[i+1] - v

        row = [
            v, np.log(v + 1e-6), float(i), float(len(atoms_vals) - i),
            tc, trc, np.log(tc + 1), np.log(trc + 1),
            (tc / n_te) / ((trc + 0.5) / n_tr),
            fte, ftr, fte - ftr,
            d_lo, d_hi, np.log(d_lo + 1e-6), np.log(d_hi + 1e-6),
        ]
        for kde in train_kdes:
            row.append(float(kde(v)[0]))
        for kde in test_kdes:
            row.append(float(kde(v)[0]))
        # density ratios
        for i_bw in range(len(bws)):
            tr_d = float(train_kdes[i_bw](v)[0])
            te_d = float(test_kdes[i_bw](v)[0])
            row.append(np.log((tr_d + 1e-6) / (te_d + 1e-6)))
            row.append(tr_d / (te_d + 1e-6))

        feats.append(row)
        atom_pos.append(pc)
        atom_neg.append(nc)
        atom_base.append(base_is_pos)

    feats = np.array(feats, dtype=float)
    atom_pos = np.array(atom_pos)
    atom_neg = np.array(atom_neg)
    atom_base = np.array(atom_base)
    print(f"  atoms={len(atoms_vals)}, features={feats.shape[1]}")

    # Oracle label per atom: net majority
    net = atom_pos - atom_neg
    oracle_atom = (net > 0).astype(int)
    # Among base=POS atoms, who needs to be flipped?
    base_pos_mask = atom_base == 1
    # target: stay POS if oracle=POS else flip to NEG (target = oracle)
    print(f"  base=POS atoms: {int(base_pos_mask.sum())}")
    print(f"    oracle=POS among base=POS: {int((oracle_atom[base_pos_mask] == 1).sum())}")
    print(f"    oracle=NEG among base=POS: {int((oracle_atom[base_pos_mask] == 0).sum())}")

    # base=NEG side
    base_neg_mask = atom_base == 0
    print(f"  base=NEG atoms: {int(base_neg_mask.sum())}")
    print(f"    oracle=POS among base=NEG: {int((oracle_atom[base_neg_mask] == 1).sum())}")
    print(f"    oracle=NEG among base=NEG: {int((oracle_atom[base_neg_mask] == 0).sum())}")

    # LR upper bound for base=POS side (can we find POS-stay vs POS-flip?)
    X = feats[base_pos_mask]
    y = oracle_atom[base_pos_mask]
    if len(set(y)) == 2:
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        lr = LogisticRegression(C=1e6, max_iter=5000, penalty='l2').fit(Xs, y)
        yhat = lr.predict(Xs)
        correct = int((yhat == y).sum())
        print(f"  LR upper bound [base=POS subset]: {correct}/{len(y)} (C=1e6)")
        # L1 to see sparsity
        lr_l1 = LogisticRegression(C=1e6, max_iter=5000, penalty='l1', solver='liblinear').fit(Xs, y)
        yhat1 = lr_l1.predict(Xs)
        print(f"  LR L1 upper bound: {int((yhat1 == y).sum())}/{len(y)}")
    else:
        print(f"  base=POS subset has single oracle class")

    # LR upper bound for base=NEG side
    X = feats[base_neg_mask]
    y = oracle_atom[base_neg_mask]
    if len(set(y)) == 2:
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        lr = LogisticRegression(C=1e6, max_iter=5000, penalty='l2').fit(Xs, y)
        yhat = lr.predict(Xs)
        correct = int((yhat == y).sum())
        print(f"  LR upper bound [base=NEG subset]: {correct}/{len(y)} (C=1e6)")
        lr_l1 = LogisticRegression(C=1e6, max_iter=5000, penalty='l1', solver='liblinear').fit(Xs, y)
        yhat1 = lr_l1.predict(Xs)
        print(f"  LR L1 upper bound: {int((yhat1 == y).sum())}/{len(y)}")
    else:
        print(f"  base=NEG subset has single oracle class")

    # Full LR (all atoms)
    sc = StandardScaler()
    Xs = sc.fit_transform(feats)
    lr = LogisticRegression(C=1e6, max_iter=5000, penalty='l2').fit(Xs, oracle_atom)
    yhat = lr.predict(Xs)
    print(f"  LR upper bound [all atoms, predict oracle]: {int((yhat == oracle_atom).sum())}/{len(oracle_atom)}")
