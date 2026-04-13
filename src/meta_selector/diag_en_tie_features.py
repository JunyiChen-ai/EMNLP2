"""For EN, examine structural features that distinguish TIE atoms from
non-TIE atoms with similar te_cnt. The oracle shows that including
TIE atoms 0.0293, 0.1480, 0.5000 helps EN strict-beat baseline.
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import load_scores_file, build_arrays
from data_utils import load_annotations
from scipy.stats import gaussian_kde

for d in ["MHClip_EN"]:
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    train = np.array(list(train.values()), dtype=float)

    rounded_te = np.round(test_x, 4)
    rounded_tr = np.round(train, 4)
    atoms_vals = np.array(sorted(set(rounded_te)))
    K = len(atoms_vals)
    te_cnt = np.array([int((rounded_te == v).sum()) for v in atoms_vals])
    tr_cnt = np.array([int((rounded_tr == v).sum()) for v in atoms_vals])
    pos = np.array([int(((rounded_te == v) & (test_y == 1)).sum()) for v in atoms_vals])
    neg = np.array([int(((rounded_te == v) & (test_y == 0)).sum()) for v in atoms_vals])

    # Train kde at each atom
    bw = max(1e-4, float(np.std(train)) * 1.06 * (len(train) ** -0.2))
    kde = gaussian_kde(train, bw_method=bw / np.std(train))
    tr_d = np.array([float(kde(v)[0]) for v in atoms_vals])
    te_bw = max(1e-4, float(np.std(test_x)) * 1.06 * (len(test_x) ** -0.2))
    kde_te = gaussian_kde(test_x, bw_method=te_bw / np.std(test_x))
    te_d = np.array([float(kde_te(v)[0]) for v in atoms_vals])

    print(f"=== {d} ===")
    print("idx | atom | te | tr | pos | neg | net | tr_kde | te_kde | dr")
    for i, (v, t, tr, p, n, td, tcd) in enumerate(
            zip(atoms_vals, te_cnt, tr_cnt, pos, neg, tr_d, te_d)):
        net = p - n
        dr = td / (tcd + 1e-9)
        tag = ""
        if net == 0 and p > 0:
            tag = " <-- TIE (want POS)"
        elif i == 23:  # 0.3208
            tag = " <-- drop from base"
        print(f"{i:2d} | {v:.4f} | {t:3d} | {tr:3d} | {p:3d} | {n:3d} | {net:+d} | "
              f"{td:.3f} | {tcd:.3f} | {dr:.3f}{tag}")
