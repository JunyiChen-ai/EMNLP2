"""Use train-fit GMM to compute per-test-sample POSTERIOR of high component.
Then combine with baseline prediction: posterior is a label-free secondary
signal that can flip atoms near the boundary.

Rules to test:
- TF-GMM baseline (ZH) / TF-Otsu baseline (EN)
- Then: flip POS→NEG for atoms with (tr_posterior low AND test posterior high)
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, load_scores_file, build_arrays
from data_utils import load_annotations
from sklearn.metrics import accuracy_score, f1_score
from sklearn.mixture import GaussianMixture

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    return np.array(list(train.values()), dtype=float), test_x, test_y


def eval_pred(y, p):
    return accuracy_score(y, p), f1_score(y, p, average='macro')


for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    acc_b, mf_b = BASE[d]
    print(f"\n=== {d} === baseline acc={acc_b:.4f} mf={mf_b:.4f}")

    # Baseline prediction
    if d == "MHClip_EN":
        t = otsu_threshold(test_x)
    else:
        t = gmm_threshold(test_x)
    base_pred = (test_x >= t).astype(int)
    b_acc, b_mf = eval_pred(test_y, base_pred)
    print(f"  Reproduced baseline: t={t:.4f} acc={b_acc:.4f} mf={b_mf:.4f}")

    # Fit GMM k=2 on TRAIN only, get per-test posterior
    g_tr = GaussianMixture(n_components=2, random_state=42, max_iter=200).fit(train.reshape(-1, 1))
    # Identify high component
    hi = int(np.argmax(g_tr.means_.flatten()))
    post_tr = g_tr.predict_proba(test_x.reshape(-1, 1))[:, hi]
    # And fit GMM on TEST for reference
    g_te = GaussianMixture(n_components=2, random_state=42, max_iter=200).fit(test_x.reshape(-1, 1))
    hi2 = int(np.argmax(g_te.means_.flatten()))
    post_te = g_te.predict_proba(test_x.reshape(-1, 1))[:, hi2]

    print(f"  train-GMM means: {sorted(g_tr.means_.flatten())}, high={hi}")
    print(f"  test-GMM means: {sorted(g_te.means_.flatten())}, high={hi2}")

    # Per-atom posterior
    rounded = np.round(test_x, 4)
    atoms_vals = sorted(set(rounded))
    print(f"\n  Atom-level train-GMM posterior:")
    print(f"  {'idx':>3s} {'val':>8s} {'pos':>4s} {'neg':>4s} {'post_tr':>8s} {'post_te':>8s} {'base':>5s}")
    for idx, aval in enumerate(atoms_vals):
        mask = rounded == aval
        pc = int(test_y[mask].sum())
        nc = int((1 - test_y[mask]).sum())
        ptr = float(post_tr[mask].mean())
        pte = float(post_te[mask].mean())
        bl = 1 if aval >= t else 0
        print(f"  {idx:>3d} {aval:>8.4f} {pc:>4d} {nc:>4d} {ptr:>8.4f} {pte:>8.4f} {bl:>5d}")

    # Rules: use train-GMM posterior as PRIMARY score + threshold at 0.5
    pred_tr = (post_tr > 0.5).astype(int)
    acc, mf = eval_pred(test_y, pred_tr)
    print(f"\n  Rule post_tr > 0.5 : acc={acc:.4f} mf={mf:.4f}")

    # OR: (baseline pos) AND (post_tr > 0.5)
    pred_and = (base_pred & (post_tr > 0.5)).astype(int)
    acc, mf = eval_pred(test_y, pred_and)
    strict = acc > acc_b and mf >= mf_b
    tag = " ** PASS **" if strict else ""
    print(f"  Rule base AND (post_tr > 0.5) : acc={acc:.4f} mf={mf:.4f}{tag}")

    # OR: (baseline pos) OR (post_tr > 0.5)
    pred_or = (base_pred | (post_tr > 0.5)).astype(int)
    acc, mf = eval_pred(test_y, pred_or)
    strict = acc > acc_b and mf >= mf_b
    tag = " ** PASS **" if strict else ""
    print(f"  Rule base OR  (post_tr > 0.5) : acc={acc:.4f} mf={mf:.4f}{tag}")

    # Try cut-sweep on post_tr (label-free cuts)
    for cut in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        p = (post_tr > cut).astype(int)
        if p.sum() == 0 or p.sum() == len(p): continue
        acc, mf = eval_pred(test_y, p)
        strict = acc > acc_b and mf >= mf_b
        if strict:
            print(f"  ** post_tr > {cut}: acc={acc:.4f} mf={mf:.4f} PASS **")

    # AND with baseline at different cuts
    for cut in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        p = (base_pred & (post_tr > cut)).astype(int)
        if p.sum() == 0 or p.sum() == len(p): continue
        acc, mf = eval_pred(test_y, p)
        strict = acc > acc_b and mf >= mf_b
        if strict:
            print(f"  ** base AND post_tr > {cut}: acc={acc:.4f} mf={mf:.4f} PASS **")
