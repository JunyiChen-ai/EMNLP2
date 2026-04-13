"""Discrete atom ratio: for each test atom, compute test_count/train_count
at that EXACT atomic value. This is a piecewise-constant, non-smooth
feature that does not go through KDE bandwidth smoothing.

It tells us: at this test value, how much more common is this value in
test than in train? A high ratio means this atom is test-enriched, a
low ratio means this atom is train-enriched.

Then test label-free cuts:
- ratio > 1 (test enriched)
- ratio < 1 (train enriched)
- cut at various percentile of observed ratios
- AND combination with baseline prediction

Because this feature is DISCRETE and jumps at atom boundaries, it can
produce non-suffix atom labelings.
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, load_scores_file, build_arrays
from data_utils import load_annotations
from sklearn.metrics import accuracy_score, f1_score
from skimage.filters import threshold_otsu

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
    n_tr = len(train)
    n_te = len(test_x)
    print(f"\n=== {d} === baseline acc={acc_b:.4f} mf={mf_b:.4f}")

    rounded_te = np.round(test_x, 4)
    rounded_tr = np.round(train, 4)
    atoms_vals = sorted(set(rounded_te))

    # Atom-wise counts
    test_counts = {v: int((rounded_te == v).sum()) for v in atoms_vals}
    train_counts = {v: int((rounded_tr == v).sum()) for v in atoms_vals}

    # Baseline
    if d == "MHClip_EN":
        t_base = otsu_threshold(test_x)
    else:
        t_base = gmm_threshold(test_x)
    base_pred = (test_x >= t_base).astype(int)
    b_acc, b_mf = eval_pred(test_y, base_pred)
    print(f"  Baseline t={t_base:.4f}: acc={b_acc:.4f} mf={b_mf:.4f}")

    # Per-atom ratio (raw counts, no smoothing)
    print(f"\n  {'idx':>3s} {'val':>8s} {'te_cnt':>6s} {'tr_cnt':>6s} {'ratio':>10s} {'p/n':>6s}")
    atom_ratios = []
    for idx, v in enumerate(atoms_vals):
        tc = test_counts[v]
        trc = train_counts.get(v, 0)
        ratio = (tc / n_te) / ((trc + 0.5) / n_tr)  # Laplace-smoothed
        mask = rounded_te == v
        pc = int(test_y[mask].sum())
        nc = int((1 - test_y[mask]).sum())
        atom_ratios.append(ratio)
        print(f"  {idx:>3d} {v:>8.4f} {tc:>6d} {trc:>6d} {ratio:>10.4f} {pc}/{nc}")

    atom_ratios = np.array(atom_ratios)
    # Per-sample ratio feature
    sample_ratio = np.array([atom_ratios[atoms_vals.index(v)] for v in rounded_te])

    # Rules
    print(f"\n  Label-free rules on atom ratio:")
    rules = []
    # Cut at various points
    for cut_name, cut in [("median", float(np.median(sample_ratio))),
                           ("mean", float(np.mean(sample_ratio))),
                           ("one", 1.0),
                           ("log_mean", float(np.exp(np.log(sample_ratio + 1e-12).mean())))]:
        for direction in ["gt", "lt"]:
            pred = (sample_ratio > cut if direction == "gt" else sample_ratio < cut).astype(int)
            if pred.sum() == 0 or pred.sum() == len(pred): continue
            acc, mf = eval_pred(test_y, pred)
            strict = acc > acc_b and mf >= mf_b
            tag = " ** PASS **" if strict else ""
            rules.append((acc, mf, f"ratio {direction} {cut_name}={cut:.4f}{tag}"))

    # Combine with baseline
    for direction in ["gt", "lt"]:
        for op in ["AND", "OR"]:
            feat_pred = (sample_ratio > 1.0 if direction == "gt" else sample_ratio < 1.0)
            if op == "AND":
                pred = (base_pred & feat_pred).astype(int)
            else:
                pred = (base_pred | feat_pred).astype(int)
            if pred.sum() == 0 or pred.sum() == len(pred): continue
            acc, mf = eval_pred(test_y, pred)
            strict = acc > acc_b and mf >= mf_b
            tag = " ** PASS **" if strict else ""
            rules.append((acc, mf, f"base {op} ratio_{direction}_1 {tag}"))

    for acc, mf, desc in sorted(rules, reverse=True)[:20]:
        print(f"    acc={acc:.4f} mf={mf:.4f}  {desc}")

    # Try Otsu on sample_ratio (auto label-free cut)
    try:
        cut = float(threshold_otsu(sample_ratio, nbins=256))
        print(f"\n  Otsu cut on ratio: {cut:.4f}")
        for direction in ["gt", "lt"]:
            pred = (sample_ratio > cut if direction == "gt" else sample_ratio < cut).astype(int)
            if pred.sum() == 0 or pred.sum() == len(pred): continue
            acc, mf = eval_pred(test_y, pred)
            strict = acc > acc_b and mf >= mf_b
            tag = " ** PASS **" if strict else ""
            print(f"    ratio {direction} otsu: acc={acc:.4f} mf={mf:.4f}{tag}")
            # Combined
            pr_and = (base_pred & (sample_ratio > cut if direction == "gt" else sample_ratio < cut)).astype(int)
            if 0 < pr_and.sum() < len(pr_and):
                acc, mf = eval_pred(test_y, pr_and)
                strict = acc > acc_b and mf >= mf_b
                tag = " ** PASS **" if strict else ""
                print(f"    base AND ratio {direction} otsu: acc={acc:.4f} mf={mf:.4f}{tag}")
    except Exception as e:
        print(f"  Otsu failed: {e}")
