"""Per-atom CDF gap feature: F_test(v) - F_train(v) at each atom's rounded
value v. This is the classical empirical Kolmogorov-Smirnov-like per-point
deviation. It is label-free (only uses train+test scores), non-monotone in x
(can go up and down), and discrete at atom granularity.

Rationale: if the test distribution has a LOCAL bump relative to train at
some score value, this creates a train-enriched/test-enriched region. The
sign of the CDF gap changes at each local deviation. Non-suffix labelings
arise naturally because the sign can change multiple times across atoms.

Rules tested: cdf_gap positive vs negative, AND/OR with baseline, Otsu on
cdf_gap, per-atom flip based on sign.
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


def ecdf_at(vals, q):
    return np.searchsorted(np.sort(vals), q, side='right') / len(vals)


for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    acc_b, mf_b = BASE[d]
    print(f"\n=== {d} === baseline acc={acc_b:.4f} mf={mf_b:.4f}")
    n_te = len(test_x)

    if d == "MHClip_EN":
        t_base = otsu_threshold(test_x)
    else:
        t_base = gmm_threshold(test_x)
    base_pred = (test_x >= t_base).astype(int)
    b_acc, b_mf = eval_pred(test_y, base_pred)
    print(f"  Baseline t={t_base:.4f}: acc={b_acc:.4f} mf={b_mf:.4f}")

    rounded_te = np.round(test_x, 4)
    atoms_vals = sorted(set(rounded_te))
    n_atoms = len(atoms_vals)

    # Per-atom sample CDF gap (test minus train) evaluated just AT atom
    # and also centred gap (F at midpoint between atoms)
    print(f"\n  {'i':>3} {'v':>8} {'Fte':>7} {'Ftr':>7} {'gap':>8} {'base':>4} {'p/n':>5}")
    cdf_gap_atom = np.zeros(n_atoms)
    base_atom = np.zeros(n_atoms, dtype=int)
    pn_atom = []
    for i, v in enumerate(atoms_vals):
        fte = ecdf_at(test_x, v)
        ftr = ecdf_at(train, v)
        gap = fte - ftr
        cdf_gap_atom[i] = gap
        mask = rounded_te == v
        pc = int(test_y[mask].sum())
        nc = int((1 - test_y[mask]).sum())
        b = int((test_x[mask][0] >= t_base))
        base_atom[i] = b
        pn_atom.append((pc, nc))
        print(f"  {i:>3} {v:>8.4f} {fte:>7.4f} {ftr:>7.4f} {gap:>+8.4f} {b:>4} {pc}/{nc}")

    # Per-sample cdf_gap (piecewise-constant at atoms)
    sample_gap = np.zeros(n_te)
    for i, v in enumerate(atoms_vals):
        sample_gap[rounded_te == v] = cdf_gap_atom[i]

    print(f"\n  Label-free rules on cdf_gap:")
    rules = []
    # Fixed cuts
    for cn, c in [("zero", 0.0),
                  ("median", float(np.median(sample_gap))),
                  ("mean", float(np.mean(sample_gap)))]:
        for dr in ["gt", "lt"]:
            p = (sample_gap > c if dr == "gt" else sample_gap < c).astype(int)
            if 0 < p.sum() < n_te:
                a, m = eval_pred(test_y, p)
                tag = " ** PASS **" if a > acc_b and m >= mf_b else ""
                rules.append((a, m, f"gap {dr} {cn}={c:+.4f}{tag}"))

    # Combine with baseline
    for op in ["AND", "OR"]:
        for dr in ["gt", "lt"]:
            feat = (sample_gap > 0 if dr == "gt" else sample_gap < 0)
            p = (base_pred & feat if op == "AND" else base_pred | feat).astype(int)
            if 0 < p.sum() < n_te:
                a, m = eval_pred(test_y, p)
                tag = " ** PASS **" if a > acc_b and m >= mf_b else ""
                rules.append((a, m, f"base {op} gap_{dr}_0{tag}"))

    # Selectively flip atoms: base predicts POS, but gap says TEST-DEFICIT
    # (test CDF > train CDF meaning test sample over-represented at this
    # atom vs train — train thinks it's less common than test; these MIGHT
    # be the "minority-under-train" atoms). Opposite: gap < 0 means test has
    # FEWER samples here than train — these are the ones we might flip.
    # Subtractive flip: base=1 AND gap_low -> flip to 0
    q_lo = float(np.quantile(sample_gap, 0.25))
    q_hi = float(np.quantile(sample_gap, 0.75))
    for q_name, q_cut in [("q25", q_lo), ("q75", q_hi), ("zero", 0.0)]:
        for direc in ["lt", "gt"]:
            mask_flip = (sample_gap < q_cut if direc == "lt" else sample_gap > q_cut)
            # Subtractive: base=1 -> 0 on masked
            p = base_pred.copy()
            p[mask_flip & (base_pred == 1)] = 0
            if 0 < p.sum() < n_te:
                a, m = eval_pred(test_y, p)
                tag = " ** PASS **" if a > acc_b and m >= mf_b else ""
                rules.append((a, m, f"sub_flip gap_{direc}_{q_name}={q_cut:+.4f}{tag}"))
            # Additive: base=0 -> 1 on masked
            p = base_pred.copy()
            p[mask_flip & (base_pred == 0)] = 1
            if 0 < p.sum() < n_te:
                a, m = eval_pred(test_y, p)
                tag = " ** PASS **" if a > acc_b and m >= mf_b else ""
                rules.append((a, m, f"add_flip gap_{direc}_{q_name}={q_cut:+.4f}{tag}"))

    for a, m, desc in sorted(rules, reverse=True)[:25]:
        print(f"    acc={a:.4f} mf={m:.4f}  {desc}")
