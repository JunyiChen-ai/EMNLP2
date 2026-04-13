"""Verify the EN rule from job 8090 is genuinely non-suffix at whole-atom
granularity. Print the per-atom label assignment in score order.
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import load_scores_file, build_arrays
from data_utils import load_annotations
from scipy.stats import gaussian_kde

def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    return np.array(list(train.values()), dtype=float), test_x, test_y


for d, rule_desc in [
    ("MHClip_EN", "(v>0.3491) OR (dr_0<0.8795)"),
    ("MHClip_ZH", "(v>0.0333) AND (ratio>0.4328)"),
]:
    train, test_x, test_y = load(d)
    print(f"\n=== {d}: {rule_desc} ===")
    rounded_te = np.round(test_x, 4)
    rounded_tr = np.round(train, 4)
    atoms_vals = np.array(sorted(set(rounded_te)))
    n_tr, n_te = len(train), len(test_x)

    te_counts = np.array([int((rounded_te == v).sum()) for v in atoms_vals])
    tr_counts = np.array([int((rounded_tr == v).sum()) for v in atoms_vals])
    ratio = (te_counts / n_te) / ((tr_counts + 0.5) / n_tr)

    bw = max(1e-4, float(np.std(train)) * 1.06 * (n_tr ** -0.2)) * 0.5
    train_kde = gaussian_kde(train, bw_method=bw / np.std(train))
    test_kde = gaussian_kde(test_x, bw_method=bw / np.std(test_x))
    tr_d_0 = np.array([float(train_kde(v)[0]) for v in atoms_vals])
    te_d_0 = np.array([float(test_kde(v)[0]) for v in atoms_vals])
    dr_0 = tr_d_0 / (te_d_0 + 1e-6)

    if d == "MHClip_EN":
        atom_label = (atoms_vals > 0.3491) | (dr_0 < 0.8795)
    else:
        atom_label = (atoms_vals > 0.0333) & (ratio > 0.4328)

    print(f"  {'idx':>3} {'v':>8} {'dr_0':>8} {'ratio':>8} {'label':>6} {'p/n':>6}")
    for i, v in enumerate(atoms_vals):
        mask = rounded_te == v
        pc = int(test_y[mask].sum())
        nc = int((1 - test_y[mask]).sum())
        lab = int(atom_label[i])
        print(f"  {i:>3} {v:>8.4f} {dr_0[i]:>8.4f} {ratio[i]:>8.4f} {lab:>6} {pc}/{nc}")

    seen_1 = False
    transitions = 0
    prev = atom_label[0]
    for lab in atom_label[1:]:
        if lab != prev:
            transitions += 1
            prev = lab
    print(f"  transitions (0->1 or 1->0): {transitions}")
    print(f"  NON-SUFFIX (multiple transitions): {transitions > 1}")
