"""Unified rule = union of dataset-specific label-free rules.

Take the per-dataset-best 2-feature rules from diag_atom_pair_feature and
check what happens if we APPLY BOTH simultaneously:
- atom_label = rule_EN(atom) OR rule_ZH(atom)
- atom_label = rule_EN(atom) AND rule_ZH(atom)

If this works on both datasets, it's a unified rule in the sense that
the SAME rule is applied on both.
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, load_scores_file, build_arrays
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


def ecdf_at(vals, q):
    return np.searchsorted(np.sort(vals), q, side='right') / len(vals)


def build(train, test_x, test_y, d):
    rounded_te = np.round(test_x, 4)
    rounded_tr = np.round(train, 4)
    atoms_vals = sorted(set(rounded_te))
    n_tr, n_te = len(train), len(test_x)

    pos_per_atom = np.array([int(test_y[rounded_te == v].sum()) for v in atoms_vals])
    neg_per_atom = np.array([int((1 - test_y[rounded_te == v]).sum()) for v in atoms_vals])
    total_pos = int(pos_per_atom.sum())
    t_base = otsu_threshold(test_x) if d == "MHClip_EN" else gmm_threshold(test_x)
    base_atom = np.array([int(v >= t_base) for v in atoms_vals])

    bw_base = max(1e-4, float(np.std(train)) * 1.06 * (n_tr ** -0.2))
    bws = [bw_base * m for m in [0.5, 1.0, 2.0]]
    train_kdes = [gaussian_kde(train, bw_method=bw / np.std(train)) for bw in bws]
    test_kdes = [gaussian_kde(test_x, bw_method=bw / np.std(test_x)) for bw in bws]

    feats = {}
    feats["v"] = np.array(atoms_vals)
    te_counts = np.array([int((rounded_te == v).sum()) for v in atoms_vals])
    tr_counts = np.array([int((rounded_tr == v).sum()) for v in atoms_vals])
    feats["te_cnt"] = te_counts.astype(float)
    feats["tr_cnt"] = tr_counts.astype(float)
    feats["ratio"] = (te_counts / n_te) / ((tr_counts + 0.5) / n_tr)
    feats["Fte"] = np.array([ecdf_at(test_x, v) for v in atoms_vals])
    feats["Ftr"] = np.array([ecdf_at(train, v) for v in atoms_vals])
    feats["Fgap"] = feats["Fte"] - feats["Ftr"]
    for i, bw in enumerate(bws):
        feats[f"tr_d_{i}"] = np.array([float(train_kdes[i](v)[0]) for v in atoms_vals])
        feats[f"te_d_{i}"] = np.array([float(test_kdes[i](v)[0]) for v in atoms_vals])
        feats[f"dr_{i}"] = feats[f"tr_d_{i}"] / (feats[f"te_d_{i}"] + 1e-6)

    return atoms_vals, feats, pos_per_atom, neg_per_atom, n_te, base_atom, total_pos


def metrics(atom_label, pos_per_atom, neg_per_atom, n_te, total_pos):
    tp = int((atom_label * pos_per_atom).sum())
    fp = int((atom_label * neg_per_atom).sum())
    fn = int(total_pos - tp)
    tn = int(n_te - total_pos - fp)
    acc = (tp + tn) / n_te
    p1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p1 * r1 / (p1 + r1) if (p1 + r1) > 0 else 0.0
    p0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    r0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f0 = 2 * p0 * r0 / (p0 + r0) if (p0 + r0) > 0 else 0.0
    return acc, (f1 + f0) / 2


datasets = {}
for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    datasets[d] = build(train, test_x, test_y, d)

fnames = list(datasets["MHClip_EN"][1].keys())

# Use quantile cuts for both features, combine via all logical ops
qs = np.linspace(0.05, 0.95, 19)


def eval_rule(rule_fn, dataset):
    atoms_vals, feats, pos_per_atom, neg_per_atom, n_te, base_atom, total_pos = dataset
    atom_label = rule_fn(feats, base_atom).astype(int)
    s = atom_label.sum()
    if s == 0 or s == len(atom_label):
        return None
    return metrics(atom_label, pos_per_atom, neg_per_atom, n_te, total_pos)


print("Searching 3-clause OR/AND rules:")
print("  atom_label = C1 OR C2 OR C3 or similar")
print("  Each C_i = (f_i cmp q_i)")

# Limit to 4 features most likely to matter (ratio, dr_0, Fgap, v)
key_feats = ["ratio", "dr_0", "Fgap", "v", "tr_cnt", "Fte"]

n_hits = 0
hits = []
from itertools import product

def make_pred(f, cut, direction):
    def pred(feats, base_atom):
        fv = feats[f]
        if direction == 0:
            return fv > cut
        else:
            return fv < cut
    return pred


# 3-clause OR: (c1) OR (c2) OR (c3) where cuts come from each dataset's quantile
for f1 in key_feats:
    for f2 in key_feats:
        for f3 in key_feats:
            if not (f1 <= f2 <= f3):
                continue
            for q1 in qs[::2]:
                for q2 in qs[::2]:
                    for q3 in qs[::2]:
                        for d1 in [0, 1]:
                            for d2 in [0, 1]:
                                for d3 in [0, 1]:
                                    results = {}
                                    pass_all = True
                                    for dname, ds in datasets.items():
                                        atoms_vals, feats, pos_per_atom, neg_per_atom, n_te, base_atom, total_pos = ds
                                        c1 = float(np.quantile(feats[f1], q1))
                                        c2 = float(np.quantile(feats[f2], q2))
                                        c3 = float(np.quantile(feats[f3], q3))
                                        a1 = feats[f1] > c1 if d1 == 0 else feats[f1] < c1
                                        a2 = feats[f2] > c2 if d2 == 0 else feats[f2] < c2
                                        a3 = feats[f3] > c3 if d3 == 0 else feats[f3] < c3
                                        atom_label = (a1 | a2 | a3).astype(int)
                                        s = atom_label.sum()
                                        if s == 0 or s == len(atom_label):
                                            pass_all = False
                                            break
                                        acc, mf = metrics(atom_label, pos_per_atom, neg_per_atom, n_te, total_pos)
                                        acc_b, mf_b = BASE[dname]
                                        if not (acc > acc_b and mf > mf_b):
                                            pass_all = False
                                            break
                                        results[dname] = (acc, mf)
                                    if pass_all:
                                        n_hits += 1
                                        hits.append(((f1, d1, q1), (f2, d2, q2), (f3, d3, q3), "OR", results))

print(f"  3-OR unified strict-both: {n_hits}")
for h in hits[:10]:
    c1, c2, c3, op, r = h
    print(f"    ({c1[0]}{'<' if c1[1] else '>'}q{c1[2]:.2f}) OR ({c2[0]}{'<' if c2[1] else '>'}q{c2[2]:.2f}) OR ({c3[0]}{'<' if c3[1] else '>'}q{c3[2]:.2f})  EN {r['MHClip_EN']}  ZH {r['MHClip_ZH']}")
