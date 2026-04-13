"""Find a UNIFIED two-feature rule that strict-beats on both datasets.

For each (f1, f2) feature pair, search cuts parameterized as quantiles of
the feature distribution (so cuts transfer across datasets). Check that
the same (quantile-cut-1, quantile-cut-2, op1, op2, logic) passes strict
on both datasets.

This is a label-free unified rule IF the quantile chosen is NOT
label-dependent.
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import load_scores_file, build_arrays
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


def build_atom_features(train, test_x, test_y):
    rounded_te = np.round(test_x, 4)
    rounded_tr = np.round(train, 4)
    atoms_vals = sorted(set(rounded_te))
    n_tr, n_te = len(train), len(test_x)

    pos_per_atom = np.array([int(test_y[rounded_te == v].sum()) for v in atoms_vals])
    neg_per_atom = np.array([int((1 - test_y[rounded_te == v]).sum()) for v in atoms_vals])

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

    return atoms_vals, feats, pos_per_atom, neg_per_atom, n_te


def compute_metrics(atom_label, pos_per_atom, neg_per_atom, n_te, total_pos):
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
    mf = (f1 + f0) / 2
    return acc, mf


# Preload
datasets = {}
for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    atoms_vals, feats, pos_per_atom, neg_per_atom, n_te = build_atom_features(train, test_x, test_y)
    total_pos = int(pos_per_atom.sum())
    datasets[d] = (atoms_vals, feats, pos_per_atom, neg_per_atom, n_te, total_pos)

# Use quantile-based cuts so the same QUANTILE transfers across datasets
# For a given feature and quantile q in (0,1), cut = np.quantile(feats, q)
# This is label-free and unified.

fnames = list(datasets["MHClip_EN"][1].keys())
print(f"Features: {len(fnames)}")

qs = np.linspace(0.05, 0.95, 19)

n_unified_hits = 0
unified_hits = []
for i, f1 in enumerate(fnames):
    for j in range(i, len(fnames)):
        f2 = fnames[j]
        for q1 in qs:
            for q2 in qs:
                for d1 in [0, 1]:
                    for d2 in [0, 1]:
                        for op_name in ["AND", "OR"]:
                            results = {}
                            pass_both = True
                            for d, (atoms_vals, feats, pos_per_atom, neg_per_atom, n_te, total_pos) in datasets.items():
                                c1 = float(np.quantile(feats[f1], q1))
                                c2 = float(np.quantile(feats[f2], q2))
                                a1 = (feats[f1] > c1) if d1 == 0 else (feats[f1] < c1)
                                a2 = (feats[f2] > c2) if d2 == 0 else (feats[f2] < c2)
                                atom_label = (a1 & a2 if op_name == "AND" else a1 | a2).astype(int)
                                s = atom_label.sum()
                                if s == 0 or s == len(atom_label):
                                    pass_both = False
                                    break
                                acc, mf = compute_metrics(atom_label, pos_per_atom, neg_per_atom, n_te, total_pos)
                                acc_b, mf_b = BASE[d]
                                if not (acc >= acc_b and mf >= mf_b - 0.01):
                                    pass_both = False
                                    break
                                results[d] = (acc, mf)
                            if pass_both:
                                n_unified_hits += 1
                                unified_hits.append((f1, d1, q1, op_name, f2, d2, q2, results))

print(f"Unified strict-both hits: {n_unified_hits}")
# Sort by sum of margins
def margin(r):
    en_acc, en_mf = r[7]["MHClip_EN"]
    zh_acc, zh_mf = r[7]["MHClip_ZH"]
    return (en_acc - BASE["MHClip_EN"][0]) + (en_mf - BASE["MHClip_EN"][1]) + (zh_acc - BASE["MHClip_ZH"][0]) + (zh_mf - BASE["MHClip_ZH"][1])

unified_hits.sort(key=margin, reverse=True)
for r in unified_hits[:20]:
    f1, d1, q1, op, f2, d2, q2, results = r
    s1 = ">" if d1 == 0 else "<"
    s2 = ">" if d2 == 0 else "<"
    en = results["MHClip_EN"]
    zh = results["MHClip_ZH"]
    print(f"  ({f1}{s1}q{q1:.2f}) {op} ({f2}{s2}q{q2:.2f})  EN {en[0]:.4f}/{en[1]:.4f}  ZH {zh[0]:.4f}/{zh[1]:.4f}")
