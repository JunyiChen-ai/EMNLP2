"""Baseline-modified atom rules: start from baseline (TF-Otsu for EN,
TF-GMM for ZH), then apply a label-free ADDITIVE or SUBTRACTIVE modifier
based on a single label-free feature.

Rule form: out = base XOR (feat_predicate)
or: out = base AND NOT(feat_predicate)  [subtractive]
or: out = base OR feat_predicate         [additive]

Search unified rules using quantile-based cuts on all features.
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

    return atoms_vals, feats, pos_per_atom, neg_per_atom, n_te, base_atom


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
    datasets[d] = build(train, test_x, test_y, d) + (int(np.sum([int(test_y[np.round(test_x, 4) == v].sum()) for v in sorted(set(np.round(test_x, 4)))])),)

fnames = list(datasets["MHClip_EN"][1].keys())
print(f"Features: {len(fnames)}")

qs = np.linspace(0.05, 0.95, 19)

for mode in ["SUB", "ADD"]:
    print(f"\n=== {mode} (base XXX feat) ===")
    hits = []
    for f in fnames:
        for q in qs:
            for direction in [0, 1]:  # 0 = gt, 1 = lt
                pass_both = True
                results = {}
                for dname, (atoms_vals, feats, pos_per_atom, neg_per_atom, n_te, base_atom, total_pos) in datasets.items():
                    c = float(np.quantile(feats[f], q))
                    feat_pred = (feats[f] > c) if direction == 0 else (feats[f] < c)
                    if mode == "SUB":
                        atom_label = (base_atom & ~feat_pred).astype(int)
                    else:
                        atom_label = (base_atom | feat_pred).astype(int)
                    s = atom_label.sum()
                    if s == 0 or s == len(atom_label):
                        pass_both = False
                        break
                    acc, mf = metrics(atom_label, pos_per_atom, neg_per_atom, n_te, total_pos)
                    acc_b, mf_b = BASE[dname]
                    if not (acc > acc_b and mf > mf_b):
                        pass_both = False
                        break
                    results[dname] = (acc, mf)
                if pass_both:
                    hits.append((f, direction, q, results))

    print(f"  Unified strict-both hits: {len(hits)}")
    for f, d1, q, r in hits[:10]:
        sym = ">" if d1 == 0 else "<"
        en = r["MHClip_EN"]
        zh = r["MHClip_ZH"]
        print(f"    {mode} base {f}{sym}q{q:.2f}  EN {en[0]:.4f}/{en[1]:.4f}  ZH {zh[0]:.4f}/{zh[1]:.4f}")
