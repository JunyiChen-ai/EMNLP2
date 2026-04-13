"""Two-feature atom-level rule (vectorized): atom_label = (f1 op1 c1) OP (f2 op2 c2).

Vectorized over atoms using direct label counts per atom. For each atom
labeling, compute sample-level ACC and MF analytically from atom pos/neg
counts (no per-sample rebuild).
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

    # Per-atom label counts
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
    """Atom_label: binary vector of length n_atoms.
    Returns (acc, macro_f1).
    """
    tp = int((atom_label * pos_per_atom).sum())
    fp = int((atom_label * neg_per_atom).sum())
    fn = int(total_pos - tp)
    tn = int(n_te - total_pos - fp)
    acc = (tp + tn) / n_te
    # macro-F1 over {POS, NEG}
    p1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p1 * r1 / (p1 + r1) if (p1 + r1) > 0 else 0.0
    p0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    r0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f0 = 2 * p0 * r0 / (p0 + r0) if (p0 + r0) > 0 else 0.0
    mf = (f1 + f0) / 2
    return acc, mf


def feature_cuts(fvec):
    uvals = sorted(set(fvec.tolist()))
    return np.array([(uvals[i] + uvals[i+1]) / 2 for i in range(len(uvals) - 1)])


for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    acc_b, mf_b = BASE[d]
    print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}", flush=True)

    atoms_vals, feats, pos_per_atom, neg_per_atom, n_te = build_atom_features(train, test_x, test_y)
    total_pos = int(pos_per_atom.sum())
    fnames = list(feats.keys())
    print(f"  features: {len(fnames)}, atoms: {len(atoms_vals)}", flush=True)

    n_hits = 0
    best_rules = []
    for i, f1 in enumerate(fnames):
        cuts1 = feature_cuts(feats[f1])
        if len(cuts1) == 0:
            continue
        # keep all cuts (fine)
        for j in range(i, len(fnames)):
            f2 = fnames[j]
            cuts2 = feature_cuts(feats[f2])
            if len(cuts2) == 0:
                continue
            for c1 in cuts1:
                for c2 in cuts2:
                    for d1 in [0, 1]:
                        a1 = (feats[f1] > c1) if d1 == 0 else (feats[f1] < c1)
                        for d2 in [0, 1]:
                            a2 = (feats[f2] > c2) if d2 == 0 else (feats[f2] < c2)
                            for op_name, op in [("AND", a1 & a2), ("OR", a1 | a2)]:
                                atom_label = op.astype(int)
                                s = atom_label.sum()
                                if s == 0 or s == len(atom_label):
                                    continue
                                acc, mf = compute_metrics(atom_label, pos_per_atom, neg_per_atom, n_te, total_pos)
                                if acc > acc_b and mf > mf_b:
                                    n_hits += 1
                                    if len(best_rules) < 100 or mf > best_rules[-1][1]:
                                        best_rules.append((acc, mf, f1, d1, c1, op_name, f2, d2, c2))
                                        best_rules.sort(key=lambda r: (-r[0], -r[1]))
                                        best_rules = best_rules[:100]

    print(f"  Two-feature strict-both hits: {n_hits}", flush=True)
    for rule in best_rules[:15]:
        acc, mf, f1, d1, c1, op, f2, d2, c2 = rule
        s1 = ">" if d1 == 0 else "<"
        s2 = ">" if d2 == 0 else "<"
        print(f"    acc={acc:.4f} mf={mf:.4f}  ({f1}{s1}{c1:.4f}) {op} ({f2}{s2}{c2:.4f})")
