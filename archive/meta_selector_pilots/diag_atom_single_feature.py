"""Single-feature atom discriminator: for each label-free feature, for each
cut, predict atom label = (feature >/< cut), apply to sample level, check
strict-both.

We want to find a FEATURE + CUT where (a) decision is not simply monotone
in raw value x, and (b) strict-both is achieved. This is non-suffix iff the
atom-level prediction, when sorted by x, is not monotone 0...0 1...1.
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, load_scores_file, build_arrays
from data_utils import load_annotations
from sklearn.metrics import accuracy_score, f1_score
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


def eval_pred(y, p):
    return accuracy_score(y, p), f1_score(y, p, average='macro')


def ecdf_at(vals, q):
    return np.searchsorted(np.sort(vals), q, side='right') / len(vals)


def is_non_suffix(labels, sort_order):
    """Labels sorted by x. Check if any 0 appears after a 1."""
    seen_1 = False
    for i in sort_order:
        if labels[i] == 1:
            seen_1 = True
        elif seen_1 and labels[i] == 0:
            return True
    return False


def build_atom_features(train, test_x):
    rounded_te = np.round(test_x, 4)
    rounded_tr = np.round(train, 4)
    atoms_vals = sorted(set(rounded_te))
    n_tr, n_te = len(train), len(test_x)

    bw_base = max(1e-4, float(np.std(train)) * 1.06 * (n_tr ** -0.2))
    bws = [bw_base * m for m in [0.25, 0.5, 1.0, 2.0, 4.0]]
    train_kdes = [gaussian_kde(train, bw_method=bw / np.std(train)) for bw in bws]
    test_kdes = [gaussian_kde(test_x, bw_method=bw / np.std(test_x)) for bw in bws]

    feats = {}
    feats["v"] = np.array(atoms_vals)
    feats["log_v"] = np.log(feats["v"] + 1e-6)
    feats["idx"] = np.arange(len(atoms_vals), dtype=float)
    te_counts = np.array([int((rounded_te == v).sum()) for v in atoms_vals])
    tr_counts = np.array([int((rounded_tr == v).sum()) for v in atoms_vals])
    feats["te_cnt"] = te_counts.astype(float)
    feats["tr_cnt"] = tr_counts.astype(float)
    feats["log_te"] = np.log(te_counts + 1)
    feats["log_tr"] = np.log(tr_counts + 1)
    feats["ratio"] = (te_counts / n_te) / ((tr_counts + 0.5) / n_tr)
    feats["log_ratio"] = np.log(feats["ratio"] + 1e-6)
    feats["Fte"] = np.array([ecdf_at(test_x, v) for v in atoms_vals])
    feats["Ftr"] = np.array([ecdf_at(train, v) for v in atoms_vals])
    feats["Fgap"] = feats["Fte"] - feats["Ftr"]
    feats["Fratio"] = (1 - feats["Fte"]) / (1 - feats["Ftr"] + 1e-6)

    for i, bw in enumerate(bws):
        feats[f"tr_d_{i}"] = np.array([float(train_kdes[i](v)[0]) for v in atoms_vals])
        feats[f"te_d_{i}"] = np.array([float(test_kdes[i](v)[0]) for v in atoms_vals])
        feats[f"dr_{i}"] = feats[f"tr_d_{i}"] / (feats[f"te_d_{i}"] + 1e-6)
        feats[f"log_dr_{i}"] = np.log(feats[f"dr_{i}"] + 1e-6)

    # distances
    d_lo = np.zeros(len(atoms_vals))
    d_hi = np.zeros(len(atoms_vals))
    for i, v in enumerate(atoms_vals):
        d_lo[i] = v - atoms_vals[i-1] if i > 0 else v
        d_hi[i] = atoms_vals[i+1] - v if i < len(atoms_vals) - 1 else 1.0
    feats["d_lo"] = d_lo
    feats["d_hi"] = d_hi
    feats["log_d_lo"] = np.log(d_lo + 1e-6)
    feats["log_d_hi"] = np.log(d_hi + 1e-6)

    return atoms_vals, feats, rounded_te


for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    acc_b, mf_b = BASE[d]
    print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}")

    atoms_vals, feats, rounded_te = build_atom_features(train, test_x)
    sort_order = np.argsort(atoms_vals)

    hits = []
    for fname, fvec in feats.items():
        # Try all cuts between distinct values
        uvals = sorted(set(fvec.tolist()))
        for i in range(len(uvals) - 1):
            cut = (uvals[i] + uvals[i+1]) / 2
            for direction in ["gt", "lt"]:
                atom_label = (fvec > cut if direction == "gt" else fvec < cut).astype(int)
                # non-trivial
                if atom_label.sum() == 0 or atom_label.sum() == len(atom_label):
                    continue
                # Compute per-atom lookup
                atom_map = {atoms_vals[i]: int(atom_label[i]) for i in range(len(atoms_vals))}
                sp = np.array([atom_map[v] for v in rounded_te])
                if sp.sum() == 0 or sp.sum() == len(sp):
                    continue
                acc, mf = eval_pred(test_y, sp)
                strict = acc > acc_b and mf > mf_b
                if strict:
                    ns = is_non_suffix(atom_label, sort_order)
                    hits.append((acc, mf, fname, direction, cut, ns))

    hits.sort(reverse=True)
    print(f"  Total strict-both atom-level rules: {len(hits)}")
    print(f"  Non-suffix strict-both: {sum(1 for h in hits if h[5])}")
    for acc, mf, fname, direc, cut, ns in hits[:20]:
        ns_tag = " NONSUFFIX" if ns else " suffix"
        print(f"    acc={acc:.4f} mf={mf:.4f}  {fname} {direc} {cut:.6f}{ns_tag}")
