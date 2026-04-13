"""Cross-check: apply EN-best rule to ZH and vice versa to see where the
decision surface actually lands on the other dataset.

From diag_atom_pair_feature (job 8090):
- EN rule (one of 13): (v>0.3491) OR (dr_0<0.8795) -> EN 0.7702/0.6842
- ZH rule (one of 93): (v>0.0333) AND (ratio>0.4328) -> ZH 0.8322/0.8068

Apply each rule to the OTHER dataset and compute the result.
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


for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    n_tr, n_te = len(train), len(test_x)
    print(f"\n=== {d} ===")

    rounded_te = np.round(test_x, 4)
    rounded_tr = np.round(train, 4)
    atoms_vals = np.array(sorted(set(rounded_te)))

    te_counts = np.array([int((rounded_te == v).sum()) for v in atoms_vals])
    tr_counts = np.array([int((rounded_tr == v).sum()) for v in atoms_vals])
    ratio = (te_counts / n_te) / ((tr_counts + 0.5) / n_tr)

    bw = max(1e-4, float(np.std(train)) * 1.06 * (n_tr ** -0.2)) * 0.5
    train_kde = gaussian_kde(train, bw_method=bw / np.std(train))
    test_kde = gaussian_kde(test_x, bw_method=bw / np.std(test_x))
    tr_d_0 = np.array([float(train_kde(v)[0]) for v in atoms_vals])
    te_d_0 = np.array([float(test_kde(v)[0]) for v in atoms_vals])
    dr_0 = tr_d_0 / (te_d_0 + 1e-6)

    # EN-winner rule: (v > 0.3491) OR (dr_0 < 0.8795)
    en_rule = (atoms_vals > 0.3491) | (dr_0 < 0.8795)
    # ZH-winner rule: (v > 0.0333) AND (ratio > 0.4328)
    zh_rule = (atoms_vals > 0.0333) & (ratio > 0.4328)

    # Apply to sample level
    for rule_name, atom_rule in [("EN-rule", en_rule), ("ZH-rule", zh_rule)]:
        atom_map = dict(zip(atoms_vals, atom_rule.astype(int)))
        sp = np.array([atom_map[v] for v in rounded_te])
        if sp.sum() == 0 or sp.sum() == n_te:
            print(f"  {rule_name}: trivial (all {sp[0]})")
            continue
        acc, mf = eval_pred(test_y, sp)
        acc_b, mf_b = BASE[d]
        strict = acc > acc_b and mf > mf_b
        tag = " PASS" if strict else ""
        print(f"  {rule_name}: acc={acc:.4f} mf={mf:.4f}{tag}")

    # ALSO: intersection and union of both
    inter = en_rule & zh_rule
    union = en_rule | zh_rule
    for rule_name, atom_rule in [("EN AND ZH", inter), ("EN OR ZH", union)]:
        atom_map = dict(zip(atoms_vals, atom_rule.astype(int)))
        sp = np.array([atom_map[v] for v in rounded_te])
        if sp.sum() == 0 or sp.sum() == n_te:
            print(f"  {rule_name}: trivial")
            continue
        acc, mf = eval_pred(test_y, sp)
        acc_b, mf_b = BASE[d]
        strict = acc > acc_b and mf > mf_b
        tag = " PASS" if strict else ""
        print(f"  {rule_name}: acc={acc:.4f} mf={mf:.4f}{tag}")
