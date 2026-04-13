"""Multi-scale density ratio: use fine bw to detect local non-monotonicity,
coarse bw for global trend. Decision uses the SIGN of local-vs-global or
the RATIO fine/coarse.

Parameter-free rules tested:
- fine_dr > coarse_dr (local train-enriched relative to global trend)
- fine_dr / coarse_dr > 1
- fine_dr - coarse_dr > 0
- Base AND (condition) / Base OR (condition)
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


all_results = []

for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    acc_b, mf_b = BASE[d]
    n_tr, n_te = len(train), len(test_x)
    print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}")

    t_base = otsu_threshold(test_x) if d == "MHClip_EN" else gmm_threshold(test_x)
    rounded_te = np.round(test_x, 4)
    atoms_vals = np.array(sorted(set(rounded_te)))
    base_atom = (atoms_vals >= t_base).astype(int)

    bw_base = max(1e-4, float(np.std(train)) * 1.06 * (n_tr ** -0.2))

    # Fine and coarse bw's
    for fine_m, coarse_m in [(0.25, 2.0), (0.5, 2.0), (0.25, 4.0), (0.5, 4.0), (1.0, 4.0)]:
        bw_f = bw_base * fine_m
        bw_c = bw_base * coarse_m
        tr_kf = gaussian_kde(train, bw_method=bw_f / np.std(train))
        tr_kc = gaussian_kde(train, bw_method=bw_c / np.std(train))
        te_kf = gaussian_kde(test_x, bw_method=bw_f / np.std(test_x))
        te_kc = gaussian_kde(test_x, bw_method=bw_c / np.std(test_x))
        tr_df = np.array([float(tr_kf(v)[0]) for v in atoms_vals])
        tr_dc = np.array([float(tr_kc(v)[0]) for v in atoms_vals])
        te_df = np.array([float(te_kf(v)[0]) for v in atoms_vals])
        te_dc = np.array([float(te_kc(v)[0]) for v in atoms_vals])
        dr_f = tr_df / (te_df + 1e-6)
        dr_c = tr_dc / (te_dc + 1e-6)
        # Fine > Coarse means local train density is MORE than global trend
        # implying a local concentration of train samples (which could be mostly NEG)
        # Atoms where dr_f > dr_c AND base=POS -> potentially subtract
        # Atoms where dr_f < dr_c AND base=NEG -> potentially add

        # Rule 1: flag iff dr_f < dr_c (fine less train-enriched than global)
        atom_lab = (dr_f < dr_c)
        atom_map = dict(zip(atoms_vals, atom_lab.astype(int)))
        sp = np.array([atom_map[v] for v in rounded_te])
        if 0 < sp.sum() < n_te:
            acc, mf = eval_pred(test_y, sp)
            trans = int(np.sum(atom_lab[1:] != atom_lab[:-1]))
            strict = acc > acc_b and mf > mf_b
            tag = " PASS" if strict else ""
            print(f"  bw=({fine_m},{coarse_m}) dr_f<dr_c: acc={acc:.4f} mf={mf:.4f} trans={trans}{tag}")

        # Rule 2: base AND NOT(dr_f > dr_c) (subtract base where fine > coarse)
        atom_lab = base_atom.astype(bool) & (dr_f < dr_c)
        atom_map = dict(zip(atoms_vals, atom_lab.astype(int)))
        sp = np.array([atom_map[v] for v in rounded_te])
        if 0 < sp.sum() < n_te:
            acc, mf = eval_pred(test_y, sp)
            trans = int(np.sum(atom_lab[1:] != atom_lab[:-1]))
            strict = acc > acc_b and mf > mf_b
            tag = " PASS" if strict else ""
            print(f"  bw=({fine_m},{coarse_m}) base AND dr_f<dr_c: acc={acc:.4f} mf={mf:.4f} trans={trans}{tag}")

        # Rule 3: base OR (dr_f < dr_c AND in upper half)
        atom_lab = base_atom.astype(bool) | ((dr_f < dr_c) & (atoms_vals > np.median(atoms_vals)))
        atom_map = dict(zip(atoms_vals, atom_lab.astype(int)))
        sp = np.array([atom_map[v] for v in rounded_te])
        if 0 < sp.sum() < n_te:
            acc, mf = eval_pred(test_y, sp)
            trans = int(np.sum(atom_lab[1:] != atom_lab[:-1]))
            strict = acc > acc_b and mf > mf_b
            tag = " PASS" if strict else ""
            print(f"  bw=({fine_m},{coarse_m}) base OR (dr_f<dr_c & upper): acc={acc:.4f} mf={mf:.4f} trans={trans}{tag}")
