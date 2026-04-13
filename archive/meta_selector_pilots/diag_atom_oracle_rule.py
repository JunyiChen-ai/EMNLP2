"""Apply atom-level oracle (net sign per atom) to get sample labels, then
measure ACC/MF vs sample labels. This tells us the ceiling of atom-level
net-sign rules. Then check ZH base=POS subset's oracle labels and what
sparse label-free LR coefficients look like on its features.
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, load_scores_file, build_arrays
from data_utils import load_annotations
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
    acc_b, mf_b = BASE[d]
    t_base = otsu_threshold(test_x) if d == "MHClip_EN" else gmm_threshold(test_x)
    print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}  t_base={t_base:.4f}")

    rounded_te = np.round(test_x, 4)
    atoms_vals = sorted(set(rounded_te))

    # Atom net-majority oracle prediction
    atom_pred_net = {}
    atom_stats = {}
    for v in atoms_vals:
        mask = rounded_te == v
        pc = int(test_y[mask].sum())
        nc = int((1 - test_y[mask]).sum())
        atom_pred_net[v] = int(pc > nc)
        atom_stats[v] = (pc, nc)

    sample_pred_net = np.array([atom_pred_net[v] for v in rounded_te])
    acc_n, mf_n = eval_pred(test_y, sample_pred_net)
    print(f"  Atom net-majority -> sample labels: acc={acc_n:.4f} mf={mf_n:.4f}")

    # Atom > 0 (at least one positive)
    atom_pred_any = {v: int(atom_stats[v][0] >= 1) for v in atoms_vals}
    sample_pred_any = np.array([atom_pred_any[v] for v in rounded_te])
    acc_a, mf_a = eval_pred(test_y, sample_pred_any)
    print(f"  Atom any-positive -> sample labels: acc={acc_a:.4f} mf={mf_a:.4f}")

    # Atom pos_rate > 0.5 strict (ties go NEG)
    print(f"  Atom-level breakdown on base=POS subset:")
    for v in atoms_vals:
        mask = rounded_te == v
        if int(test_x[mask][0] >= t_base) == 1:
            pc, nc = atom_stats[v]
            net = "POS" if pc > nc else ("NEG" if pc < nc else "TIE")
            print(f"    v={v:.4f} p/n={pc}/{nc} net={net}")
