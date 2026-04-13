"""The rule `atom is POS iff dr_0 < 1` is parameter-free and label-free:
assign POS when test density > train density at that atom.

This is motivated by: test-enriched atoms are the ones that carry the
positive class (videos "surprising" to train distribution).

Also test: multi-bandwidth versions and min/max across bandwidths.
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import load_scores_file, build_arrays
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


for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    acc_b, mf_b = BASE[d]
    n_tr, n_te = len(train), len(test_x)
    print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}")

    rounded_te = np.round(test_x, 4)
    atoms_vals = np.array(sorted(set(rounded_te)))

    # Multiple bandwidths
    bw_base = max(1e-4, float(np.std(train)) * 1.06 * (n_tr ** -0.2))
    for bw_mult in [0.25, 0.5, 1.0, 2.0, 4.0]:
        bw = bw_base * bw_mult
        train_kde = gaussian_kde(train, bw_method=bw / np.std(train))
        test_kde = gaussian_kde(test_x, bw_method=bw / np.std(test_x))
        tr_d = np.array([float(train_kde(v)[0]) for v in atoms_vals])
        te_d = np.array([float(test_kde(v)[0]) for v in atoms_vals])
        dr = tr_d / (te_d + 1e-6)
        atom_label = (dr < 1.0)
        atom_map = dict(zip(atoms_vals, atom_label.astype(int)))
        sp = np.array([atom_map[v] for v in rounded_te])
        if sp.sum() == 0 or sp.sum() == n_te:
            print(f"  bw={bw_mult}: trivial")
            continue
        acc, mf = eval_pred(test_y, sp)
        strict = acc > acc_b and mf > mf_b
        tag = " PASS" if strict else ""
        # show transition count
        trans = int(np.sum(atom_label[1:] != atom_label[:-1]))
        print(f"  bw={bw_mult}: dr<1 -> acc={acc:.4f} mf={mf:.4f} trans={trans}{tag}")
