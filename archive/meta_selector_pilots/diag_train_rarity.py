"""Parameter-free label-free rules based on train rarity / train frequency.

Train is dominated by NEG class (train label distribution: ~80% NEG in
both datasets). So train count at atom v is roughly proportional to
P(NEG at v). Atoms with LOW train count are either (a) rare scores in
general, (b) over-represented in test POS.

Rules:
- flag iff tr_cnt < median(tr_cnt)
- flag iff tr_cnt < mean(tr_cnt)
- flag iff te_cnt > tr_cnt (test-dominant atom)
- flag iff tr_cnt / (te_cnt + tr_cnt) < 0.5 (test majority in pool)
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, load_scores_file, build_arrays
from data_utils import load_annotations
from sklearn.metrics import accuracy_score, f1_score

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

    t_base = otsu_threshold(test_x) if d == "MHClip_EN" else gmm_threshold(test_x)
    rounded_te = np.round(test_x, 4)
    rounded_tr = np.round(train, 4)
    atoms_vals = np.array(sorted(set(rounded_te)))
    base_atom = (atoms_vals >= t_base)

    te_cnt = np.array([int((rounded_te == v).sum()) for v in atoms_vals])
    tr_cnt = np.array([int((rounded_tr == v).sum()) for v in atoms_vals])
    test_share = te_cnt / (te_cnt + tr_cnt + 1e-6)

    rules = []
    rules.append(("tr_cnt<median", tr_cnt < np.median(tr_cnt)))
    rules.append(("tr_cnt<mean", tr_cnt < np.mean(tr_cnt)))
    rules.append(("te>tr", te_cnt > tr_cnt))
    rules.append(("te_share>0.5", test_share > 0.5))
    rules.append(("te_share>tr_share", test_share > (1 - test_share)))  # == >0.5
    # Baseline combinations
    for name, feat in list(rules):
        rules.append((f"base OR {name}", np.asarray(base_atom) | np.asarray(feat)))
        rules.append((f"base AND {name}", np.asarray(base_atom) & np.asarray(feat)))
        rules.append((f"base AND NOT {name}", np.asarray(base_atom) & ~np.asarray(feat)))
        rules.append((f"base OR NOT {name}", np.asarray(base_atom) | ~np.asarray(feat)))

    for name, atom_lab in rules:
        atom_lab = np.asarray(atom_lab)
        s = atom_lab.sum()
        if s == 0 or s == len(atom_lab):
            continue
        atom_map = dict(zip(atoms_vals, atom_lab.astype(int)))
        sp = np.array([atom_map[v] for v in rounded_te])
        acc, mf = eval_pred(test_y, sp)
        trans = int(np.sum(atom_lab[1:] != atom_lab[:-1]))
        strict = acc > acc_b and mf > mf_b
        tag = " PASS" if strict else ""
        print(f"  {name}: acc={acc:.4f} mf={mf:.4f} trans={trans}{tag}")
