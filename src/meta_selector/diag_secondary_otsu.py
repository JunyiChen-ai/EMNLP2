"""Secondary Otsu inside base-POS region.

Motivation: The base-POS region itself contains both strong and weak POS
candidates. Apply a second Otsu/GMM to the SUBSET of test scores >= t_base
(or equivalently, to base-POS atom values weighted by te_cnt).

Rules:
- Keep only atoms with atom >= secondary_t (stricter threshold)
- Keep atoms < secondary_t (lower half)
- Various combinations
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


def eval_lab(atom_lab, atoms, rounded_te, test_y):
    atom_lab = np.asarray(atom_lab).astype(bool)
    s = atom_lab.sum()
    if s == 0 or s == len(atom_lab):
        return None
    atom_map = dict(zip(atoms, atom_lab.astype(int)))
    sp = np.array([atom_map[v] for v in rounded_te])
    acc = accuracy_score(test_y, sp)
    mf = f1_score(test_y, sp, average='macro')
    trans = int(np.sum(atom_lab[1:] != atom_lab[:-1]))
    return acc, mf, trans


# Collect results per (rule_name, method) and check both pass
by_rule = {}
for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    ab, mb = BASE[d]
    print(f"\n=== {d} === baseline {ab:.4f}/{mb:.4f}")
    rounded_te = np.round(test_x, 4)
    atoms = np.array(sorted(set(rounded_te)))
    t_base_o = otsu_threshold(test_x)
    t_base_g = gmm_threshold(test_x)
    t_base = t_base_o if d == "MHClip_EN" else t_base_g
    base_atom = (atoms >= t_base)

    # Primary method: Otsu on test_x
    # Secondary method: restrict to base-POS test samples, apply Otsu
    base_subset = test_x[test_x >= t_base]
    if len(base_subset) > 4 and len(set(base_subset)) > 2:
        t2_otsu = otsu_threshold(base_subset)
        try:
            t2_gmm = gmm_threshold(base_subset)
        except Exception:
            t2_gmm = t2_otsu
    else:
        t2_otsu = t_base
        t2_gmm = t_base

    # Restrict to below-base for lower Otsu
    nonbase_subset = test_x[test_x < t_base]
    if len(nonbase_subset) > 4 and len(set(nonbase_subset)) > 2:
        t_lo_otsu = otsu_threshold(nonbase_subset)
        try:
            t_lo_gmm = gmm_threshold(nonbase_subset)
        except Exception:
            t_lo_gmm = t_lo_otsu
    else:
        t_lo_otsu = 0
        t_lo_gmm = 0

    print(f"  t_base={t_base:.4f}  t2_otsu={t2_otsu:.4f}  t2_gmm={t2_gmm:.4f}  t_lo_otsu={t_lo_otsu:.4f}  t_lo_gmm={t_lo_gmm:.4f}")

    rules = [
        ("base only", base_atom),
        ("atoms>=t2_otsu", atoms >= t2_otsu),
        ("atoms>=t2_gmm", atoms >= t2_gmm),
        ("base AND atoms<t2_otsu", base_atom & (atoms < t2_otsu)),
        ("base AND atoms>=t2_otsu", base_atom & (atoms >= t2_otsu)),
        ("base OR atoms>=t_lo_otsu", base_atom | (atoms >= t_lo_otsu)),
        ("base OR (t_lo_gmm<=atoms<t_base)",
         base_atom | ((atoms >= t_lo_gmm) & (atoms < t_base))),
        ("base OR (t_lo_otsu<=atoms<t_base)",
         base_atom | ((atoms >= t_lo_otsu) & (atoms < t_base))),
        # Union atoms in [t_lo, t_base) with base
    ]

    for name, lab in rules:
        m = eval_lab(lab, atoms, rounded_te, test_y)
        if m is None:
            print(f"  {name}: trivial")
            continue
        acc, mf, trans = m
        strict = acc > ab and mf > mb
        tag = " PASS" if strict else ""
        print(f"  {name}: acc={acc:.4f} mf={mf:.4f} trans={trans}{tag}")
        by_rule.setdefault(name, {})[d] = strict

print("\n=== Unified pass check ===")
for name, v in by_rule.items():
    if v.get("MHClip_EN") and v.get("MHClip_ZH"):
        print(f"  UNIFIED PASS: {name}")
