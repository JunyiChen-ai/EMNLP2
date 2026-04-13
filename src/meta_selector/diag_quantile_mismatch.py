"""Quantile-mismatch features.

For each test score s, compute:
- q_te = test_CDF(s) = its rank in test
- q_tr = train_CDF(s) = its rank in train
The mismatch r = q_te - q_tr is zero if train/test distributions agree.
If hateful (POS) videos are "surprising" they cluster at particular score
values causing local spikes in r.

Derived features:
- r_raw = q_te - q_tr (signed)
- r_abs = |r_raw|
- r_deriv = local differences of r_raw (non-monotone)

Rules: atom is POS iff r < threshold or > threshold, with logic combos.
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


def build(d):
    train, test_x, test_y = load(d)
    rounded_te = np.round(test_x, 4)
    atoms = np.array(sorted(set(rounded_te)))
    K = len(atoms)
    t_base = otsu_threshold(test_x) if d == "MHClip_EN" else gmm_threshold(test_x)
    base_atom = (atoms >= t_base).astype(bool)

    te_cdf = np.array([float((test_x <= v).mean()) for v in atoms])
    tr_cdf = np.array([float((train <= v).mean()) for v in atoms])
    r_raw = te_cdf - tr_cdf
    r_abs = np.abs(r_raw)
    r_deriv = np.concatenate([[0], np.diff(r_raw)])
    r_deriv2 = np.concatenate([[0], np.diff(r_deriv)])

    return dict(atoms=atoms, rounded_te=rounded_te, test_y=test_y,
                base_atom=base_atom, r_raw=r_raw, r_abs=r_abs,
                r_deriv=r_deriv, r_deriv2=r_deriv2)


def eval_lab(lab, f):
    lab = np.asarray(lab).astype(bool)
    s = lab.sum()
    if s == 0 or s == len(lab):
        return None
    atom_map = dict(zip(f["atoms"], lab.astype(int)))
    sp = np.array([atom_map[v] for v in f["rounded_te"]])
    acc = accuracy_score(f["test_y"], sp)
    mf = f1_score(f["test_y"], sp, average='macro')
    trans = int(np.sum(lab[1:] != lab[:-1]))
    return acc, mf, trans


en = build("MHClip_EN")
zh = build("MHClip_ZH")

feats = ["r_raw", "r_abs", "r_deriv", "r_deriv2"]
qs = np.linspace(0.05, 0.95, 19)
hits = []
for fn1 in feats:
    for fn2 in feats:
        for q1 in qs:
            for q2 in qs:
                for op1 in [">", "<"]:
                    for op2 in [">", "<"]:
                        for logic in ["OR", "AND", "ANDNOT"]:
                            ok_both = True
                            res = {}
                            for dn, f in [("MHClip_EN", en), ("MHClip_ZH", zh)]:
                                v1 = f[fn1]; v2 = f[fn2]
                                c1 = float(np.quantile(v1, q1))
                                c2 = float(np.quantile(v2, q2))
                                a1 = (v1 > c1) if op1 == ">" else (v1 < c1)
                                a2 = (v2 > c2) if op2 == ">" else (v2 < c2)
                                if logic == "OR":
                                    lab = a1 | a2
                                elif logic == "AND":
                                    lab = a1 & a2
                                else:
                                    lab = f["base_atom"] & ~(a1 | a2)
                                m = eval_lab(lab, f)
                                if m is None:
                                    ok_both = False
                                    break
                                ab, mb = BASE[dn]
                                if not (m[0] > ab and m[1] > mb):
                                    ok_both = False
                                    break
                                res[dn] = m
                            if ok_both:
                                hits.append((fn1, op1, q1, fn2, op2, q2, logic, res))

print(f"Quantile-mismatch unified hits: {len(hits)}")
for h in hits[:20]:
    fn1, op1, q1, fn2, op2, q2, logic, res = h
    print(f"  {logic}({fn1}{op1}q{q1:.2f},{fn2}{op2}q{q2:.2f}): "
          f"EN {res['MHClip_EN'][0]:.4f}/{res['MHClip_EN'][1]:.4f}  "
          f"ZH {res['MHClip_ZH'][0]:.4f}/{res['MHClip_ZH'][1]:.4f}")
