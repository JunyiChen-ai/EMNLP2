"""Cross-dataset alignment: use the OTHER dataset's pool as a reference
for the current dataset's feature derivation.

Scenario: compute for each EN atom v its rank within the ZH train pool,
and vice versa. Then combine with own-pool features. The hypothesis is
that hateful atoms may be anomalous *in both* distributions, while
dataset-specific nuisance variation appears only in one pool.

Allowed under constraints: uses only the 4 in-scope JSONL files.
Cross-pool reference is not an external dataset — both are in-scope.
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


def build(d, other_train, other_test_x):
    train, test_x, test_y = load(d)
    rounded_te = np.round(test_x, 4)
    rounded_tr = np.round(train, 4)
    atoms = np.array(sorted(set(rounded_te)))
    K = len(atoms)
    t_base = otsu_threshold(test_x) if d == "MHClip_EN" else gmm_threshold(test_x)
    base_atom = (atoms >= t_base).astype(bool)

    # Own-pool features
    te_cdf = np.array([float((test_x <= v).mean()) for v in atoms])
    tr_cdf = np.array([float((train <= v).mean()) for v in atoms])
    # Other-pool features (same v, but ranked in the other pool)
    other_tr_cdf = np.array([float((other_train <= v).mean()) for v in atoms])
    other_te_cdf = np.array([float((other_test_x <= v).mean()) for v in atoms])

    # Cross-pool features
    cross_tr_diff = tr_cdf - other_tr_cdf
    cross_te_diff = te_cdf - other_te_cdf
    cross_sym = 0.5 * (tr_cdf + other_tr_cdf)
    cross_asym = np.abs(tr_cdf - other_tr_cdf)
    cross_ratio = (te_cdf + 1e-6) / (other_te_cdf + 1e-6)

    return dict(atoms=atoms, rounded_te=rounded_te, test_y=test_y,
                base_atom=base_atom,
                cross_tr_diff=cross_tr_diff, cross_te_diff=cross_te_diff,
                cross_sym=cross_sym, cross_asym=cross_asym, cross_ratio=cross_ratio,
                own_te_cdf=te_cdf, own_tr_cdf=tr_cdf)


def eval_lab(lab, f):
    lab = np.asarray(lab).astype(bool)
    s = lab.sum()
    if s == 0 or s == len(lab):
        return None
    atom_map = dict(zip(f["atoms"], lab.astype(int)))
    sp = np.array([atom_map[v] for v in f["rounded_te"]])
    acc = accuracy_score(f["test_y"], sp)
    mf = f1_score(f["test_y"], sp, average='macro')
    return acc, mf


# Load both
en_train, en_test_x, en_test_y = load("MHClip_EN")
zh_train, zh_test_x, zh_test_y = load("MHClip_ZH")

en = build("MHClip_EN", zh_train, zh_test_x)
zh = build("MHClip_ZH", en_train, en_test_x)

feat_names = ["cross_tr_diff", "cross_te_diff", "cross_sym", "cross_asym", "cross_ratio"]
qs = np.linspace(0.05, 0.95, 19)
logics = ["plain", "base_or", "base_and", "base_and_not"]
ops = [">", "<"]

hits = []
for fn in feat_names:
    for q in qs:
        for op in ops:
            for lg in logics:
                ok_both = True
                res = {}
                for dn, f in [("MHClip_EN", en), ("MHClip_ZH", zh)]:
                    v = f[fn]
                    c = float(np.quantile(v, q))
                    cond = (v > c) if op == ">" else (v < c)
                    if lg == "plain":
                        lab = cond
                    elif lg == "base_or":
                        lab = f["base_atom"] | cond
                    elif lg == "base_and":
                        lab = f["base_atom"] & cond
                    else:
                        lab = f["base_atom"] & ~cond
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
                    hits.append((fn, q, op, lg, res))

# 2-feature AND/OR
hits2 = []
for fn1 in feat_names:
    for fn2 in feat_names:
        for q1 in qs:
            for q2 in qs:
                for op1 in ops:
                    for op2 in ops:
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
                                hits2.append((fn1, op1, q1, fn2, op2, q2, logic, res))

print(f"Cross-dataset 1-feat hits: {len(hits)}")
for h in hits[:10]:
    print(" ", h[:-1], "EN", h[-1]['MHClip_EN'], "ZH", h[-1]['MHClip_ZH'])
print(f"Cross-dataset 2-feat hits: {len(hits2)}")
for h in hits2[:10]:
    print(" ", h[:-1], "EN", h[-1]['MHClip_EN'], "ZH", h[-1]['MHClip_ZH'])
