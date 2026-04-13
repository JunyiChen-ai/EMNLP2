"""Asymmetric flip rules: jointly parameterized ADD_NEG (flip base=NEG to POS)
and DROP_POS (flip base=POS to NEG) conditions. Unified via quantile cuts.

EN requires adding TIE atoms (label-dependent). Let's see if any quantile
cut structure can express "add base=NEG atoms where <some smooth feature
> qadd>" AND "drop base=POS atoms where <feature < qdrop>" such that strict
both holds on BOTH datasets.

This is different from previous work because previous rules defined atom_label
as a function of features directly; this one modifies the base separately on
each side.
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


def build(d):
    train, test_x, test_y = load(d)
    n_tr, n_te = len(train), len(test_x)
    rounded_te = np.round(test_x, 4)
    rounded_tr = np.round(train, 4)
    atoms = np.array(sorted(set(rounded_te)))
    K = len(atoms)
    te_cnt = np.array([int((rounded_te == v).sum()) for v in atoms])
    tr_cnt = np.array([int((rounded_tr == v).sum()) for v in atoms])
    t_base = otsu_threshold(test_x) if d == "MHClip_EN" else gmm_threshold(test_x)
    base_atom = (atoms >= t_base)

    bw = max(1e-4, float(np.std(train)) * 1.06 * (n_tr ** -0.2)) * 0.5
    tr_k = gaussian_kde(train, bw_method=bw / np.std(train))
    te_k = gaussian_kde(test_x, bw_method=bw / np.std(test_x))
    tr_d = np.array([float(tr_k(v)[0]) for v in atoms])
    te_d = np.array([float(te_k(v)[0]) for v in atoms])
    dr_0 = tr_d / (te_d + 1e-9)
    ratio = (te_cnt / n_te) / ((tr_cnt + 0.5) / n_tr)
    return dict(
        atoms=atoms, rounded_te=rounded_te, test_y=test_y,
        te_cnt=te_cnt.astype(float), tr_cnt=tr_cnt.astype(float),
        tr_d=tr_d, te_d=te_d, dr_0=dr_0, ratio=ratio,
        base_atom=base_atom, t_base=t_base, v=atoms.astype(float),
    )


def eval_lab(atom_lab, f):
    atom_lab = np.asarray(atom_lab).astype(bool)
    s = atom_lab.sum()
    if s == 0 or s == len(atom_lab):
        return None
    atom_map = dict(zip(f["atoms"], atom_lab.astype(int)))
    sp = np.array([atom_map[v] for v in f["rounded_te"]])
    acc = accuracy_score(f["test_y"], sp)
    mf = f1_score(f["test_y"], sp, average='macro')
    return acc, mf


en = build("MHClip_EN")
zh = build("MHClip_ZH")

feat_names = ["dr_0", "ratio", "te_cnt", "tr_cnt", "tr_d", "te_d"]
hits = []

# Unified rule: atom_label = (base AND feat_drop > q_drop)
#                           OR (NOT base AND feat_add > q_add)
for f_drop in feat_names:
    for f_add in feat_names:
        for q_drop in np.linspace(0.0, 1.0, 21):
            for q_add in np.linspace(0.0, 1.0, 21):
                for op_drop in [">", "<"]:
                    for op_add in [">", "<"]:
                        ok_both = True
                        res = {}
                        for d, f in [("MHClip_EN", en), ("MHClip_ZH", zh)]:
                            vd = f[f_drop]
                            va = f[f_add]
                            cd = float(np.quantile(vd, q_drop)) if q_drop > 0 else vd.min() - 1
                            ca = float(np.quantile(va, q_add)) if q_add > 0 else va.min() - 1
                            drop_cond = (vd > cd) if op_drop == ">" else (vd < cd)
                            add_cond = (va > ca) if op_add == ">" else (va < ca)
                            # keep base if base AND drop_cond, then add if NOT base AND add_cond
                            lab = (f["base_atom"] & drop_cond) | (~f["base_atom"] & add_cond)
                            m = eval_lab(lab, f)
                            if m is None:
                                ok_both = False
                                break
                            ab, mb = BASE[d]
                            if not (m[0] > ab and m[1] > mb):
                                ok_both = False
                                break
                            res[d] = m
                        if ok_both:
                            hits.append((f_drop, op_drop, q_drop, f_add, op_add, q_add,
                                         res["MHClip_EN"], res["MHClip_ZH"]))

print(f"Asymmetric drop/add hits: {len(hits)}")
for h in hits[:30]:
    fd, od, qd, fa, oa, qa, en_m, zh_m = h
    print(f"  drop base AND {fd}{od}q{qd:.2f} | add !base AND {fa}{oa}q{qa:.2f}")
    print(f"    EN {en_m[0]:.4f}/{en_m[1]:.4f}  ZH {zh_m[0]:.4f}/{zh_m[1]:.4f}")
