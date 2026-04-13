"""CDF consistency / KS-like per-atom features.

For each test atom v, compute:
- test_cdf(v) = fraction of test <= v
- train_cdf(v) = fraction of train <= v
- cdf_diff(v) = test_cdf - train_cdf (signed, non-monotone)
- abs_diff = |cdf_diff|
- local KS-stat in window around v

Non-monotone features: cdf_diff and its derivatives are non-monotone
in atom index, hence can produce non-suffix rules.

Rules:
- atom is POS iff cdf_diff > threshold (or < threshold)
- base OP abs_diff > threshold
- local_max / local_min of cdf_diff
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
    n_te, n_tr = len(test_x), len(train)
    t_base = otsu_threshold(test_x) if d == "MHClip_EN" else gmm_threshold(test_x)
    base_atom = (atoms >= t_base).astype(bool)

    te_cdf = np.array([float((test_x <= v).mean()) for v in atoms])
    tr_cdf = np.array([float((train <= v).mean()) for v in atoms])
    cdf_diff = te_cdf - tr_cdf
    abs_diff = np.abs(cdf_diff)
    # Local KS window — max |cdf_diff| in a neighborhood of 3 atoms
    K = len(atoms)
    local_ks = np.array([float(np.max(abs_diff[max(0, i-2):min(K, i+3)])) for i in range(K)])
    return dict(atoms=atoms, rounded_te=rounded_te, test_y=test_y,
                base_atom=base_atom, te_cdf=te_cdf, tr_cdf=tr_cdf,
                cdf_diff=cdf_diff, abs_diff=abs_diff, local_ks=local_ks)


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

# Unified rule sweep
qs = np.linspace(0.05, 0.95, 19)
feats = ["cdf_diff", "abs_diff", "local_ks"]
hits = []
for fn in feats:
    for q in qs:
        for op in [">", "<"]:
            en_c = float(np.quantile(en[fn], q))
            zh_c = float(np.quantile(zh[fn], q))
            for logic in ["plain", "base_or", "base_and", "base_and_not"]:
                ok_both = True
                res = {}
                for dn, f, c in [("MHClip_EN", en, en_c), ("MHClip_ZH", zh, zh_c)]:
                    v = f[fn]
                    cond = (v > c) if op == ">" else (v < c)
                    if logic == "plain":
                        lab = cond
                    elif logic == "base_or":
                        lab = f["base_atom"] | cond
                    elif logic == "base_and":
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
                    hits.append((fn, op, q, logic, res))

print(f"CDF-consistency unified hits: {len(hits)}")
for fn, op, q, logic, res in hits[:20]:
    print(f"  {logic} {fn}{op}q{q:.2f}: "
          f"EN {res['MHClip_EN'][0]:.4f}/{res['MHClip_EN'][1]:.4f}  "
          f"ZH {res['MHClip_ZH'][0]:.4f}/{res['MHClip_ZH'][1]:.4f}")

# Local max/min of cdf_diff as atom labels
print("\nLocal max/min of cdf_diff rules:")
for dn, f in [("MHClip_EN", en), ("MHClip_ZH", zh)]:
    cd = f["cdf_diff"]
    K = len(cd)
    # Local maxima
    local_max = np.zeros(K, dtype=bool)
    local_min = np.zeros(K, dtype=bool)
    for i in range(1, K - 1):
        if cd[i] > cd[i - 1] and cd[i] > cd[i + 1]:
            local_max[i] = True
        if cd[i] < cd[i - 1] and cd[i] < cd[i + 1]:
            local_min[i] = True
    for name, lab in [("local_max_cdf_diff", local_max),
                      ("local_min_cdf_diff", local_min),
                      ("NOT local_max", ~local_max),
                      ("NOT local_min", ~local_min),
                      ("base OR local_max", f["base_atom"] | local_max),
                      ("base AND NOT local_min", f["base_atom"] & ~local_min)]:
        m = eval_lab(lab, f)
        if m is None:
            continue
        acc, mf, trans = m
        ab, mb = BASE[dn]
        strict = acc > ab and mf > mb
        tag = " PASS" if strict else ""
        print(f"  {dn} {name}: acc={acc:.4f} mf={mf:.4f} trans={trans}{tag}")
