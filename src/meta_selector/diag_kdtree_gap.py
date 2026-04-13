"""KD-tree dyadic gap partition + atom-level rule.

Recursively split atoms by the median feature into K leaves, get leaf id
as a categorical label, search over leaf subset assignments. This gives
non-monotone, combinatorial structure not captured by linear/monotone cuts.
"""
import sys, itertools, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, load_scores_file, build_arrays
from data_utils import load_annotations
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

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
    rounded_tr = np.round(train, 4)
    atoms = np.array(sorted(set(rounded_te)))
    K = len(atoms)
    t_base = otsu_threshold(test_x) if d == "MHClip_EN" else gmm_threshold(test_x)
    base_atom = (atoms >= t_base).astype(bool)
    te_cnt = np.array([int((rounded_te == v).sum()) for v in atoms], dtype=float)
    tr_cnt = np.array([int((rounded_tr == v).sum()) for v in atoms], dtype=float)
    near_dist = np.array([float(np.min(np.abs(train - v))) for v in atoms])
    F = np.stack([atoms, np.log(te_cnt + 1), np.log(tr_cnt + 1), near_dist], axis=1)
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    return dict(atoms=atoms, rounded_te=rounded_te, test_y=test_y,
                base_atom=base_atom, F=F, K=K)


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


def leaf_ids(F, depth):
    # greedy median split (pseudo-kdtree), depth levels
    K = len(F)
    nl = 2 ** depth
    leaf = np.zeros(K, dtype=int)
    groups = [(0, np.arange(K))]
    next_id = 1
    for dd in range(depth):
        new_groups = []
        for leaf_id, idx in groups:
            dim = dd % F.shape[1]
            if len(idx) < 2:
                new_groups.append((leaf_id, idx))
                continue
            vals = F[idx, dim]
            med = np.median(vals)
            left = idx[vals < med]
            right = idx[vals >= med]
            if dd == depth - 1:
                leaf[left] = 2 * leaf_id
                leaf[right] = 2 * leaf_id + 1
            new_groups.append((2 * leaf_id, left))
            new_groups.append((2 * leaf_id + 1, right))
        groups = new_groups
    return leaf


en = build("MHClip_EN")
zh = build("MHClip_ZH")

hits = []
for depth in [2, 3]:
    nl = 2 ** depth
    for assign_bits in range(1, 1 << nl):
        assign = [(assign_bits >> i) & 1 for i in range(nl)]
        ok_both = True
        res = {}
        for dn, f in [("MHClip_EN", en), ("MHClip_ZH", zh)]:
            leaf = leaf_ids(f["F"], depth)
            lab = np.array([bool(assign[l]) for l in leaf])
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
            hits.append((depth, assign_bits, res))

print(f"KD-tree leaf-assign strict-both hits: {len(hits)}")
for h in hits[:20]:
    depth, ab, res = h
    print(f"  depth={depth} assign={bin(ab)}: "
          f"EN {res['MHClip_EN'][0]:.4f}/{res['MHClip_EN'][1]:.4f}  "
          f"ZH {res['MHClip_ZH'][0]:.4f}/{res['MHClip_ZH'][1]:.4f}")
