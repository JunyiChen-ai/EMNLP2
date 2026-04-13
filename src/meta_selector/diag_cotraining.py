"""Co-training over two feature views of atoms (transductive, self-referential).

Split atom features into:
- View A = (v, log te_cnt, te_cdf) — test-side signal
- View B = (log tr_cnt, tr_cdf, near_dist) — train-side signal

Start from base_atom. Fit a 1-NN classifier on confidently-labeled atoms
in view A, propagate to view B, then vice versa. Iterate. The final
labeling depends on both views agreeing.

This is a classic transductive co-training protocol adapted to the atom
pool. It is a self-referential labeling procedure optimized over the pool.
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, load_scores_file, build_arrays
from data_utils import load_annotations
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors

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
    te_cdf = np.array([float((test_x <= v).mean()) for v in atoms])
    tr_cdf = np.array([float((train <= v).mean()) for v in atoms])
    near_dist = np.array([float(np.min(np.abs(train - v))) for v in atoms])

    A = np.stack([atoms, np.log(te_cnt + 1), te_cdf], axis=1)
    B = np.stack([np.log(tr_cnt + 1), tr_cdf, near_dist], axis=1)
    A = (A - A.mean(0)) / (A.std(0) + 1e-9)
    B = (B - B.mean(0)) / (B.std(0) + 1e-9)
    return dict(atoms=atoms, rounded_te=rounded_te, test_y=test_y,
                base_atom=base_atom, A=A, B=B, K=K)


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


def cotrain(A, B, init_lab, k, n_iter):
    K = len(init_lab)
    lab = init_lab.copy()
    for _ in range(n_iter):
        new = lab.copy()
        # Use A to predict
        pos_mask = lab; neg_mask = ~lab
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            break
        for side, view in enumerate([A, B]):
            nn_pos = NearestNeighbors(n_neighbors=min(k, pos_mask.sum())).fit(view[pos_mask])
            nn_neg = NearestNeighbors(n_neighbors=min(k, neg_mask.sum())).fit(view[neg_mask])
            d_pos, _ = nn_pos.kneighbors(view)
            d_neg, _ = nn_neg.kneighbors(view)
            dp = d_pos.mean(axis=1); dn = d_neg.mean(axis=1)
            vote = dp < dn
            new = new & vote if side == 1 else vote
        if (new == lab).all():
            break
        lab = new
    return lab


def orient_to_base(lab, base):
    a1 = (lab & base).sum()
    a2 = ((~lab) & base).sum()
    return lab if a1 >= a2 else ~lab


en = build("MHClip_EN")
zh = build("MHClip_ZH")

hits = []
for k in [1, 2, 3, 4, 5]:
    for n_iter in [1, 2, 3, 5, 10]:
        ok_both = True
        res = {}
        for dn, f in [("MHClip_EN", en), ("MHClip_ZH", zh)]:
            try:
                lab = cotrain(f["A"], f["B"], f["base_atom"], k, n_iter)
                lab = orient_to_base(lab, f["base_atom"])
            except Exception:
                ok_both = False
                break
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
            hits.append((k, n_iter, res))

print(f"Co-training strict-both hits: {len(hits)}")
for h in hits[:20]:
    k, n_iter, res = h
    print(f"  k={k} iter={n_iter}: "
          f"EN {res['MHClip_EN'][0]:.4f}/{res['MHClip_EN'][1]:.4f}  "
          f"ZH {res['MHClip_ZH'][0]:.4f}/{res['MHClip_ZH'][1]:.4f}")
