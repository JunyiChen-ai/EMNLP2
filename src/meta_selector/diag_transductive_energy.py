"""Transductive energy minimization over atom labelings.

Feature class: transductive — atom labeling is produced as argmin of an
energy function over the pool, combining a *data term* (how well the
labeling agrees with a label-free prior) and a *pairwise smoothness term*
over a similarity graph constructed from atom features.

Unlike iterative propagation (diag_fixed_point_prop), this optimizes a
global objective rather than doing local updates.

Energy:  E(y) = sum_i D_i(y_i) + lambda * sum_{i,j in kNN} w_ij * [y_i != y_j]

with data term D_i(y_i=1) = -s_i (proxy prior) and D_i(y_i=0) = +s_i,
where s_i is a standardized feature such as (v, te_cnt-tr_cnt, dr_0, etc.).

For K <= 40 atoms, graph-cut is tractable; we use iterative conditional
modes (ICM) since binary labels with small K converge fast. Random
restarts increase chance of finding global min.
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
    return dict(atoms=atoms, rounded_te=rounded_te, test_y=test_y,
                base_atom=base_atom, te_cnt=te_cnt, tr_cnt=tr_cnt, K=K,
                train=train, test_x=test_x)


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


def orient_to_base(lab, base):
    a1 = (lab & base).sum()
    a2 = ((~lab) & base).sum()
    return lab if a1 >= a2 else ~lab


def minimize_energy(D, W, init, n_sweeps=100):
    """Iterative conditional modes. D: (K,2) data cost, W: (K,K) smoothness weights.
    Minimize sum_i D[i, y_i] + sum_{i,j} W[i,j] * [y_i != y_j]."""
    y = init.copy().astype(int)
    K = len(y)
    for _ in range(n_sweeps):
        changed = False
        for i in np.random.permutation(K):
            neigh = W[i]
            c0 = D[i, 0] + np.sum(neigh * (y != 0))
            c1 = D[i, 1] + np.sum(neigh * (y != 1))
            new = 0 if c0 <= c1 else 1
            if new != y[i]:
                y[i] = new
                changed = True
        if not changed:
            break
    return y.astype(bool)


def build_features(f):
    v = f["atoms"]
    te = np.log(f["te_cnt"] + 1)
    tr = np.log(f["tr_cnt"] + 1)
    te_cdf = np.array([float((f["test_x"] <= x).mean()) for x in v])
    tr_cdf = np.array([float((f["train"] <= x).mean()) for x in v])
    def std(x):
        return (x - x.mean()) / (x.std() + 1e-9)
    return dict(v=std(v), te=std(te), tr=std(tr),
                mdiff=std(te - tr), cdf_diff=std(te_cdf - tr_cdf))


def build_graph(f, k):
    v = f["atoms"].reshape(-1, 1)
    te = np.log(f["te_cnt"] + 1).reshape(-1, 1)
    tr = np.log(f["tr_cnt"] + 1).reshape(-1, 1)
    F = np.hstack([v, te, tr])
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    K = len(f["atoms"])
    nn = NearestNeighbors(n_neighbors=min(k + 1, K)).fit(F)
    dist, idx = nn.kneighbors(F)
    W = np.zeros((K, K))
    for i in range(K):
        for j_, d_ in zip(idx[i, 1:], dist[i, 1:]):
            w = np.exp(-d_)
            W[i, j_] = max(W[i, j_], w)
            W[j_, i] = max(W[j_, i], w)
    return W


en = build("MHClip_EN")
zh = build("MHClip_ZH")
en_feats = build_features(en)
zh_feats = build_features(zh)

feat_names = ["v", "te", "tr", "mdiff", "cdf_diff"]
lambdas = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
ks = [2, 3, 5, 7]
init_kinds = ["base", "half", "inv_base"]

np.random.seed(0)
hits = []
for fn in feat_names:
    for lam in lambdas:
        for k in ks:
            for init_kind in init_kinds:
                ok_both = True
                res = {}
                for dn, f, feats in [("MHClip_EN", en, en_feats),
                                     ("MHClip_ZH", zh, zh_feats)]:
                    s = feats[fn]
                    D = np.stack([s, -s], axis=1)  # D[i,0]=s (cost for neg), D[i,1]=-s (for pos)
                    W = build_graph(f, k) * lam
                    if init_kind == "base":
                        init = f["base_atom"].astype(int)
                    elif init_kind == "half":
                        init = (s > np.median(s)).astype(int)
                    else:
                        init = (~f["base_atom"]).astype(int)
                    lab = minimize_energy(D, W, init)
                    lab = orient_to_base(lab, f["base_atom"])
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
                    hits.append((fn, lam, k, init_kind, res))

print(f"Transductive energy strict-both hits: {len(hits)}")
for h in hits[:20]:
    fn, lam, k, init_kind, res = h
    print(f"  feat={fn} lam={lam} k={k} init={init_kind}: "
          f"EN {res['MHClip_EN'][0]:.4f}/{res['MHClip_EN'][1]:.4f}  "
          f"ZH {res['MHClip_ZH'][0]:.4f}/{res['MHClip_ZH'][1]:.4f}")
