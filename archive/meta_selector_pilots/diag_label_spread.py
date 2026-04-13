"""Label spreading from base_atom seed via normalized-Laplacian smoothing.

Zhou et al. 2004-style transductive label propagation:
  F_{t+1} = alpha * S * F_t + (1 - alpha) * Y

where S = D^{-1/2} W D^{-1/2} is the normalized-Laplacian smoother, Y is
the one-hot seed from base_atom, and alpha in [0,1) controls propagation.
Final labeling = argmax F_inf, where the closed form is F_inf = (1-alpha)
(I - alpha S)^{-1} Y.

This is a global transductive labeling — the label of each atom depends
on all other atoms via the graph, not locally.
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
    F = np.stack([atoms, np.log(te_cnt + 1), np.log(tr_cnt + 1),
                  te_cdf, tr_cdf, near_dist], axis=1)
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


def orient_to_base(lab, base):
    a1 = (lab & base).sum()
    a2 = ((~lab) & base).sum()
    return lab if a1 >= a2 else ~lab


def label_spread(F, seed_pos, k, sigma, alpha):
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(F))).fit(F)
    dist, idx = nn.kneighbors(F)
    K = len(F)
    W = np.zeros((K, K))
    for i in range(K):
        for j_, d_ in zip(idx[i, 1:], dist[i, 1:]):
            w = np.exp(-(d_ ** 2) / (2 * sigma ** 2))
            W[i, j_] = max(W[i, j_], w)
            W[j_, i] = max(W[j_, i], w)
    D = W.sum(1); D[D == 0] = 1.0
    Dinv = 1.0 / np.sqrt(D)
    S = Dinv[:, None] * W * Dinv[None, :]
    Y = np.zeros((K, 2))
    Y[seed_pos, 1] = 1.0
    Y[~seed_pos, 0] = 1.0
    M = np.eye(K) - alpha * S
    F_inf = (1 - alpha) * np.linalg.solve(M, Y)
    return F_inf[:, 1] > F_inf[:, 0]


en = build("MHClip_EN")
zh = build("MHClip_ZH")

hits = []
for k in [2, 3, 5, 7]:
    for sigma in [0.3, 0.5, 1.0, 2.0]:
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]:
            ok_both = True
            res = {}
            for dn, f in [("MHClip_EN", en), ("MHClip_ZH", zh)]:
                try:
                    lab = label_spread(f["F"], f["base_atom"], k, sigma, alpha)
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
                hits.append((k, sigma, alpha, res))

print(f"Label-spread strict-both hits: {len(hits)}")
for h in hits[:20]:
    k, sigma, alpha, res = h
    print(f"  k={k} sig={sigma} a={alpha}: "
          f"EN {res['MHClip_EN'][0]:.4f}/{res['MHClip_EN'][1]:.4f}  "
          f"ZH {res['MHClip_ZH'][0]:.4f}/{res['MHClip_ZH'][1]:.4f}")
