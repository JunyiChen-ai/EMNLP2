"""Graph Laplacian eigenmap label: transductive label from Fiedler vector.

Build kNN/RBF similarity graph over atom feature vectors. Compute the
normalized Laplacian. The Fiedler vector (2nd smallest eigenvector) gives
a graph-based relaxed label. Sign of Fiedler gives a binary labeling.
Higher-order eigenvectors (3rd, 4th) give alternate cuts.

This is a transductive label: the labeling is a function of the entire
pool's graph structure, not a per-atom closed-form statistic.
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


def laplacian_eig(F, k, sigma):
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(F))).fit(F)
    dist, idx = nn.kneighbors(F)
    K = len(F)
    W = np.zeros((K, K))
    for i in range(K):
        for j_, d_ in zip(idx[i, 1:], dist[i, 1:]):
            w = np.exp(-(d_ ** 2) / (2 * sigma ** 2))
            W[i, j_] = max(W[i, j_], w)
            W[j_, i] = max(W[j_, i], w)
    D = W.sum(1)
    D[D == 0] = 1.0
    Dinv = 1.0 / np.sqrt(D)
    L = np.eye(K) - (Dinv[:, None] * W * Dinv[None, :])
    vals, vecs = np.linalg.eigh(L)
    return vals, vecs


en = build("MHClip_EN")
zh = build("MHClip_ZH")

hits = []
for k in [2, 3, 5, 7]:
    for sigma in [0.3, 0.5, 1.0, 2.0]:
        for eig_idx in [1, 2, 3, 4]:
            for op in [">", "<"]:
                for thr_mode in ["zero", "median"]:
                    ok_both = True
                    res = {}
                    for dn, f in [("MHClip_EN", en), ("MHClip_ZH", zh)]:
                        try:
                            vals, vecs = laplacian_eig(f["F"], k, sigma)
                            if eig_idx >= vecs.shape[1]:
                                ok_both = False
                                break
                            v = vecs[:, eig_idx]
                        except Exception:
                            ok_both = False
                            break
                        c = 0.0 if thr_mode == "zero" else float(np.median(v))
                        lab = (v > c) if op == ">" else (v < c)
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
                        hits.append((k, sigma, eig_idx, op, thr_mode, res))

print(f"Laplacian-eigenmap strict-both hits: {len(hits)}")
for h in hits[:20]:
    k, sigma, ei, op, tm, res = h
    print(f"  k={k} sig={sigma} eig={ei} {op} thr={tm}: "
          f"EN {res['MHClip_EN'][0]:.4f}/{res['MHClip_EN'][1]:.4f}  "
          f"ZH {res['MHClip_ZH'][0]:.4f}/{res['MHClip_ZH'][1]:.4f}")
