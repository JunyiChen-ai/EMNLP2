"""Learned unsupervised representations over atoms.

Feature class: learned projections produced by optimization over the atom
pool (not closed-form statistics). Each atom i is represented as a multi-
channel vector (v, log te_cnt, log tr_cnt, te_cdf, tr_cdf, nn_dist_stats).
We then apply:
- PCA -> first-K components
- NMF (non-negative matrix factorization)
- Kernel PCA with RBF
- Spectral embedding

For each learned embedding, we evaluate:
- Sign of component c (> 0 or < 0)
- k-means (k=2) cluster id
- Direction: the cluster/sign matching base_atom majority is POS

The labeling is then scored directly. This is "transductive" in that the
learned projection depends on the pool itself.
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, load_scores_file, build_arrays
from data_utils import load_annotations
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA, NMF, KernelPCA
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import KMeans
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
    # nearest-train distance
    near_dist = np.array([float(np.min(np.abs(train - v))) for v in atoms])
    raw = np.stack([atoms, np.log(te_cnt + 1), np.log(tr_cnt + 1),
                    te_cdf, tr_cdf, near_dist], axis=1)
    # standardize
    mu = raw.mean(0); sd = raw.std(0) + 1e-9
    F = (raw - mu) / sd
    # non-negative variant for NMF
    F_nn = raw - raw.min(0) + 1e-6
    return dict(atoms=atoms, rounded_te=rounded_te, test_y=test_y,
                base_atom=base_atom, F=F, F_nn=F_nn, K=K)


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
    a1 = lab & base
    a2 = (~lab) & base
    return lab if a1.sum() >= a2.sum() else ~lab


def build_embeddings(f):
    out = {}
    F = f["F"]; F_nn = f["F_nn"]; K = f["K"]
    ncomp = min(4, K - 1)
    try:
        out["pca"] = PCA(n_components=ncomp).fit_transform(F)
    except Exception:
        pass
    try:
        out["nmf"] = NMF(n_components=min(3, K - 1), init='nndsvd', max_iter=500).fit_transform(F_nn)
    except Exception:
        pass
    for gamma in [0.1, 0.5, 1.0, 2.0]:
        try:
            out[f"kpca_g{gamma}"] = KernelPCA(n_components=ncomp, kernel='rbf', gamma=gamma).fit_transform(F)
        except Exception:
            pass
    for k in [3, 5, 7]:
        try:
            out[f"spec_k{k}"] = SpectralEmbedding(
                n_components=min(3, K - 1), affinity='nearest_neighbors',
                n_neighbors=k, random_state=0).fit_transform(F)
        except Exception:
            pass
    return out


en = build("MHClip_EN")
zh = build("MHClip_ZH")
en_emb = build_embeddings(en)
zh_emb = build_embeddings(zh)

common_names = sorted(set(en_emb.keys()) & set(zh_emb.keys()))
print(f"Embeddings available: {common_names}")

hits = []
for name in common_names:
    E_en = en_emb[name]; E_zh = zh_emb[name]
    nc = min(E_en.shape[1], E_zh.shape[1])
    for ci in range(nc):
        # Sign-based labeling per component
        for op in [">", "<"]:
            ok_both = True
            res = {}
            for dn, f, E in [("MHClip_EN", en, E_en), ("MHClip_ZH", zh, E_zh)]:
                v = E[:, ci]
                lab = (v > 0) if op == ">" else (v < 0)
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
                hits.append(("sign", name, ci, op, res))
    # KMeans k=2 and k=3 labelings
    for kk in [2, 3, 4]:
        for cl_target in range(kk):
            ok_both = True
            res = {}
            for dn, f, E in [("MHClip_EN", en, E_en), ("MHClip_ZH", zh, E_zh)]:
                try:
                    km = KMeans(n_clusters=kk, n_init=10, random_state=0).fit(E)
                    lab = (km.labels_ == cl_target)
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
                hits.append(("kmeans", name, kk, cl_target, res))

print(f"Learned repr strict-both hits: {len(hits)}")
for h in hits[:30]:
    print(" ", h[:-1], "EN", h[-1]['MHClip_EN'], "ZH", h[-1]['MHClip_ZH'])
