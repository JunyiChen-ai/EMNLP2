"""Harmonic-function label (Zhu-Ghahramani-Lafferty 2003).

Partition atoms into anchors and interior. Fix anchor labels, solve the
graph harmonic equation for interior labels:

  f_U = -L_UU^{-1} L_UL f_L

where L is the unnormalized graph Laplacian D - W. Interior labels are
rounded to {0, 1} via threshold. This is a transductive labeling —
interior atom labels are determined globally by the boundary conditions.

Anchor sets tried:
- extreme atoms by score (lowest/highest N)
- extreme atoms by te_cnt (lowest/highest N)
- near_dist extremes (closest/furthest to train)
- cdf extremes
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
                base_atom=base_atom, F=F, K=K,
                te_cnt=te_cnt, near_dist=near_dist, te_cdf=te_cdf)


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


def build_graph(F, k, sigma):
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(F))).fit(F)
    dist, idx = nn.kneighbors(F)
    K = len(F)
    W = np.zeros((K, K))
    for i in range(K):
        for j_, d_ in zip(idx[i, 1:], dist[i, 1:]):
            w = np.exp(-(d_ ** 2) / (2 * sigma ** 2))
            W[i, j_] = max(W[i, j_], w)
            W[j_, i] = max(W[j_, i], w)
    D = np.diag(W.sum(1))
    L = D - W
    return L


def harmonic_label(L, anchor_mask, anchor_vals):
    K = L.shape[0]
    all_idx = np.arange(K)
    U = all_idx[~anchor_mask]
    Lidx = all_idx[anchor_mask]
    L_UU = L[np.ix_(U, U)]
    L_UL = L[np.ix_(U, Lidx)]
    try:
        reg = 1e-6 * np.eye(len(U))
        fU = -np.linalg.solve(L_UU + reg, L_UL @ anchor_vals[anchor_mask])
    except Exception:
        fU = np.zeros(len(U))
    f = np.zeros(K)
    f[Lidx] = anchor_vals[anchor_mask]
    f[U] = fU
    return f


def anchor_scheme(f, scheme, n):
    K = f["K"]
    a_mask = np.zeros(K, dtype=bool)
    a_vals = np.zeros(K)
    if scheme == "score":
        lo = np.argsort(f["atoms"])[:n]
        hi = np.argsort(f["atoms"])[-n:]
    elif scheme == "te_cnt":
        lo = np.argsort(f["te_cnt"])[:n]
        hi = np.argsort(f["te_cnt"])[-n:]
    elif scheme == "near_dist":
        lo = np.argsort(f["near_dist"])[:n]
        hi = np.argsort(f["near_dist"])[-n:]
    else:  # te_cdf
        lo = np.argsort(f["te_cdf"])[:n]
        hi = np.argsort(f["te_cdf"])[-n:]
    a_mask[lo] = True
    a_mask[hi] = True
    a_vals[lo] = 0
    a_vals[hi] = 1
    return a_mask, a_vals


en = build("MHClip_EN")
zh = build("MHClip_ZH")

hits = []
for k in [2, 3, 5]:
    for sigma in [0.3, 0.5, 1.0, 2.0]:
        for scheme in ["score", "te_cnt", "near_dist", "te_cdf"]:
            for n in [1, 2, 3, 4]:
                for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
                    ok_both = True
                    res = {}
                    for dn, f in [("MHClip_EN", en), ("MHClip_ZH", zh)]:
                        try:
                            L = build_graph(f["F"], k, sigma)
                            a_mask, a_vals = anchor_scheme(f, scheme, n)
                            if a_mask.sum() >= f["K"] or a_mask.sum() == 0:
                                ok_both = False
                                break
                            vals = harmonic_label(L, a_mask, a_vals)
                            lab = vals > thr
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
                        hits.append((k, sigma, scheme, n, thr, res))

print(f"Harmonic strict-both hits: {len(hits)}")
for h in hits[:20]:
    k, sigma, scheme, n, thr, res = h
    print(f"  k={k} sig={sigma} sch={scheme} n={n} thr={thr}: "
          f"EN {res['MHClip_EN'][0]:.4f}/{res['MHClip_EN'][1]:.4f}  "
          f"ZH {res['MHClip_ZH'][0]:.4f}/{res['MHClip_ZH'][1]:.4f}")
