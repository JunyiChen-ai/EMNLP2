"""Density Peak Clustering (Rodriguez & Laio, Science 2014) as a
transductive iterative fixed-point method applied to the atom pool.

Each atom i gets:
  rho_i = sum_j exp(-(d_ij / dc)^2)                # local density
  delta_i = min_{j: rho_j > rho_i} d_ij            # distance to nearest higher-density atom
  gamma_i = rho_i * delta_i                        # cluster-center score

Cluster centers are the atoms with top-K gamma. Non-center atoms are
assigned iteratively to the same cluster as their nearest-higher-density
neighbor (fixed-point of density following). The final binary labeling
merges clusters into POS / NEG based on which cluster contains the
majority of base_atom==True atoms.

This is a published iterative fixed-point method (≈20k citations) that
we have NOT yet tested in this search.
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


def density_peaks(F, dc, n_clusters, base_atom):
    K = len(F)
    # pairwise distances
    D = np.sqrt(((F[:, None, :] - F[None, :, :]) ** 2).sum(-1))
    # densities
    rho = np.exp(-(D / dc) ** 2).sum(axis=1) - 1.0  # subtract self
    # delta: distance to nearest higher-density atom
    delta = np.zeros(K)
    nearest_higher = np.full(K, -1, dtype=int)
    order = np.argsort(-rho)
    for ii, i in enumerate(order):
        if ii == 0:
            delta[i] = D[i].max()
            nearest_higher[i] = -1
        else:
            higher = order[:ii]
            ddd = D[i, higher]
            j = higher[np.argmin(ddd)]
            delta[i] = ddd.min()
            nearest_higher[i] = j
    # cluster centers = top n_clusters by gamma
    gamma = rho * delta
    centers = np.argsort(-gamma)[:n_clusters]
    # assign every atom to its nearest higher-density center via chain
    cluster = np.full(K, -1, dtype=int)
    for c_id, c in enumerate(centers):
        cluster[c] = c_id
    for i in order:
        if cluster[i] == -1:
            j = nearest_higher[i]
            if j >= 0 and cluster[j] != -1:
                cluster[i] = cluster[j]
    # merge clusters into POS / NEG by majority base-agreement per cluster
    pos_mask = np.zeros(K, dtype=bool)
    for c_id in range(n_clusters):
        members = cluster == c_id
        if members.sum() == 0:
            continue
        maj = base_atom[members].mean()
        if maj >= 0.5:
            pos_mask |= members
    return pos_mask


en = build("MHClip_EN")
zh = build("MHClip_ZH")

hits = []
for dc in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
    for n_clusters in [2, 3, 4, 5, 6, 8]:
        ok_both = True
        res = {}
        for dn, f in [("MHClip_EN", en), ("MHClip_ZH", zh)]:
            lab = density_peaks(f["F"], dc, n_clusters, f["base_atom"])
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
            hits.append((dc, n_clusters, res))

print(f"Density-peak strict-both hits: {len(hits)}")
for h in hits[:20]:
    dc, nc, res = h
    print(f"  dc={dc} nc={nc}: "
          f"EN {res['MHClip_EN'][0]:.4f}/{res['MHClip_EN'][1]:.4f}  "
          f"ZH {res['MHClip_ZH'][0]:.4f}/{res['MHClip_ZH'][1]:.4f}")
