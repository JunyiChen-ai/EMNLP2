"""Autoencoder reconstruction residual as a label-free atom feature.

Fit a tiny linear autoencoder (W, W^T) on the atom feature matrix F using
gradient descent, take reconstruction error per atom as feature, then:
- threshold via quantile
- combine with base_atom (OR/AND/ANDNOT)

This is a learned unsupervised representation (optimized over the pool),
distinct from PCA since we include sparsity / non-linearity and random
restarts. The residual is a fixed point of the autoencoder optimization.
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
    raw = np.stack([atoms, np.log(te_cnt + 1), np.log(tr_cnt + 1),
                    te_cdf, tr_cdf, near_dist], axis=1)
    F = (raw - raw.mean(0)) / (raw.std(0) + 1e-9)
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


def train_ae(F, latent, n_steps=2000, lr=0.05, l1=0.0, seed=0):
    rng = np.random.default_rng(seed)
    d = F.shape[1]
    W = rng.normal(0, 0.3, size=(d, latent))
    b1 = np.zeros(latent)
    b2 = np.zeros(d)
    for _ in range(n_steps):
        z = np.tanh(F @ W + b1)
        rec = z @ W.T + b2
        err = rec - F
        dW_dec = err.T @ z / F.shape[0]
        dz = err @ W * (1 - z ** 2)
        dW_enc = F.T @ dz / F.shape[0]
        dW = dW_dec + dW_enc
        db2 = err.mean(0)
        db1 = dz.mean(0)
        W -= lr * dW + l1 * np.sign(W) * lr
        b1 -= lr * db1
        b2 -= lr * db2
    z = np.tanh(F @ W + b1)
    rec = z @ W.T + b2
    residual = ((rec - F) ** 2).sum(axis=1)
    return residual, z


en = build("MHClip_EN")
zh = build("MHClip_ZH")

latents = [1, 2, 3, 4]
seeds = [0, 1, 2, 3]
qs = np.linspace(0.05, 0.95, 19)
logics = ["plain", "base_or", "base_and", "base_and_not"]
ops = [">", "<"]

hits = []
residuals = {}
for dn, f in [("MHClip_EN", en), ("MHClip_ZH", zh)]:
    residuals[dn] = {}
    for lat in latents:
        rs = []
        for sd in seeds:
            r, _ = train_ae(f["F"], lat, seed=sd)
            rs.append(r)
        residuals[dn][lat] = np.mean(rs, axis=0)

for lat in latents:
    for q in qs:
        for op in ops:
            for lg in logics:
                ok_both = True
                res = {}
                for dn, f in [("MHClip_EN", en), ("MHClip_ZH", zh)]:
                    v = residuals[dn][lat]
                    c = float(np.quantile(v, q))
                    cond = (v > c) if op == ">" else (v < c)
                    if lg == "plain":
                        lab = cond
                    elif lg == "base_or":
                        lab = f["base_atom"] | cond
                    elif lg == "base_and":
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
                    hits.append((lat, q, op, lg, res))

print(f"AE-residual strict-both hits: {len(hits)}")
for h in hits[:20]:
    lat, q, op, lg, res = h
    print(f"  lat={lat} q{q:.2f} {op} {lg}: "
          f"EN {res['MHClip_EN'][0]:.4f}/{res['MHClip_EN'][1]:.4f}  "
          f"ZH {res['MHClip_ZH'][0]:.4f}/{res['MHClip_ZH'][1]:.4f}")
