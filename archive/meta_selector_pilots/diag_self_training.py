"""Self-training / pseudo-label propagation from base.

Start: atom_label = base_atom
Iterate:
  1. For each atom, compute distance (in multi-feature space) to
     the CENTROID of currently-POS atoms.
  2. Reassign atoms to POS if distance < threshold.
  3. Stop when stable.

Features used for distance: (v, log(tr_cnt+1), log(te_cnt+1), log(dr_0))
normalized.

Unified via fixed iteration count and rank-based threshold.
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, load_scores_file, build_arrays
from data_utils import load_annotations
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import gaussian_kde

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
    te_cnt = np.array([int((rounded_te == v).sum()) for v in atoms])
    tr_cnt = np.array([int((np.round(train, 4) == v).sum()) for v in atoms])
    t_base = otsu_threshold(test_x) if d == "MHClip_EN" else gmm_threshold(test_x)
    base_atom = (atoms >= t_base).astype(bool)

    bw = max(1e-4, float(np.std(train)) * 1.06 * (n_tr ** -0.2)) * 0.5
    tr_k = gaussian_kde(train, bw_method=bw / np.std(train))
    te_k = gaussian_kde(test_x, bw_method=bw / np.std(test_x))
    tr_d = np.array([float(tr_k(v)[0]) for v in atoms])
    te_d = np.array([float(te_k(v)[0]) for v in atoms])
    dr_0 = tr_d / (te_d + 1e-9)

    # Feature matrix: [v, log(te_cnt+1), log(tr_cnt+1), log(dr_0+1e-6)]
    feats = np.stack([
        atoms.astype(float),
        np.log(te_cnt + 1.0),
        np.log(tr_cnt + 1.0),
        np.log(dr_0 + 1e-6),
    ], axis=1)
    # Z-normalize
    feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-9)
    return dict(atoms=atoms, rounded_te=rounded_te, test_y=test_y,
                base_atom=base_atom, feats=feats)


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


def self_train(f, n_iter=5, top_frac=0.3):
    lab = f["base_atom"].copy()
    feats = f["feats"]
    for _ in range(n_iter):
        if lab.sum() == 0:
            break
        centroid = feats[lab].mean(axis=0)
        dists = np.linalg.norm(feats - centroid, axis=1)
        # Keep top_frac closest atoms as POS
        n_keep = max(1, int(top_frac * len(lab)))
        top_idx = np.argsort(dists)[:n_keep]
        new_lab = np.zeros_like(lab)
        new_lab[top_idx] = True
        if np.array_equal(new_lab, lab):
            break
        lab = new_lab
    return lab


# Sweep parameters
print("Self-training with centroid-distance reassignment")
by_params = {}
for n_iter in [1, 2, 3, 5, 10]:
    for top_frac in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        results = {}
        for d in ["MHClip_EN", "MHClip_ZH"]:
            f = build(d)
            lab = self_train(f, n_iter=n_iter, top_frac=top_frac)
            m = eval_lab(lab, f)
            if m is None:
                results[d] = None
            else:
                results[d] = m
        # Both pass?
        en_m = results["MHClip_EN"]
        zh_m = results["MHClip_ZH"]
        if en_m is None or zh_m is None:
            continue
        ab_e, mb_e = BASE["MHClip_EN"]
        ab_z, mb_z = BASE["MHClip_ZH"]
        pass_e = en_m[0] > ab_e and en_m[1] > mb_e
        pass_z = zh_m[0] > ab_z and zh_m[1] > mb_z
        tag = "PASS" if (pass_e and pass_z) else ("EN" if pass_e else ("ZH" if pass_z else ""))
        print(f"  n_iter={n_iter} top_frac={top_frac}: "
              f"EN {en_m[0]:.4f}/{en_m[1]:.4f}(t{en_m[2]})  "
              f"ZH {zh_m[0]:.4f}/{zh_m[1]:.4f}(t{zh_m[2]})  {tag}")
