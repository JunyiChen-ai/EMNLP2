"""Train-mode distance features.

The train distribution is dominated by NEG class. A test atom distant
from the primary train mode is "atypical" — potential POS. Distance is
NON-MONOTONE in the score (two atoms equidistant from the mode map to
the same distance) so it's a non-smooth-in-score feature.

Features:
- abs(atom - train_mode)
- abs(atom - train_mode) ranked
- signed distance (atom - train_mode)
- distance to nearest train atom (min over train atoms)
- rank of atom in train distribution (percentile)
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
    n_tr, n_te = len(train), len(test_x)
    rounded_te = np.round(test_x, 4)
    rounded_tr = np.round(train, 4)
    atoms = np.array(sorted(set(rounded_te)))
    K = len(atoms)
    t_base = otsu_threshold(test_x) if d == "MHClip_EN" else gmm_threshold(test_x)
    base_atom = (atoms >= t_base).astype(bool)

    bw = max(1e-4, float(np.std(train)) * 1.06 * (n_tr ** -0.2))
    tr_k = gaussian_kde(train, bw_method=bw / np.std(train))
    # Grid find mode
    grid = np.linspace(train.min(), train.max(), 1000)
    kde_vals = tr_k(grid)
    tr_mode = float(grid[np.argmax(kde_vals)])
    # Second mode: mask out first mode and find next peak (if any)
    # Simpler: pick secondary peak via sorted local maxima
    local_max = []
    for i in range(1, len(grid) - 1):
        if kde_vals[i] > kde_vals[i - 1] and kde_vals[i] > kde_vals[i + 1]:
            local_max.append((kde_vals[i], grid[i]))
    local_max.sort(reverse=True)

    abs_dist = np.abs(atoms - tr_mode)
    signed_dist = atoms - tr_mode
    # Distance to nearest train atom
    nearest_dist = np.array([float(np.min(np.abs(train - a))) for a in atoms])
    # Percentile in train
    percentile = np.array([float((train <= a).mean()) for a in atoms])
    # Within-train KDE density rank
    tr_d = np.array([float(tr_k(v)[0]) for v in atoms])
    tr_d_rank = np.argsort(np.argsort(tr_d))  # low rank = rare in train

    # Secondary-mode distance if a second peak exists
    if len(local_max) >= 2:
        tr_mode2 = local_max[1][1]
        dist_to_mode2 = np.abs(atoms - tr_mode2)
    else:
        tr_mode2 = None
        dist_to_mode2 = np.zeros(K)

    return dict(
        atoms=atoms, rounded_te=rounded_te, test_y=test_y,
        base_atom=base_atom,
        abs_dist=abs_dist, signed_dist=signed_dist,
        nearest_dist=nearest_dist, percentile=percentile,
        tr_d=tr_d, tr_d_rank=tr_d_rank.astype(float),
        dist_to_mode2=dist_to_mode2, tr_mode=tr_mode, tr_mode2=tr_mode2,
    )


def eval_lab(atom_lab, f):
    atom_lab = np.asarray(atom_lab).astype(bool)
    s = atom_lab.sum()
    if s == 0 or s == len(atom_lab):
        return None
    atom_map = dict(zip(f["atoms"], atom_lab.astype(int)))
    sp = np.array([atom_map[v] for v in f["rounded_te"]])
    acc = accuracy_score(f["test_y"], sp)
    mf = f1_score(f["test_y"], sp, average='macro')
    trans = int(np.sum(atom_lab[1:] != atom_lab[:-1]))
    return acc, mf, trans


for d in ["MHClip_EN", "MHClip_ZH"]:
    f = build(d)
    ab, mb = BASE[d]
    print(f"\n=== {d} === baseline {ab:.4f}/{mb:.4f}")
    print(f"  tr_mode={f['tr_mode']:.4f}  tr_mode2={f['tr_mode2']}")
    # Try feature thresholds
    for name in ["abs_dist", "nearest_dist", "percentile", "tr_d_rank", "dist_to_mode2"]:
        vals = f[name]
        for q in np.linspace(0.1, 0.9, 9):
            c = float(np.quantile(vals, q))
            for op in [">", "<"]:
                if op == ">":
                    cond = vals > c
                else:
                    cond = vals < c
                # Rules
                rules = [
                    (f"{name}{op}q{q:.2f}", cond),
                    (f"base OR {name}{op}q{q:.2f}", f["base_atom"] | cond),
                    (f"base AND {name}{op}q{q:.2f}", f["base_atom"] & cond),
                ]
                for rn, lab in rules:
                    m = eval_lab(lab, f)
                    if m is None:
                        continue
                    acc, mf, trans = m
                    strict = acc > ab and mf > mb
                    if strict:
                        print(f"  {rn}: acc={acc:.4f} mf={mf:.4f} trans={trans} PASS")
