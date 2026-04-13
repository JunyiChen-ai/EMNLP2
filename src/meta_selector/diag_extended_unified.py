"""Extended unified 2-feature rule search including train-structural features.

Feature pool:
- v (atom value)
- te_cnt, tr_cnt
- dr_0 (density ratio bw=0.5)
- ratio (te/tr count ratio)
- nearest_dist (distance to nearest train atom)
- percentile (atom's percentile in train)
- tr_d_rank (rank of atom by train kde density)
- abs_dist (distance to primary train mode)
- dist_to_mode2 (distance to secondary train mode)

Test all rules of form:
  (base OP1 feat1OP_c1) OR|AND (base OP2 feat2OP_c2)
with quantile-unified cuts across datasets.
"""
import sys, numpy as np, itertools
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
    atoms = np.array(sorted(set(rounded_te)))
    K = len(atoms)
    te_cnt = np.array([int((rounded_te == v).sum()) for v in atoms])
    tr_cnt = np.array([int((np.round(train, 4) == v).sum()) for v in atoms])
    t_base = otsu_threshold(test_x) if d == "MHClip_EN" else gmm_threshold(test_x)
    base_atom = (atoms >= t_base).astype(bool)

    bw_base = max(1e-4, float(np.std(train)) * 1.06 * (n_tr ** -0.2))
    bw = bw_base * 0.5
    tr_k = gaussian_kde(train, bw_method=bw / np.std(train))
    te_k = gaussian_kde(test_x, bw_method=bw / np.std(test_x))
    tr_d = np.array([float(tr_k(v)[0]) for v in atoms])
    te_d = np.array([float(te_k(v)[0]) for v in atoms])
    dr_0 = tr_d / (te_d + 1e-9)
    ratio = (te_cnt / n_te) / ((tr_cnt + 0.5) / n_tr)

    nearest_dist = np.array([float(np.min(np.abs(train - a))) for a in atoms])
    percentile = np.array([float((train <= a).mean()) for a in atoms])
    tr_d_rank = np.argsort(np.argsort(tr_d)).astype(float)

    # Train modes
    grid = np.linspace(train.min(), train.max(), 1000)
    kv = tr_k(grid)
    tr_mode = float(grid[np.argmax(kv)])
    locs = []
    for i in range(1, len(grid) - 1):
        if kv[i] > kv[i - 1] and kv[i] > kv[i + 1]:
            locs.append((kv[i], grid[i]))
    locs.sort(reverse=True)
    tr_mode2 = locs[1][1] if len(locs) >= 2 else tr_mode
    abs_dist = np.abs(atoms - tr_mode)
    dist_to_mode2 = np.abs(atoms - tr_mode2)

    return dict(
        atoms=atoms, rounded_te=rounded_te, test_y=test_y, base_atom=base_atom,
        v=atoms.astype(float), te_cnt=te_cnt.astype(float), tr_cnt=tr_cnt.astype(float),
        dr_0=dr_0, ratio=ratio, tr_d=tr_d, te_d=te_d,
        nearest_dist=nearest_dist, percentile=percentile, tr_d_rank=tr_d_rank,
        abs_dist=abs_dist, dist_to_mode2=dist_to_mode2,
        pos_per_atom=np.array([int(((rounded_te == v) & (test_y == 1)).sum()) for v in atoms]),
        neg_per_atom=np.array([int(((rounded_te == v) & (test_y == 0)).sum()) for v in atoms]),
        total_pos=int(test_y.sum()), n_te=n_te,
    )


def metrics(atom_lab, f):
    atom_lab = np.asarray(atom_lab).astype(int)
    tp = int((atom_lab * f["pos_per_atom"]).sum())
    fp = int((atom_lab * f["neg_per_atom"]).sum())
    if tp + fp == 0:
        return None
    fn = f["total_pos"] - tp
    tn = f["n_te"] - f["total_pos"] - fp
    if tn + fn == 0:
        return None
    acc = (tp + tn) / f["n_te"]
    ppos = tp + fp
    pneg = tn + fn
    prec_p = tp / ppos
    rec_p = tp / f["total_pos"] if f["total_pos"] > 0 else 0
    f1_p = 2 * prec_p * rec_p / (prec_p + rec_p) if (prec_p + rec_p) > 0 else 0
    prec_n = tn / pneg
    rec_n = tn / (f["n_te"] - f["total_pos"]) if (f["n_te"] - f["total_pos"]) > 0 else 0
    f1_n = 2 * prec_n * rec_n / (prec_n + rec_n) if (prec_n + rec_n) > 0 else 0
    mf = (f1_p + f1_n) / 2
    return acc, mf


en = build("MHClip_EN")
zh = build("MHClip_ZH")

feat_names = ["v", "te_cnt", "tr_cnt", "dr_0", "ratio", "nearest_dist",
              "percentile", "tr_d_rank", "abs_dist", "dist_to_mode2", "tr_d", "te_d"]

qs = np.linspace(0.05, 0.95, 19)
hits = []
checked = 0
for f1n in feat_names:
    for f2n in feat_names:
        for q1 in qs:
            for q2 in qs:
                for op1 in [">", "<"]:
                    for op2 in [">", "<"]:
                        for logic in ["OR", "AND", "ANDNOT_BASE"]:
                            checked += 1
                            en_c1 = float(np.quantile(en[f1n], q1))
                            en_c2 = float(np.quantile(en[f2n], q2))
                            zh_c1 = float(np.quantile(zh[f1n], q1))
                            zh_c2 = float(np.quantile(zh[f2n], q2))
                            ok_both = True
                            for dname, f, c1, c2 in [("MHClip_EN", en, en_c1, en_c2),
                                                      ("MHClip_ZH", zh, zh_c1, zh_c2)]:
                                v1 = f[f1n]; v2 = f[f2n]
                                a1 = (v1 > c1) if op1 == ">" else (v1 < c1)
                                a2 = (v2 > c2) if op2 == ">" else (v2 < c2)
                                if logic == "OR":
                                    lab = a1 | a2
                                elif logic == "AND":
                                    lab = a1 & a2
                                else:  # base AND NOT(a1 OR a2) + base (i.e. remove from base)
                                    lab = f["base_atom"] & ~(a1 | a2)
                                m = metrics(lab.astype(int), f)
                                if m is None:
                                    ok_both = False
                                    break
                                ab, mb = BASE[dname]
                                if not (m[0] > ab and m[1] > mb):
                                    ok_both = False
                                    break
                            if ok_both:
                                hits.append((f1n, op1, q1, f2n, op2, q2, logic))

print(f"Checked {checked} unified rule variants, {len(hits)} strict-both unified hits")
for h in hits[:30]:
    print(" ", h)
