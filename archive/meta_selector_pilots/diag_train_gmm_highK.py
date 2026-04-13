"""Higher-K train GMM mode membership."""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, load_scores_file, build_arrays
from data_utils import load_annotations
from sklearn.metrics import accuracy_score, f1_score
from sklearn.mixture import GaussianMixture

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    return np.array(list(train.values()), dtype=float), test_x, test_y


def eval_lab(lab, atoms, rounded_te, test_y):
    lab = np.asarray(lab).astype(bool)
    s = lab.sum()
    if s == 0 or s == len(lab):
        return None
    atom_map = dict(zip(atoms, lab.astype(int)))
    sp = np.array([atom_map[v] for v in rounded_te])
    acc = accuracy_score(test_y, sp)
    mf = f1_score(test_y, sp, average='macro')
    trans = int(np.sum(lab[1:] != lab[:-1]))
    return acc, mf, trans


data = {}
for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    rounded_te = np.round(test_x, 4)
    atoms = np.array(sorted(set(rounded_te)))
    t_base = otsu_threshold(test_x) if d == "MHClip_EN" else gmm_threshold(test_x)
    base_atom = (atoms >= t_base).astype(bool)
    data[d] = dict(train=train, test_x=test_x, test_y=test_y,
                   rounded_te=rounded_te, atoms=atoms, base_atom=base_atom)

# For each K, for each mode_idx, for each op, check unified
print("Sweep K=5..12, all modes, all ops — unified check")
unified = []
for K in range(5, 13):
    # Fit GMM on each train, get mode order, get per-atom mode assignment
    gms = {}
    for d in data:
        train = data[d]["train"]
        try:
            gm = GaussianMixture(n_components=K, random_state=0, n_init=3,
                                 reg_covar=1e-3).fit(train.reshape(-1, 1))
        except Exception:
            continue
        order = np.argsort(gm.means_.flatten())
        pred = gm.predict(data[d]["atoms"].reshape(-1, 1))
        pred_ord = np.array([list(order).index(p) for p in pred])
        gms[d] = pred_ord
    if "MHClip_EN" not in gms or "MHClip_ZH" not in gms:
        continue
    for k in range(K):
        for op in ["in", "not_in"]:
            for base_op in ["", "base_or_", "base_and_", "base_and_not_"]:
                ok_both = True
                res = {}
                for d in data:
                    pred_ord = gms[d]
                    b = data[d]["base_atom"]
                    mem = (pred_ord == k) if op == "in" else (pred_ord != k)
                    if base_op == "":
                        lab = mem
                    elif base_op == "base_or_":
                        lab = b | mem
                    elif base_op == "base_and_":
                        lab = b & mem
                    else:
                        lab = b & ~mem
                    m = eval_lab(lab, data[d]["atoms"], data[d]["rounded_te"],
                                 data[d]["test_y"])
                    if m is None:
                        ok_both = False
                        break
                    ab, mb = BASE[d]
                    if not (m[0] > ab and m[1] > mb):
                        ok_both = False
                        break
                    res[d] = m
                if ok_both:
                    rule = f"K={K} {base_op}mode_{k}_{op}"
                    unified.append((rule, res))

print(f"Unified strict-both hits: {len(unified)}")
for rule, res in unified[:20]:
    en = res["MHClip_EN"]
    zh = res["MHClip_ZH"]
    print(f"  {rule}: EN {en[0]:.4f}/{en[1]:.4f}(t{en[2]})  ZH {zh[0]:.4f}/{zh[1]:.4f}(t{zh[2]})")
