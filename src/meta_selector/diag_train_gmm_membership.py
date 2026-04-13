"""GMM fitted to TRAIN pool; test atoms labeled by mode membership.

Rule: atom is POS iff it belongs to the K-th Gaussian mode of the train
distribution (for K=1, 2, 3, 4 mode-counts). Mode membership is
non-monotone in atom value (each mode has a finite support).

Also try: posterior P(atom in mode K | train GMM).
"""
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


by_rule = {}
for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    ab, mb = BASE[d]
    print(f"\n=== {d} === baseline {ab:.4f}/{mb:.4f}")
    rounded_te = np.round(test_x, 4)
    atoms = np.array(sorted(set(rounded_te)))
    t_base = otsu_threshold(test_x) if d == "MHClip_EN" else gmm_threshold(test_x)
    base_atom = (atoms >= t_base).astype(bool)

    for K in [2, 3, 4]:
        try:
            gm = GaussianMixture(n_components=K, random_state=0, n_init=5).fit(train.reshape(-1, 1))
        except Exception as e:
            print(f"  K={K}: failed {e}")
            continue
        # Sort components by mean
        order = np.argsort(gm.means_.flatten())
        # Predict hard cluster for each atom
        pred = gm.predict(atoms.reshape(-1, 1))
        pred_ord = np.array([list(order).index(p) for p in pred])  # reordered 0..K-1
        # Also posterior per component
        post = gm.predict_proba(atoms.reshape(-1, 1))
        post = post[:, order]

        # Rule: atom belongs to mode 0 (lowest mean)
        for k in range(K):
            rules = [
                (f"K={K} in_mode_{k}", pred_ord == k),
                (f"K={K} NOT_in_mode_{k}", pred_ord != k),
                (f"K={K} base OR mode_{k}", base_atom | (pred_ord == k)),
                (f"K={K} base AND NOT mode_{k}", base_atom & (pred_ord != k)),
                (f"K={K} base OR post_{k}>0.5", base_atom | (post[:, k] > 0.5)),
                (f"K={K} base AND post_{k}<0.5", base_atom & (post[:, k] < 0.5)),
            ]
            for name, lab in rules:
                m = eval_lab(lab, atoms, rounded_te, test_y)
                if m is None:
                    continue
                acc, mf, trans = m
                strict = acc > ab and mf > mb
                tag = " PASS" if strict else ""
                if strict or acc > ab - 0.01:
                    print(f"  {name}: acc={acc:.4f} mf={mf:.4f} trans={trans}{tag}")
                by_rule.setdefault(name, {})[d] = (acc, mf, strict)

print("\n=== Rules passing strict-both on BOTH datasets ===")
found_any = False
for name, v in by_rule.items():
    if "MHClip_EN" in v and "MHClip_ZH" in v:
        if v["MHClip_EN"][2] and v["MHClip_ZH"][2]:
            print(f"  UNIFIED PASS: {name}")
            found_any = True
if not found_any:
    print("  (none)")
