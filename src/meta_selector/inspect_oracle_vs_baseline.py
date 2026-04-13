"""Compare oracle passing subset to baseline Otsu cut prediction.
Identify exactly which atoms need to be flipped from neg to pos or
vice versa to reach the passing subset.
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import load_scores_file, build_arrays
from data_utils import load_annotations
from sklearn.metrics import accuracy_score, f1_score
from skimage.filters import threshold_otsu

BASE_ACC = {"MHClip_EN": 0.7639751552795031, "MHClip_ZH": 0.8120805369127517}
BASE_MF = {"MHClip_EN": 0.6531746031746032, "MHClip_ZH": 0.7871428571428571}

BEST_EN = [12, 19, 24, 25, 26, 27, 28, 29, 30, 31]
BEST_ZH = [16, 18, 20, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32]


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    return np.array(list(train.values()), dtype=float), test_x, test_y


for d, best in [("MHClip_EN", BEST_EN), ("MHClip_ZH", BEST_ZH)]:
    train, test_x, test_y = load(d)
    print(f"\n=== {d} ===")
    # baseline is TF-Otsu
    t_otsu = float(threshold_otsu(test_x, nbins=256))
    baseline_pred = (test_x >= t_otsu).astype(int)
    acc = accuracy_score(test_y, baseline_pred)
    mf = f1_score(test_y, baseline_pred, average='macro')
    print(f"  Baseline Otsu cut: {t_otsu:.4f}, acc={acc:.4f} mf={mf:.4f}")

    # Oracle: positive atoms = best list
    rounded = np.round(test_x, 4)
    atoms_vals = sorted(set(rounded))
    oracle_pred = np.zeros(len(test_x), dtype=int)
    for idx, aval in enumerate(atoms_vals):
        if idx in best:
            mask = rounded == aval
            oracle_pred[mask] = 1
    acc_o = accuracy_score(test_y, oracle_pred)
    mf_o = f1_score(test_y, oracle_pred, average='macro')
    print(f"  Oracle subset: acc={acc_o:.4f} mf={mf_o:.4f}")

    # Per-atom comparison
    print(f"\n  Atom-level differences (oracle vs baseline):")
    print(f"  {'idx':>3s} {'val':>8s} {'pos':>4s} {'neg':>4s} {'base':>5s} {'orac':>5s} {'diff':>6s}")
    for idx, aval in enumerate(atoms_vals):
        mask = rounded == aval
        pc = int(test_y[mask].sum())
        nc = int((1 - test_y[mask]).sum())
        base_lbl = 1 if aval >= t_otsu else 0
        orac_lbl = 1 if idx in best else 0
        diff = ""
        if base_lbl != orac_lbl:
            diff = "FLIP" + ("+" if orac_lbl == 1 else "-")
        print(f"  {idx:>3d} {aval:>8.4f} {pc:>4d} {nc:>4d} {base_lbl:>5d} {orac_lbl:>5d} {diff:>6s}")

    # Summary
    flips_pos = []
    flips_neg = []
    for idx, aval in enumerate(atoms_vals):
        base_lbl = 1 if aval >= t_otsu else 0
        orac_lbl = 1 if idx in best else 0
        if base_lbl == 0 and orac_lbl == 1:
            flips_pos.append(idx)
        elif base_lbl == 1 and orac_lbl == 0:
            flips_neg.append(idx)
    print(f"\n  Flipped NEG→POS atoms (in oracle, out of baseline): {flips_pos}")
    print(f"  Flipped POS→NEG atoms (out of oracle, in baseline):  {flips_neg}")
