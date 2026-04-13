"""Same as inspect_oracle_vs_gmm but use the ACTUAL team baseline:
EN TF-Otsu (quick_eval_all.otsu_threshold) and ZH TF-GMM (quick_eval_all.gmm_threshold).
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, load_scores_file, build_arrays
from data_utils import load_annotations
from sklearn.metrics import accuracy_score, f1_score

BEST_EN = [12, 19, 24, 25, 26, 27, 28, 29, 30, 31]
BEST_ZH = [16, 18, 20, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32]


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    return np.array(list(train.values()), dtype=float), test_x, test_y


for d, best, baseline_method in [
    ("MHClip_EN", BEST_EN, otsu_threshold),
    ("MHClip_ZH", BEST_ZH, gmm_threshold),
]:
    train, test_x, test_y = load(d)
    print(f"\n=== {d} === baseline_method={baseline_method.__name__}")
    t = float(baseline_method(test_x))
    print(f"  Cut: {t:.4f}")
    pred = (test_x >= t).astype(int)
    acc = accuracy_score(test_y, pred)
    mf = f1_score(test_y, pred, average='macro')
    print(f"  Baseline: acc={acc:.4f} mf={mf:.4f}")

    rounded = np.round(test_x, 4)
    atoms_vals = sorted(set(rounded))
    flips_pos, flips_neg = [], []
    print(f"\n  {'idx':>3s} {'val':>8s} {'pos':>4s} {'neg':>4s} {'base':>5s} {'orac':>5s} {'diff':>6s}")
    for idx, aval in enumerate(atoms_vals):
        mask = rounded == aval
        pc = int(test_y[mask].sum())
        nc = int((1 - test_y[mask]).sum())
        base_lbl = 1 if aval >= t else 0
        orac_lbl = 1 if idx in best else 0
        diff = ""
        if base_lbl != orac_lbl:
            diff = "FLIP+" if orac_lbl == 1 else "FLIP-"
            if orac_lbl == 1: flips_pos.append(idx)
            else: flips_neg.append(idx)
        print(f"  {idx:>3d} {aval:>8.4f} {pc:>4d} {nc:>4d} {base_lbl:>5d} {orac_lbl:>5d} {diff:>6s}")
    print(f"\n  Flips NEG→POS: {flips_pos}")
    print(f"  Flips POS→NEG: {flips_neg}")
