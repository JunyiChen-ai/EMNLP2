"""Same but use TF-GMM for ZH baseline (which is the actual team baseline)."""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import load_scores_file, build_arrays
from data_utils import load_annotations
from sklearn.metrics import accuracy_score, f1_score
from sklearn.mixture import GaussianMixture

BEST_ZH = [16, 18, 20, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32]


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    return np.array(list(train.values()), dtype=float), test_x, test_y

d = "MHClip_ZH"
train, test_x, test_y = load(d)
# TF-GMM k=2 midpoint of means
g = GaussianMixture(n_components=2, random_state=0).fit(test_x.reshape(-1, 1))
mus = sorted(g.means_.flatten())
t_gmm = float((mus[0] + mus[1]) / 2)
print(f"TF-GMM cut: {t_gmm:.4f}")
pred = (test_x >= t_gmm).astype(int)
acc = accuracy_score(test_y, pred)
mf = f1_score(test_y, pred, average='macro')
print(f"Baseline TF-GMM: acc={acc:.4f} mf={mf:.4f}")

rounded = np.round(test_x, 4)
atoms_vals = sorted(set(rounded))

print(f"\nAtom-level baseline vs oracle:")
flips_pos, flips_neg = [], []
for idx, aval in enumerate(atoms_vals):
    mask = rounded == aval
    pc = int(test_y[mask].sum())
    nc = int((1 - test_y[mask]).sum())
    base_lbl = 1 if aval >= t_gmm else 0
    orac_lbl = 1 if idx in BEST_ZH else 0
    diff = ""
    if base_lbl != orac_lbl:
        diff = "FLIP+" if orac_lbl == 1 else "FLIP-"
        if orac_lbl == 1: flips_pos.append(idx)
        else: flips_neg.append(idx)
    print(f"  idx {idx:>2d} val={aval:.4f} p/n={pc}/{nc}  base={base_lbl} orac={orac_lbl} {diff}")

print(f"\nFlips NEG→POS: {flips_pos}")
print(f"Flips POS→NEG: {flips_neg}")
