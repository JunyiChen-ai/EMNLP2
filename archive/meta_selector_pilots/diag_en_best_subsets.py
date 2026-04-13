"""Find the atom subsets achieving EN best (tp=22, fp=10) via backtracking DP.
Enumerate all subsets and filter those that yield one of the 4 strict-both (tp,fp).
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import load_scores_file, build_arrays
from data_utils import load_annotations

base_d = "/data/jehc223/EMNLP2/results/holistic_2b/MHClip_EN"
test = load_scores_file(f"{base_d}/test_binary.jsonl")
ann = load_annotations("MHClip_EN")
test_x, test_y = build_arrays(test, ann)

acc_b, mf_b = 0.7639751552795031, 0.6531746031746032
n_te = len(test_x)
rounded_te = np.round(test_x, 4)
atoms_vals = np.array(sorted(set(rounded_te)))
K = len(atoms_vals)

pos = np.array([int(((rounded_te == v) & (test_y == 1)).sum()) for v in atoms_vals])
neg = np.array([int(((rounded_te == v) & (test_y == 0)).sum()) for v in atoms_vals])
total_pos = int(test_y.sum())
total_neg = n_te - total_pos

# Enumerate 2^32 is too many. But we only care about subsets where the
# rule forms SPECIFIC (tp, fp). Let's use iterative search: find subsets
# by dropping one atom at a time from "all POS" or growing from "none".
#
# Strict-both pairs: find them first
strict_pairs = []
for tp in range(total_pos + 1):
    for fp in range(total_neg + 1):
        fn = total_pos - tp
        tn = total_neg - fp
        acc = (tp + tn) / n_te
        if tp + fp == 0 or tn + fn == 0:
            continue
        prec_p = tp / (tp + fp)
        rec_p = tp / total_pos
        f1_p = 2 * prec_p * rec_p / (prec_p + rec_p) if (prec_p + rec_p) > 0 else 0
        prec_n = tn / (tn + fn)
        rec_n = tn / total_neg
        f1_n = 2 * prec_n * rec_n / (prec_n + rec_n) if (prec_n + rec_n) > 0 else 0
        mf = (f1_p + f1_n) / 2
        if acc > acc_b and mf > mf_b:
            strict_pairs.append((tp, fp, acc, mf))
print(f"Strict pairs: {len(strict_pairs)}")
for tp, fp, acc, mf in strict_pairs:
    print(f"  tp={tp} fp={fp} acc={acc:.4f} mf={mf:.4f}")

# For each strict pair, find all atom subsets achieving it — use subset enumeration
# via meet-in-the-middle: split atoms into two halves, enumerate each, join.
halfL = K // 2
halfR = K - halfL
print(f"\nMeet-in-middle: halfL={halfL} (2^{halfL} subsets), halfR={halfR}")

left_pos = pos[:halfL]
left_neg = neg[:halfL]
right_pos = pos[halfL:]
right_neg = neg[halfL:]

# For each left subset, record (tp_L, fp_L) -> list of masks
left_map = {}
for mL in range(2 ** halfL):
    tpL = sum(int(left_pos[i]) for i in range(halfL) if (mL >> i) & 1)
    fpL = sum(int(left_neg[i]) for i in range(halfL) if (mL >> i) & 1)
    left_map.setdefault((tpL, fpL), []).append(mL)

# For each strict target, find right subsets that complement
target_set = set((tp, fp) for tp, fp, _, _ in strict_pairs)
hits = []
for mR in range(2 ** halfR):
    tpR = sum(int(right_pos[i]) for i in range(halfR) if (mR >> i) & 1)
    fpR = sum(int(right_neg[i]) for i in range(halfR) if (mR >> i) & 1)
    for tp, fp in target_set:
        needL = (tp - tpR, fp - fpR)
        if needL in left_map:
            for mL in left_map[needL][:3]:  # cap each left side
                mask = mL | (mR << halfL)
                hits.append((mask, tp, fp))
                if len(hits) > 300:
                    break
        if len(hits) > 300:
            break
    if len(hits) > 300:
        break

print(f"\nFound {len(hits)} hit subsets (capped)")
# Show first 10
for mask, tp, fp in hits[:10]:
    sel_atoms = [atoms_vals[i] for i in range(K) if (mask >> i) & 1]
    print(f"  tp={tp} fp={fp} K_sel={len(sel_atoms)}")
    print(f"    atoms: {[f'{v:.4f}' for v in sel_atoms]}")
    # Transition count
    atom_lab = np.array([(mask >> i) & 1 for i in range(K)], dtype=bool)
    trans = int(np.sum(atom_lab[1:] != atom_lab[:-1]))
    print(f"    trans={trans}")
