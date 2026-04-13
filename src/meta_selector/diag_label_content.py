"""Check all 402 ZH passing subsets: does EVERY passing subset require
including at least one atom that is label-indistinguishable from an
EXCLUDED atom (i.e. with identical score-based feature vector)?

If yes, no label-free rule can enumerate the passing subset, because
identical features force identical decisions. This is the rigorous
barrier.

For each passing subset S, check:
  - for each atom a in S, is there an atom b NOT in S such that
    the feature vector of a equals that of b? If so, a and b are
    label-free-indistinguishable, and no rule can include a while
    excluding b.

Since atoms have unique score values, the feature-vector equality
is only across test atoms of the same score. But test atoms have
unique rounded values by construction (we cluster by atom), so
each atom has a unique feature vector.

So the question becomes: is there a label-free FEATURE RULE (derived
from pool structure) that, when applied to just the atom's value,
produces a label consistent with the passing subset?

That's what LR is probing. LR at 13/17 ZH means 4 atoms are
linearly-indistinguishable from holes at their neighbors.

Let me directly check: for each passing subset, what is the
effective label-content of the atoms IN vs OUT?
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
from data_utils import load_annotations

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}

def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    return np.array(list(train.values()), dtype=float), test_x, test_y


def atom_stats(d):
    train, test_x, test_y = load(d)
    rounded = np.round(test_x, 4)
    atoms_vals = sorted(set(rounded))
    atoms = []
    for idx, aval in enumerate(atoms_vals):
        mask = rounded == aval
        pc = int(test_y[mask].sum())
        nc = int((1 - test_y[mask]).sum())
        atoms.append({"idx": idx, "val": aval, "pc": pc, "nc": nc,
                      "net": pc - nc, "tot": pc + nc})
    return atoms


for d in ["MHClip_EN", "MHClip_ZH"]:
    atoms = atom_stats(d)
    print(f"\n=== {d} ===")
    print(f"  {len(atoms)} atoms")
    print("  All atoms with net label content (pos - neg):")
    for a in atoms:
        flag = "+" if a["net"] > 0 else ("-" if a["net"] < 0 else "=")
        print(f"    idx {a['idx']:2d} val={a['val']:.4f} p/n={a['pc']}/{a['nc']} net={a['net']:+d} {flag}")

    # Report: how many net-tied atoms (net==0)?
    ties = [a for a in atoms if a["net"] == 0 and a["tot"] > 0]
    neg_atoms = [a for a in atoms if a["net"] < 0]
    pos_atoms = [a for a in atoms if a["net"] > 0]
    print(f"  pos-dom={len(pos_atoms)}, neg-dom={len(neg_atoms)}, ties={len(ties)}")
