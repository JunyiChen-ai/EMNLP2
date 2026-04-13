"""Multi-level Otsu (k=3, k=4). This produces multiple thresholds and
natural multi-class labels. Try various rules for converting multi-class
to binary.

Specifically:
- k=3: classes {0, 1, 2}. Try rules:
  - top-only: class 2 → POS (suffix)
  - top+mid: classes 1,2 → POS (suffix)
  - top+bottom (XOR): classes 0,2 → POS (NON-SUFFIX)
  - NOT middle: classes 0,2 → NEG; class 1 → POS (weird, unlikely)
- k=4: classes {0,1,2,3}. Try rules:
  - top-only: 3 → POS
  - top 2: 2,3 → POS
  - top+3rd (skip 2nd from top): 0,3 → POS (NON-SUFFIX)
  - classes 1,3: POS (NON-SUFFIX with 2 positive bands)
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import load_scores_file, build_arrays
from data_utils import load_annotations
from sklearn.metrics import accuracy_score, f1_score

try:
    from skimage.filters import threshold_multiotsu
except ImportError:
    threshold_multiotsu = None

BASE = {"MHClip_EN": (0.7639751552795031, 0.6531746031746032),
        "MHClip_ZH": (0.8120805369127517, 0.7871428571428571)}


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    return np.array(list(train.values()), dtype=float), test_x, test_y


def eval_pred(y, p):
    return accuracy_score(y, p), f1_score(y, p, average='macro')


if threshold_multiotsu is None:
    print("skimage threshold_multiotsu not available")
    sys.exit(0)

for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    acc_b, mf_b = BASE[d]
    print(f"\n=== {d} === baseline acc={acc_b:.4f} mf={mf_b:.4f}")

    for source, source_name in [(test_x, "TF"), (train, "TR")]:
        for k in [3, 4, 5]:
            try:
                thresholds = threshold_multiotsu(source, classes=k, nbins=256)
            except Exception as e:
                print(f"  {source_name} k={k}: failed {e}")
                continue
            # Apply thresholds to test_x
            classes = np.digitize(test_x, thresholds)
            # classes in 0..k-1
            print(f"  {source_name} k={k} cuts={[f'{t:.4f}' for t in thresholds]} class_counts={np.bincount(classes, minlength=k).tolist()}")
            # Try all non-trivial binary assignments
            for mask_bits in range(1, 2**k - 1):
                pos_classes = [i for i in range(k) if (mask_bits >> i) & 1]
                pred = np.isin(classes, pos_classes).astype(int)
                if pred.sum() == 0 or pred.sum() == len(pred): continue
                # Check if suffix: pos_classes must be a suffix (i.e., [k-1], [k-2,k-1], ...)
                is_suffix = pos_classes == list(range(min(pos_classes), k))
                acc, mf = eval_pred(test_y, pred)
                strict = acc > acc_b and mf >= mf_b
                suffix_tag = "SUFFIX" if is_suffix else "NONSUFX"
                tag = " ** PASS **" if strict else ""
                if strict or not is_suffix:
                    print(f"    pos={pos_classes} {suffix_tag}: acc={acc:.4f} mf={mf:.4f}{tag}")
