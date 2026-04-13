"""Check pool-GMM behavior on both datasets as a unified rule."""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import gmm_threshold, otsu_threshold, metrics, load_scores_file, build_arrays
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


# Try: pool-GMM with different K, then use cut between components 0 and 1
for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    pool = np.concatenate([train, test_x])
    acc_b, mf_b = BASE[d]
    print(f"\n=== {d} === baseline {acc_b:.4f}/{mf_b:.4f}")

    # Vanilla GMM K=2 on pool
    t = gmm_threshold(pool)
    m = metrics(test_x, test_y, t)
    print(f"  pool-GMM K=2: t={t!r}  acc={m['acc']:.4f}  mf={m['mf']:.4f}  "
          f"{'STRICT' if m['acc']>acc_b and m['mf']>mf_b else ''}")

    # Check how close the threshold lands to cluster boundaries
    sorted_pool = sorted(set(pool))
    below_gap = max([v for v in sorted_pool if v < t], default=None)
    above_gap = min([v for v in sorted_pool if v > t], default=None)
    print(f"    closest below: {below_gap!r}")
    print(f"    closest above: {above_gap!r}")
    if below_gap is not None and above_gap is not None:
        print(f"    gap: {above_gap - below_gap!r}  t-position in gap: {(t-below_gap)/(above_gap-below_gap):.4f}")

    # Pool-Otsu
    t = otsu_threshold(pool)
    m = metrics(test_x, test_y, t)
    print(f"  pool-Otsu: t={t!r}  acc={m['acc']:.4f}  mf={m['mf']:.4f}  "
          f"{'STRICT' if m['acc']>acc_b and m['mf']>mf_b else ''}")

    # TF-GMM (test only) - this is the ZH baseline
    t = gmm_threshold(test_x)
    m = metrics(test_x, test_y, t)
    print(f"  TF-GMM: t={t!r}  acc={m['acc']:.4f}  mf={m['mf']:.4f}  "
          f"{'STRICT' if m['acc']>acc_b and m['mf']>mf_b else ''}")

    # TF-Otsu (test only) - this is the EN baseline
    t = otsu_threshold(test_x)
    m = metrics(test_x, test_y, t)
    print(f"  TF-Otsu: t={t!r}  acc={m['acc']:.4f}  mf={m['mf']:.4f}  "
          f"{'STRICT' if m['acc']>acc_b and m['mf']>mf_b else ''}")
