"""Where exactly does the ZH baseline threshold sit, and is it inside or outside
the clusters that matter? Also check: what's the whole-atom cut that reproduces
the baseline mf = 0.7871 on ZH?
"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, otsu_threshold, gmm_threshold, load_scores_file, build_arrays
from data_utils import load_annotations


def load(d):
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    return np.array(list(train.values()), dtype=float), test_x, test_y


for d in ["MHClip_EN", "MHClip_ZH"]:
    train, test_x, test_y = load(d)
    print(f"\n=== {d} ===")
    t_o = otsu_threshold(test_x)
    t_g = gmm_threshold(test_x)
    print(f"  TF-Otsu t = {t_o!r}")
    print(f"  TF-GMM  t = {t_g!r}")
    for t in [t_o, t_g]:
        m = metrics(test_x, test_y, t)
        print(f"    t={t:.6f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}  n_pred_pos={int((test_x>=t).sum())}")

    # Where does ZH baseline sit relative to 4-dec atoms?
    if d == "MHClip_ZH":
        t = t_g
    else:
        t = t_o
    print(f"\n  baseline cut t={t:.6f}")
    # Find atoms immediately around it
    atoms = sorted(set(round(float(v), 4) for v in test_x))
    prev_a = max([a for a in atoms if a <= t], default=None)
    next_a = min([a for a in atoms if a > t], default=None)
    print(f"    closest atom ≤ t: {prev_a}")
    print(f"    closest atom > t: {next_a}")
    # count test samples in rounded-atom [prev_a, next_a)
    samples_at_prev = [v for v in test_x if round(float(v), 4) == prev_a]
    samples_at_next = [v for v in test_x if round(float(v), 4) == next_a] if next_a else []
    print(f"    samples at {prev_a}: {sorted(samples_at_prev)[:6]}{'...' if len(samples_at_prev)>6 else ''} (n={len(samples_at_prev)})")
    print(f"    samples at {next_a}: {sorted(samples_at_next)[:6]}{'...' if len(samples_at_next)>6 else ''} (n={len(samples_at_next)})")

    # Is t above or below all samples at prev_a and next_a?
    below_t_at_prev = sum(1 for v in samples_at_prev if v < t)
    above_t_at_prev = sum(1 for v in samples_at_prev if v >= t)
    print(f"    at atom {prev_a}: {below_t_at_prev} samples < t, {above_t_at_prev} samples ≥ t")
