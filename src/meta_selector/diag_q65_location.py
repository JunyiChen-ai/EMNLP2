"""Where does pool-q=0.65 land on ZH? Is it within a cluster or between?"""
import sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import metrics, load_scores_file, build_arrays
from data_utils import load_annotations


base_d = "/data/jehc223/EMNLP2/results/holistic_2b/MHClip_ZH"
train = load_scores_file(f"{base_d}/train_binary.jsonl")
test = load_scores_file(f"{base_d}/test_binary.jsonl")
ann = load_annotations("MHClip_ZH")
test_x, test_y = build_arrays(test, ann)
train_arr = np.array(list(train.values()), dtype=float)
pool = np.concatenate([train_arr, test_x])

for q in [0.65, 0.66]:
    t = float(np.quantile(pool, q))
    print(f"\nq={q}  t={t!r}  t rounded to 10 dec: {t:.10f}")

    # Distance to nearest pool sample below and above
    below = [v for v in pool if v < t]
    above = [v for v in pool if v >= t]
    nearest_below = max(below) if below else None
    nearest_above = min(above) if above else None
    print(f"  nearest pool sample < t: {nearest_below!r}")
    print(f"  nearest pool sample >= t: {nearest_above!r}")
    print(f"  gap below: {t - nearest_below if nearest_below else 'N/A'}")
    print(f"  gap above: {nearest_above - t if nearest_above else 'N/A'}")

    m = metrics(test_x, test_y, t)
    print(f"  acc={m['acc']:.4f}  mf={m['mf']:.4f}  n_pred_pos={int((test_x>=t).sum())}")

# How does test_x distribute around this cluster?
print("\nTest samples in [0.028, 0.040]:")
items = sorted(zip(test_x, test_y), key=lambda p: p[0])
for x, y in items:
    if 0.028 <= x <= 0.040:
        print(f"  x={x!r}  label={int(y)}")
