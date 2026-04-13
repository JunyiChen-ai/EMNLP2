"""Look at exact raw values near the atom 0.0373 in ZH."""
import numpy as np, sys
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import load_scores_file, build_arrays
from data_utils import load_annotations

d = "MHClip_ZH"
base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
train = load_scores_file(f"{base_d}/train_binary.jsonl")
test = load_scores_file(f"{base_d}/test_binary.jsonl")
ann = load_annotations(d)
test_x, test_y = build_arrays(test, ann)
train_arr = np.array(list(train.values()), dtype=float)
pool = np.concatenate([train_arr, test_x])

# Find all pool values that round to 0.0373
target = 0.0373
close = pool[(pool > 0.035) & (pool < 0.040)]
print(f"Pool values rounded ~0.0373: n={len(close)}")
print("Unique values:")
for v in sorted(set(close)):
    count = int((pool == v).sum())
    print(f"  {v!r}  (count={count})")

print("\nRaw test values near 0.0373:")
close_test = [(float(x), int(y)) for x, y in zip(test_x, test_y) if 0.035 < x < 0.040]
for x, y in sorted(close_test):
    print(f"  {x!r}  label={y}")

print("\nRaw test values near 0.0474:")
close_test2 = [(float(x), int(y)) for x, y in zip(test_x, test_y) if 0.045 < x < 0.050]
for x, y in sorted(close_test2):
    print(f"  {x!r}  label={y}")

# What threshold is MAD rule using? And what does baseline use?
from quick_eval_all import gmm_threshold, otsu_threshold
t_gmm_test = gmm_threshold(test_x)
print(f"\nbaseline ZH TF-GMM threshold: {t_gmm_test!r}")
print(f"MAD rule ZH threshold:         0.037326883418294175")
# Count test samples flagged by each
n_baseline = int((test_x >= t_gmm_test).sum())
n_mad = int((test_x >= 0.037326883418294175).sum())
print(f"\nbaseline flags {n_baseline} test samples; MAD flags {n_mad}")
