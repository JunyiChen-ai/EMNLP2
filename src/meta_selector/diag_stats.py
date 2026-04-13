"""Compare distribution statistics between EN and ZH to find a discriminating feature."""
import os, sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, metrics, load_scores_file, build_arrays
from data_utils import load_annotations

for dataset in ["MHClip_EN", "MHClip_ZH"]:
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{dataset}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    train_arr = np.array(list(train.values()), dtype=float)
    test_arr = np.array(list(test.values()), dtype=float)
    pool = np.concatenate([train_arr, test_arr])
    print(f"\n=== {dataset} ===")
    for name, x in [("pool", pool), ("train", train_arr), ("test", test_arr)]:
        print(f"  {name}  n={len(x)}  mean={x.mean():.4f}  median={np.median(x):.4f}  std={x.std():.4f}")
        print(f"         q25={np.quantile(x,.25):.4f}  q50={np.quantile(x,.5):.4f}  q75={np.quantile(x,.75):.4f}  q90={np.quantile(x,.9):.4f}")
        print(f"         skew={((x-x.mean())**3).mean()/(x.std()**3):.3f}  kurt={((x-x.mean())**4).mean()/(x.std()**4):.3f}")
        print(f"         frac<0.1={(x<0.1).mean():.3f}  frac>0.5={(x>0.5).mean():.3f}")
        print(f"         otsu_t={otsu_threshold(x):.4f}  gmm_t={gmm_threshold(x):.4f}")
    # what q corresponds to otsu? to gmm?
    ot = otsu_threshold(pool)
    gm = gmm_threshold(pool)
    print(f"  pool otsu_t={ot:.4f}  q_otsu={(pool < ot).mean():.4f}")
    print(f"  pool gmm_t={gm:.4f}  q_gmm={(pool < gm).mean():.4f}")
