"""Reproduce the baseline numbers and inspect the actual thresholds."""
import os, sys, numpy as np
sys.path.insert(0, "/data/jehc223/EMNLP2/src")
from quick_eval_all import otsu_threshold, gmm_threshold, metrics, load_scores_file, build_arrays
from data_utils import load_annotations

for d in ["MHClip_EN", "MHClip_ZH"]:
    base_d = f"/data/jehc223/EMNLP2/results/holistic_2b/{d}"
    train = load_scores_file(f"{base_d}/train_binary.jsonl")
    test = load_scores_file(f"{base_d}/test_binary.jsonl")
    ann = load_annotations(d)
    test_x, test_y = build_arrays(test, ann)
    train_arr = np.array(list(train.values()), dtype=float)
    pool = np.concatenate([train_arr, test_x])

    test_arr_only = test_x  # for test-only fitting
    print(f"\n=== {d} ===")
    # TF-Otsu (on test scores)
    t_otsu_test = otsu_threshold(test_arr_only)
    m = metrics(test_x, test_y, t_otsu_test)
    print(f"  TF-Otsu (Otsu on test scores)    t={t_otsu_test:.6f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}")
    # TF-GMM
    t_gmm_test = gmm_threshold(test_arr_only)
    m = metrics(test_x, test_y, t_gmm_test)
    print(f"  TF-GMM (GMM on test scores)      t={t_gmm_test:.6f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}")
    # Otsu on pool
    t_otsu_pool = otsu_threshold(pool)
    m = metrics(test_x, test_y, t_otsu_pool)
    print(f"  Otsu on pool                     t={t_otsu_pool:.6f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}")
    # GMM on pool
    t_gmm_pool = gmm_threshold(pool)
    m = metrics(test_x, test_y, t_gmm_pool)
    print(f"  GMM on pool                      t={t_gmm_pool:.6f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}")
    # Otsu on train
    t_otsu_train = otsu_threshold(train_arr)
    m = metrics(test_x, test_y, t_otsu_train)
    print(f"  Otsu on train                    t={t_otsu_train:.6f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}")
    # GMM on train
    t_gmm_train = gmm_threshold(train_arr)
    m = metrics(test_x, test_y, t_gmm_train)
    print(f"  GMM on train                     t={t_gmm_train:.6f}  acc={m['acc']:.4f}  mf={m['mf']:.4f}")
