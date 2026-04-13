"""
Shared evaluator for generative-prediction JSONL files (naive + MARS).

Each input record must contain `video_id` and `pred` ∈ {0, 1, -1}. The
helper loads annotations, applies dataset-aware label collapse, and prints
acc / macro-F1 / unparseable counts / pred distribution.

Usage:
  python src/naive_baseline/eval_generative_predictions.py \
    --input results/naive_2b/MHClip_EN/test_naive.jsonl \
    --dataset MHClip_EN
  python src/naive_baseline/eval_generative_predictions.py \
    --out results/analysis/naive_2b_eval.json \
    --naive    # auto: all 3 datasets from results/naive_2b/
  python src/naive_baseline/eval_generative_predictions.py \
    --out results/analysis/mars_2b_eval.json \
    --mars     # auto: all 3 datasets from results/mars_2b/
"""

import argparse
import json
import os
import sys

# Import data_utils from sibling src/our_method/
_OUR_METHOD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "our_method")
sys.path.insert(0, _OUR_METHOD)
from data_utils import load_annotations, SKIP_VIDEOS  # noqa: E402

ALL_DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM"]


def collapse_label(dataset, gt_label):
    if dataset == "HateMM":
        return 1 if gt_label == "Hate" else 0
    return 1 if gt_label in ("Hateful", "Offensive") else 0


def eval_one(input_path, dataset):
    annotations = load_annotations(dataset)
    skip_set = SKIP_VIDEOS.get(dataset, set())
    y_true, y_pred, unparseable = [], [], 0
    n_records = 0
    pred_dist = {-1: 0, 0: 0, 1: 0}
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            vid = r.get("video_id")
            pred = r.get("pred")
            if vid is None or pred is None:
                continue
            if vid in skip_set:
                continue
            if vid not in annotations:
                continue
            n_records += 1
            pred_dist[pred] = pred_dist.get(pred, 0) + 1
            gt = collapse_label(dataset, annotations[vid]["label"])
            y_true.append(gt)
            if pred == -1:
                unparseable += 1
                # Fall back to 0 for metric computation; the count is surfaced separately.
                y_pred.append(0)
            else:
                y_pred.append(pred)

    if n_records == 0:
        return {
            "input": input_path, "dataset": dataset,
            "n_total": 0, "n_unparseable": 0,
            "acc": None, "mf": None, "n_pos_pred": 0,
        }

    n = len(y_true)
    tp = sum(1 for p, t in zip(y_pred, y_true) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(y_pred, y_true) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(y_pred, y_true) if p == 0 and t == 1)
    tn = sum(1 for p, t in zip(y_pred, y_true) if p == 0 and t == 0)

    acc = (tp + tn) / n
    p_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f_pos = 2 * p_pos * r_pos / (p_pos + r_pos) if (p_pos + r_pos) > 0 else 0.0
    p_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    r_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f_neg = 2 * p_neg * r_neg / (p_neg + r_neg) if (p_neg + r_neg) > 0 else 0.0
    mf = (f_pos + f_neg) / 2

    return {
        "input": input_path,
        "dataset": dataset,
        "n_total": n,
        "n_pos_gt": sum(y_true),
        "n_pos_pred": sum(1 for p in y_pred if p == 1),
        "n_unparseable": unparseable,
        "pred_dist": pred_dist,
        "acc": acc,
        "mf": mf,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "f_pos": f_pos, "f_neg": f_neg,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Single input JSONL")
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--naive", action="store_true",
                        help="Auto-eval results/naive_2b/*/test_naive.jsonl")
    parser.add_argument("--mars", action="store_true",
                        help="Auto-eval results/mars_2b/*/test_mars.jsonl")
    parser.add_argument("--out", help="Write summary JSON to this path")
    args = parser.parse_args()

    project_root = "/data/jehc223/EMNLP2"
    results = []

    if args.naive:
        for ds in ALL_DATASETS:
            p = f"{project_root}/results/naive_2b/{ds}/test_naive.jsonl"
            if os.path.isfile(p):
                results.append(eval_one(p, ds))
    elif args.mars:
        for ds in ALL_DATASETS:
            p = f"{project_root}/results/mars_2b/{ds}/test_mars.jsonl"
            if os.path.isfile(p):
                results.append(eval_one(p, ds))
    elif args.input and args.dataset:
        results.append(eval_one(args.input, args.dataset))
    else:
        parser.error("Provide --naive / --mars / (--input and --dataset).")

    hdr = f"{'dataset':10s}  {'ACC':>7}  {'mF1':>7}  {'pos':>6}  {'unpars':>7}  {'n':>5}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if r["acc"] is None:
            print(f"{r['dataset']:10s}  (no data)")
            continue
        print(f"{r['dataset']:10s}  {r['acc']:.4f}  {r['mf']:.4f}  "
              f"{r['n_pos_pred']:>6d}  {r['n_unparseable']:>7d}  {r['n_total']:>5d}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"results": results}, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
