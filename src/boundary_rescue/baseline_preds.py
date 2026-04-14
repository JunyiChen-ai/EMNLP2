"""
Task 0: Reproduce the current baseline predictions per dataset.

Reads the 3 score jsonls, applies the per-dataset TF threshold
(Otsu/GMM/li_lee), writes baseline_preds.jsonl per dataset, and pins
the ZH pre-repro baseline acc/mf1 to `zh_prerepro_baseline.json` so
the rescue loop cannot drift the ZH strict-beat target.

EN and HateMM baselines are verified against known numbers (±1e-4).
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "our_method"))
sys.path.insert(0, os.path.join(_HERE, "..", "naive_baseline"))

from thresholds import CRITERION  # noqa: E402
from quick_eval_all import load_scores_file  # noqa: E402
from data_utils import load_annotations, SKIP_VIDEOS  # noqa: E402
from eval_generative_predictions import collapse_label  # noqa: E402

PROJECT_ROOT = "/data/jehc223/EMNLP2"
OUT_ROOT = os.path.join(PROJECT_ROOT, "results", "boundary_rescue")

SCORE_FILES = {
    "MHClip_EN": os.path.join(
        PROJECT_ROOT, "results", "holistic_2b", "MHClip_EN", "test_binary.jsonl"
    ),
    "MHClip_ZH": os.path.join(
        PROJECT_ROOT,
        "results",
        "holistic_2b",
        "MHClip_ZH",
        "test_binary.jsonl.prerepro_20260413",
    ),
    "HateMM": os.path.join(
        PROJECT_ROOT, "results", "holistic_2b", "HateMM", "test_binary.jsonl"
    ),
}

KNOWN_BASELINES = {
    "MHClip_EN": (0.7640, 0.6532),
    "HateMM": (0.8047, 0.7930),
}


def dataset_metrics(video_ids, scores, preds, dataset):
    ann = load_annotations(dataset)
    skip = SKIP_VIDEOS.get(dataset, set())
    ys_true, ys_pred = [], []
    for vid, s, p in zip(video_ids, scores, preds):
        if vid in skip:
            continue
        if vid not in ann:
            continue
        y = collapse_label(dataset, ann[vid]["label"])
        ys_true.append(y)
        ys_pred.append(int(p))
    n = len(ys_true)
    tp = sum(1 for p, y in zip(ys_pred, ys_true) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(ys_pred, ys_true) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(ys_pred, ys_true) if p == 0 and y == 1)
    tn = sum(1 for p, y in zip(ys_pred, ys_true) if p == 0 and y == 0)
    acc = (tp + tn) / n if n else 0.0
    p_pos = tp / (tp + fp) if (tp + fp) else 0.0
    r_pos = tp / (tp + fn) if (tp + fn) else 0.0
    f_pos = 2 * p_pos * r_pos / (p_pos + r_pos) if (p_pos + r_pos) else 0.0
    p_neg = tn / (tn + fn) if (tn + fn) else 0.0
    r_neg = tn / (tn + fp) if (tn + fp) else 0.0
    f_neg = 2 * p_neg * r_neg / (p_neg + r_neg) if (p_neg + r_neg) else 0.0
    mf = (f_pos + f_neg) / 2
    return {"n": n, "acc": acc, "mf1": mf, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def process(dataset):
    path = SCORE_FILES[dataset]
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    score_dict = load_scores_file(path)
    # Preserve file order so the S1 atom rule is deterministic
    with open(path) as f:
        order = []
        seen = set()
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            vid = r.get("video_id")
            if vid is None or vid in seen:
                continue
            seen.add(vid)
            order.append(vid)
    video_ids = [v for v in order if v in score_dict]
    scores = np.array([score_dict[v] for v in video_ids], dtype=float)

    crit_name, crit_fn = CRITERION[dataset]
    thr = crit_fn(scores)
    preds = (scores >= thr).astype(int)

    met = dataset_metrics(video_ids, scores, preds, dataset)

    out_dir = os.path.join(OUT_ROOT, dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "baseline_preds.jsonl")
    with open(out_path, "w") as f:
        for vid, s, p in zip(video_ids, scores, preds):
            f.write(
                json.dumps(
                    {
                        "video_id": vid,
                        "score": float(s),
                        "threshold": float(thr),
                        "pred_baseline": int(p),
                        "criterion": crit_name,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return {
        "dataset": dataset,
        "criterion": crit_name,
        "threshold": float(thr),
        "n_total_records": len(video_ids),
        "metrics": met,
        "out_path": out_path,
        "score_path": path,
    }


def main():
    rows = []
    for ds in ["MHClip_EN", "MHClip_ZH", "HateMM"]:
        row = process(ds)
        rows.append(row)

    # Print sanity table
    print()
    print(
        f"{'dataset':<10} {'crit':<8} {'N':>5} {'t':>8} {'ACC':>8} {'mF1':>8}"
        f"  {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}"
    )
    print("-" * 72)
    all_ok = True
    for r in rows:
        m = r["metrics"]
        print(
            f"{r['dataset']:<10} {r['criterion']:<8} {m['n']:>5d} {r['threshold']:>8.4f} "
            f"{m['acc']:>8.4f} {m['mf1']:>8.4f}  "
            f"{m['tp']:>3d} {m['fp']:>3d} {m['fn']:>3d} {m['tn']:>3d}"
        )
        if r["dataset"] in KNOWN_BASELINES:
            exp_acc, exp_mf1 = KNOWN_BASELINES[r["dataset"]]
            dacc = abs(m["acc"] - exp_acc)
            dmf1 = abs(m["mf1"] - exp_mf1)
            if dacc > 1e-4 or dmf1 > 1e-4:
                all_ok = False
                print(
                    f"  ** MISMATCH on {r['dataset']}: expected "
                    f"{exp_acc:.4f}/{exp_mf1:.4f} (Δacc={dacc:.4f}, Δmf1={dmf1:.4f})"
                )
    print()

    # Pin the ZH pre-repro target
    zh_row = next(r for r in rows if r["dataset"] == "MHClip_ZH")
    zh_target_path = os.path.join(OUT_ROOT, "zh_prerepro_baseline.json")
    with open(zh_target_path, "w") as f:
        json.dump(
            {
                "acc": zh_row["metrics"]["acc"],
                "mf1": zh_row["metrics"]["mf1"],
                "threshold": zh_row["threshold"],
                "n": zh_row["metrics"]["n"],
                "criterion": zh_row["criterion"],
                "score_file": zh_row["score_path"],
            },
            f,
            indent=2,
        )
    print(f"Pinned ZH pre-repro target to: {zh_target_path}")
    print(
        f"  acc={zh_row['metrics']['acc']:.4f}  "
        f"mf1={zh_row['metrics']['mf1']:.4f}  "
        f"n={zh_row['metrics']['n']}"
    )

    if not all_ok:
        print("\n** BASELINE REPRODUCTION FAILED — halting. **")
        sys.exit(2)


if __name__ == "__main__":
    main()
