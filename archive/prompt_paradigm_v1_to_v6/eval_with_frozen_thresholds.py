"""
prompt_paradigm evaluator — reuses frozen back-half functions from src/quick_eval_all.py.

Reads:
  results/prompt_paradigm/{MHClip_EN,MHClip_ZH}/{test,train}_obsjudge.jsonl

Computes for each dataset:
  - Oracle (diagnostic, uses test labels) — NOT used for selection
  - TF-Otsu, TF-GMM (fit on test scores, apply on test)
  - TR-Otsu, TR-GMM (fit on train scores, apply on test) — label-free

Writes:
  results/prompt_paradigm/report.json
  results/analysis/prompt_paradigm_report.md

No test labels are used for threshold selection. The comparison to baseline
is purely at report-time (the baseline numbers are hard-coded for reference
from STATE_ARCHIVE.md / current_baseline memory).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_utils import load_annotations
from quick_eval_all import (
    otsu_threshold,
    gmm_threshold,
    metrics,
    load_scores_file,
    build_arrays,
)

PROJECT_ROOT = "/data/jehc223/EMNLP2"

BASELINE = {
    "MHClip_EN": {"acc": 0.7640, "mf": 0.653, "source": "2B binary_nodef TF-Otsu"},
    "MHClip_ZH": {"acc": 0.8121, "mf": 0.787, "source": "2B binary_nodef TF-GMM"},
}


def evaluate_dataset(dataset):
    ann = load_annotations(dataset)
    base = os.path.join(PROJECT_ROOT, "results", "prompt_paradigm", dataset)
    test_path = os.path.join(base, "test_obsjudge.jsonl")
    train_path = os.path.join(base, "train_obsjudge.jsonl")

    if not os.path.isfile(test_path):
        return {"error": f"missing {test_path}"}

    test_dict = load_scores_file(test_path)
    test_x, test_y = build_arrays(test_dict, ann)
    if len(test_x) == 0:
        return {"error": "no test scores overlap annotations"}

    result = {
        "dataset": dataset,
        "n_test": int(len(test_x)),
        "n_test_pos": int(test_y.sum()),
    }

    # Diagnostic oracle (uses labels — NOT used for selection, only reported)
    best_acc = 0.0
    best_t = 0.5
    for t in np.arange(0.0, 1.001, 0.01):
        m = metrics(test_x, test_y, float(t))
        if m["acc"] > best_acc:
            best_acc = m["acc"]
            best_t = float(t)
    result["oracle"] = {**metrics(test_x, test_y, best_t), "t": best_t}

    # TF cells
    try:
        t = otsu_threshold(test_x)
        result["tf_otsu"] = {**metrics(test_x, test_y, t), "t": t}
    except Exception as e:
        result["tf_otsu"] = {"err": str(e)}
    try:
        t = gmm_threshold(test_x)
        result["tf_gmm"] = {**metrics(test_x, test_y, t), "t": t}
    except Exception as e:
        result["tf_gmm"] = {"err": str(e)}

    # TR cells (requires train file)
    if os.path.isfile(train_path):
        train_dict = load_scores_file(train_path)
        train_scores = np.array([v for v in train_dict.values() if v is not None])
        result["n_train"] = int(len(train_scores))
        try:
            t = otsu_threshold(train_scores)
            result["tr_otsu"] = {**metrics(test_x, test_y, t), "t": t}
        except Exception as e:
            result["tr_otsu"] = {"err": str(e)}
        try:
            t = gmm_threshold(train_scores)
            result["tr_gmm"] = {**metrics(test_x, test_y, t), "t": t}
        except Exception as e:
            result["tr_gmm"] = {"err": str(e)}

    # Per-class score stats (for diagnostics — NOT used for selection)
    pos = test_x[test_y == 1]
    neg = test_x[test_y == 0]
    result["pos_mean"] = float(pos.mean()) if len(pos) else None
    result["neg_mean"] = float(neg.mean()) if len(neg) else None
    result["pos_std"] = float(pos.std()) if len(pos) else None
    result["neg_std"] = float(neg.std()) if len(neg) else None

    return result


def fmt(m):
    if not m or "err" in m:
        return "ERR"
    return f"{m['acc']:.4f}/{m['mf']:.3f}"


def choose_unified(en_row, zh_row):
    """Pick a single (TR/TF x Otsu/GMM) pair that beats baseline on BOTH.

    Returns (cell_key, {'EN': ..., 'ZH': ...}) or (None, ...) if no cell passes.
    """
    cells = ["tf_otsu", "tf_gmm", "tr_otsu", "tr_gmm"]
    passing = []
    for c in cells:
        e = en_row.get(c, {})
        z = zh_row.get(c, {})
        if "err" in e or "err" in z or not e or not z:
            continue
        en_ok = e["acc"] > BASELINE["MHClip_EN"]["acc"] and e["mf"] >= BASELINE["MHClip_EN"]["mf"]
        zh_ok = z["acc"] > BASELINE["MHClip_ZH"]["acc"] and z["mf"] >= BASELINE["MHClip_ZH"]["mf"]
        passing.append((c, e, z, en_ok, zh_ok, en_ok and zh_ok))
    unified = [p for p in passing if p[5]]
    if unified:
        # rank by min(acc) margin
        unified.sort(
            key=lambda p: min(
                p[1]["acc"] - BASELINE["MHClip_EN"]["acc"],
                p[2]["acc"] - BASELINE["MHClip_ZH"]["acc"],
            ),
            reverse=True,
        )
        return unified[0]
    return None


def main():
    out_dir = os.path.join(PROJECT_ROOT, "results", "prompt_paradigm")
    os.makedirs(out_dir, exist_ok=True)
    analysis_dir = os.path.join(PROJECT_ROOT, "results", "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    en = evaluate_dataset("MHClip_EN")
    zh = evaluate_dataset("MHClip_ZH")
    report = {"baseline": BASELINE, "MHClip_EN": en, "MHClip_ZH": zh}
    unified = choose_unified(en, zh) if "error" not in en and "error" not in zh else None
    if unified:
        cell, e, z, _, _, _ = unified
        report["unified_winner"] = {
            "cell": cell,
            "EN": {"acc": e["acc"], "mf": e["mf"], "t": e.get("t")},
            "ZH": {"acc": z["acc"], "mf": z["mf"], "t": z.get("t")},
        }
    else:
        report["unified_winner"] = None

    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Markdown summary
    lines = []
    lines.append("# prompt_paradigm v1 report — Observe-then-Judge\n")
    lines.append("## Baseline (must beat on BOTH under one unified cell)\n")
    lines.append(f"- MHClip_EN: ACC={BASELINE['MHClip_EN']['acc']:.4f} "
                 f"macro-F1={BASELINE['MHClip_EN']['mf']:.3f} ({BASELINE['MHClip_EN']['source']})")
    lines.append(f"- MHClip_ZH: ACC={BASELINE['MHClip_ZH']['acc']:.4f} "
                 f"macro-F1={BASELINE['MHClip_ZH']['mf']:.3f} ({BASELINE['MHClip_ZH']['source']})\n")

    lines.append("## Per-dataset cells (ACC / macro-F1)\n")
    lines.append("| Dataset | N | Oracle | TF-Otsu | TF-GMM | TR-Otsu | TR-GMM |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, row in [("MHClip_EN", en), ("MHClip_ZH", zh)]:
        if "error" in row:
            lines.append(f"| {name} | — | ERR: {row['error']} | | | | |")
            continue
        lines.append(
            f"| {name} | {row['n_test']} | {fmt(row.get('oracle'))} "
            f"| {fmt(row.get('tf_otsu'))} | {fmt(row.get('tf_gmm'))} "
            f"| {fmt(row.get('tr_otsu'))} | {fmt(row.get('tr_gmm'))} |"
        )
    lines.append("")

    lines.append("## Per-class score statistics\n")
    for name, row in [("MHClip_EN", en), ("MHClip_ZH", zh)]:
        if "error" in row:
            continue
        lines.append(
            f"- {name}: pos_mean={row['pos_mean']:.4f} (std {row['pos_std']:.4f}), "
            f"neg_mean={row['neg_mean']:.4f} (std {row['neg_std']:.4f}), "
            f"margin={row['pos_mean'] - row['neg_mean']:+.4f}"
        )
    lines.append("")

    lines.append("## Unified (TR/TF × Otsu/GMM) winner\n")
    if report["unified_winner"]:
        uw = report["unified_winner"]
        lines.append(f"Selected cell: **{uw['cell']}**")
        lines.append(f"- EN: ACC={uw['EN']['acc']:.4f} macro-F1={uw['EN']['mf']:.3f} "
                     f"(Δ ACC={uw['EN']['acc']-BASELINE['MHClip_EN']['acc']:+.4f})")
        lines.append(f"- ZH: ACC={uw['ZH']['acc']:.4f} macro-F1={uw['ZH']['mf']:.3f} "
                     f"(Δ ACC={uw['ZH']['acc']-BASELINE['MHClip_ZH']['acc']:+.4f})")
        lines.append("\n**Gate-2 status: beats baseline on both datasets under one unified cell.**")
    else:
        lines.append("No unified cell beats baseline on BOTH datasets.")
        lines.append("\n**Gate-2 status: MISS. Return to Gate 1 with v2.**")
    lines.append("")

    report_md = os.path.join(analysis_dir, "prompt_paradigm_report.md")
    with open(report_md, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nWrote {os.path.join(out_dir, 'report.json')}")
    print(f"Wrote {report_md}")


if __name__ == "__main__":
    main()
