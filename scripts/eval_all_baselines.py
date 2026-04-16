"""Walk all `results/<method>/<dataset>/test*.jsonl`, score each with
`eval_generative_predictions.eval_one()`, and emit a consolidated table
keyed by `(method, variant, dataset)`.

Outputs:
  - `docs/results_<stamp>.csv`      — flat CSV, one row per file
  - `docs/results_<stamp>.md`       — markdown-formatted pivot table
                                      (method × variant → per-dataset
                                      ACC / mF1) plus a full flat list.

The "method" is the first directory component under `results/`. The
"dataset" is the second component and must be one of
`{MHClip_EN, MHClip_ZH, HateMM, ImpliHateVid}` — directories that don't
match that list are skipped (e.g. `results/analysis`, `results/scores`,
`results/reproduction`). The "variant" is the filename stem with the
trailing `.jsonl` removed; for directories with a single test jsonl
variant is usually something like `test_naive` or `test_mars`, and for
sweep directories (`boundary_rescue/`) the variant captures the
sweep parameter string.

SKIP_VIDEOS is applied automatically inside `eval_one()`, so evaluation
is deterministic and matches the per-baseline eval numbers the
individual repro scripts emit.

Usage:
    python scripts/eval_all_baselines.py
    python scripts/eval_all_baselines.py --stamp 2026_04_15
    python scripts/eval_all_baselines.py --out-dir docs
    python scripts/eval_all_baselines.py --include-variants "test,test_naive,test_mars,test_procap,test_match,final"
"""

import argparse
import csv
import glob
import json
import os
import sys
import traceback
from datetime import datetime

_PROJECT_ROOT = "/data/jehc223/EMNLP2"
_NAIVE_BASELINE = os.path.join(_PROJECT_ROOT, "src", "naive_baseline")
sys.path.insert(0, _NAIVE_BASELINE)
from eval_generative_predictions import eval_one, ALL_DATASETS  # noqa: E402


def discover(results_root):
    """Return a sorted list of (method, dataset, variant, path) tuples.

    - method  = first path segment under `results/`
    - dataset = second path segment; must be in `ALL_DATASETS` or the
                entry is skipped
    - variant = filename stem (no `.jsonl`)
    - path    = absolute file path
    """
    entries = []
    pattern = os.path.join(results_root, "*", "*", "test*.jsonl")
    for p in glob.glob(pattern):
        rel = os.path.relpath(p, results_root)
        parts = rel.split(os.sep)
        if len(parts) != 3:
            continue
        method, dataset, fname = parts
        if dataset not in ALL_DATASETS:
            continue
        variant = fname[:-6] if fname.endswith(".jsonl") else fname
        entries.append((method, dataset, variant, p))
    entries.sort()
    return entries


def score_entry(method, dataset, variant, path):
    """Run eval_one and return a row-ready dict. Errors are captured."""
    try:
        r = eval_one(path, dataset)
    except Exception as e:
        return {
            "method": method,
            "dataset": dataset,
            "variant": variant,
            "path": path,
            "error": f"{type(e).__name__}: {str(e)[:160]}",
            "acc": None,
            "mf": None,
            "n_total": 0,
            "n_pos_gt": None,
            "n_pos_pred": None,
            "n_unparseable": None,
        }
    row = {
        "method": method,
        "dataset": dataset,
        "variant": variant,
        "path": path,
        "error": "",
        "acc": r.get("acc"),
        "mf": r.get("mf"),
        "n_total": r.get("n_total", 0),
        "n_pos_gt": r.get("n_pos_gt"),
        "n_pos_pred": r.get("n_pos_pred"),
        "n_unparseable": r.get("n_unparseable"),
    }
    return row


def write_csv(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cols = [
        "method",
        "variant",
        "dataset",
        "n_total",
        "n_pos_gt",
        "n_pos_pred",
        "n_unparseable",
        "acc",
        "mf",
        "path",
        "error",
    ]
    tmp = out_path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = dict(row)
            if out.get("acc") is not None:
                out["acc"] = f"{out['acc']:.4f}"
            if out.get("mf") is not None:
                out["mf"] = f"{out['mf']:.4f}"
            w.writerow(out)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)


def fmt_cell(acc, mf):
    if acc is None or mf is None:
        return "--"
    return f"{acc:.4f} / {mf:.4f}"


def write_markdown(rows, out_path, stamp):
    """Markdown report with a pivot (method,variant) × dataset and a
    flat listing. The pivot cell is `ACC / mF1`; missing cells show `--`.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Pivot: {(method, variant) -> {dataset -> row}}
    pivot = {}
    for row in rows:
        key = (row["method"], row["variant"])
        pivot.setdefault(key, {})[row["dataset"]] = row

    ordered_keys = sorted(pivot.keys())

    lines = []
    lines.append(f"# Consolidated baseline results — {stamp}")
    lines.append("")
    lines.append(
        "Generated by `scripts/eval_all_baselines.py`. All cells go "
        "through `eval_generative_predictions.eval_one()` with the "
        "canonical `SKIP_VIDEOS` filter. Each cell shows `ACC / mF1`."
    )
    lines.append("")
    lines.append(
        f"Source file count: **{len(rows)}** across **{len(pivot)}** "
        f"(method, variant) combinations."
    )
    lines.append("")

    # Pivot table
    lines.append("## Pivot — (method, variant) × dataset")
    lines.append("")
    header = (
        "| method | variant | "
        + " | ".join(ALL_DATASETS)
        + " |"
    )
    sep = (
        "|---|---|"
        + "|".join(["---"] * len(ALL_DATASETS))
        + "|"
    )
    lines.append(header)
    lines.append(sep)
    for method, variant in ordered_keys:
        cells = []
        for ds in ALL_DATASETS:
            r = pivot[(method, variant)].get(ds)
            if r is None:
                cells.append("--")
            elif r.get("error"):
                cells.append("err")
            else:
                cells.append(fmt_cell(r.get("acc"), r.get("mf")))
        lines.append(
            f"| {method} | {variant} | " + " | ".join(cells) + " |"
        )
    lines.append("")

    # Flat listing
    lines.append("## Flat listing")
    lines.append("")
    flat_hdr = (
        "| method | variant | dataset | n | n_pos_gt | n_pos_pred "
        "| n_unparseable | ACC | mF1 | error |"
    )
    flat_sep = "|---|---|---|---|---|---|---|---|---|---|"
    lines.append(flat_hdr)
    lines.append(flat_sep)
    for row in rows:
        acc_s = f"{row['acc']:.4f}" if row.get("acc") is not None else "--"
        mf_s = f"{row['mf']:.4f}" if row.get("mf") is not None else "--"
        err_s = row.get("error") or ""
        lines.append(
            f"| {row['method']} | {row['variant']} | {row['dataset']} | "
            f"{row.get('n_total', 0)} | {row.get('n_pos_gt', '')} | "
            f"{row.get('n_pos_pred', '')} | {row.get('n_unparseable', '')} | "
            f"{acc_s} | {mf_s} | {err_s} |"
        )
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Every ACC / mF1 pair is computed with `SKIP_VIDEOS` applied "
        "inside `eval_one()`."
    )
    lines.append(
        "- Missing cells (`--`) mean that `(method, variant, dataset)` "
        "combination has no `test*.jsonl` under `results/<method>/<dataset>/`."
    )
    lines.append(
        "- Error cells (`err`) mean `eval_one()` raised during scoring — "
        "see the flat listing for the `error` column and `path` for the "
        "source file."
    )

    body = "\n".join(lines) + "\n"
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        f.write(body)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Walk results/ and emit a consolidated CSV + markdown table"
    )
    parser.add_argument(
        "--results-root",
        default=os.path.join(_PROJECT_ROOT, "results"),
        help="Root of the results tree (default: /data/jehc223/EMNLP2/results)",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(_PROJECT_ROOT, "docs"),
        help="Directory to write the CSV + markdown reports",
    )
    parser.add_argument(
        "--stamp",
        default=datetime.now().strftime("%Y_%m_%d"),
        help="Timestamp suffix for the output filenames (default: today's YYYY_MM_DD)",
    )
    parser.add_argument(
        "--include-methods",
        default="",
        help=(
            "Comma-separated whitelist of method directory names. If "
            "empty, every method dir with at least one matching "
            "dataset subdir is included."
        ),
    )
    parser.add_argument(
        "--include-variants",
        default="",
        help=(
            "Comma-separated whitelist of variant stems (e.g. "
            "'test_naive,test_mars,test_procap,test_match,test_v2_winning'). "
            "If empty, every variant is included."
        ),
    )
    parser.add_argument(
        "--exclude-methods",
        default="",
        help="Comma-separated method directory names to skip",
    )
    args = parser.parse_args()

    include_methods = {m.strip() for m in args.include_methods.split(",") if m.strip()}
    include_variants = {v.strip() for v in args.include_variants.split(",") if v.strip()}
    exclude_methods = {m.strip() for m in args.exclude_methods.split(",") if m.strip()}

    entries = discover(args.results_root)
    if include_methods:
        entries = [e for e in entries if e[0] in include_methods]
    if exclude_methods:
        entries = [e for e in entries if e[0] not in exclude_methods]
    if include_variants:
        entries = [e for e in entries if e[2] in include_variants]

    print(
        f"Discovered {len(entries)} eligible test*.jsonl files under "
        f"{args.results_root}"
    )
    if not entries:
        print("Nothing to score. Exiting.")
        return

    rows = []
    n_errors = 0
    for i, (method, dataset, variant, path) in enumerate(entries, 1):
        row = score_entry(method, dataset, variant, path)
        if row.get("error"):
            n_errors += 1
        rows.append(row)
        if i % 50 == 0 or i == len(entries):
            print(
                f"  [{i}/{len(entries)}] scored; errors so far: {n_errors}"
            )

    csv_path = os.path.join(args.out_dir, f"results_{args.stamp}.csv")
    md_path = os.path.join(args.out_dir, f"results_{args.stamp}.md")
    write_csv(rows, csv_path)
    write_markdown(rows, md_path, args.stamp)

    print()
    print(f"Wrote CSV      : {csv_path}")
    print(f"Wrote Markdown : {md_path}")
    print(f"Scored         : {len(rows)}   Errors: {n_errors}")


if __name__ == "__main__":
    main()
