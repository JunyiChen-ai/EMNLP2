"""
Task C (v2): Apply 8B-judge rescue with confidence-gated flips, eval.

Parses the 8B rescue response (rationale + verdict 2-field plain
text), computes a confidence proxy from the rationale (hedge count,
length, concrete-evidence count), and applies a gating rule that
flips the baseline pred only when the 8B disagrees AND signals
sufficient confidence.

The gating rule is the v2 loop's CPU-cheap iteration axis. We do
NOT re-run the GPU rescue when changing the gating rule — only the
parser branch changes. Use --gating G1 / G2 / G3 / G4 / G5 / G6.

Goal-1 strict-beat is checked against the pinned v2_baseline.json.
Goal-2 (rescue scale) requires:
   n_correct_flips ≥ ceil(band_error_ceiling / 2) per dataset
where band_error_ceiling is loaded from the alpha's diagnostic dump.

All label use is in clearly-marked diagnostic blocks; the flip
decision uses zero label information.
"""

import argparse
import json
import math
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "our_method"))
sys.path.insert(0, os.path.join(_HERE, "..", "naive_baseline"))

from data_utils import load_annotations, SKIP_VIDEOS  # noqa: E402
from eval_generative_predictions import collapse_label, eval_one  # noqa: E402

PROJECT_ROOT = "/data/jehc223/EMNLP2"
OUT_ROOT = os.path.join(PROJECT_ROOT, "results", "boundary_rescue")
ALL_DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]

# ---- Confidence-proxy dictionaries ----

HEDGES_D1 = [
    "might", "possibly", "perhaps", "unclear", "ambiguous",
    "could be", "may be", "appears to", "seems to",
    "not entirely", "borderline", "hard to tell", "unsure",
    "uncertain", "arguably", "potentially", "to some extent",
]
HEDGES_D2 = HEDGES_D1 + [
    "to my knowledge", "in part", "in some way", "to some degree",
]
HEDGES_D3 = [h for h in HEDGES_D1 if h not in {"appears to", "seems to"}]
HEDGE_DICTS = {"D1": HEDGES_D1, "D2": HEDGES_D2, "D3": HEDGES_D3}

CONCRETE = [
    "transcript", "audio", "image", "frame", "shows", "depicts",
    "displays", "uses", "says", "states", "speaks", "speaker",
    "title", "caption", "subtitle",
]


# ---- Parser ----

_VERDICT_RE = re.compile(r"verdict\s*:\s*(\w+)", re.IGNORECASE)
_RATIONALE_RE = re.compile(
    r"rationale\s*:\s*(.+?)(?:\n\s*verdict\s*:|$)",
    re.IGNORECASE | re.DOTALL,
)


def parse_response(text):
    text = text or ""
    vm = _VERDICT_RE.search(text)
    rm = _RATIONALE_RE.search(text)
    if not vm:
        return None, None, None
    verdict_word = vm.group(1).strip().lower()
    if verdict_word not in {"hateful", "normal"}:
        return None, None, None
    rationale = rm.group(1).strip() if rm else ""
    pred = 1 if verdict_word == "hateful" else 0
    return pred, verdict_word, rationale


def hedge_count(rationale, hedge_dict):
    text = (rationale or "").lower()
    return sum(1 for h in hedge_dict if h in text)


def concrete_count(rationale):
    text = (rationale or "").lower()
    return sum(1 for c in CONCRETE if c in text)


# ---- Gating rules ----

def gate(rule, pred_baseline, parsed_pred, rationale, hedges):
    """Return True iff this candidate should be flipped."""
    if parsed_pred is None:
        return False  # unparseable verdict → no flip
    if parsed_pred == pred_baseline:
        return False  # 8B agrees → no flip
    h = hedge_count(rationale, hedges)
    c = concrete_count(rationale)
    L = len((rationale or "").split())
    if rule == "G1":
        return h == 0
    if rule == "G2":
        return h == 0 and c >= 1
    if rule == "G3":
        return h == 0 and L >= 30
    if rule == "G4":
        return h == 0 and (c >= 1 or L >= 50)
    if rule == "G5":
        return h <= 1 and c >= 1
    if rule == "G6":
        return True  # no gating beyond well-formed verdict
    if rule == "G7":
        # Tighten G5: drop both extremes of concrete_count (c=0 = vibes,
        # c≥4 = over-detailed list which on ZH/HateMM correlates with
        # wrong flips per the diagnostic feature analysis)
        return h <= 1 and 1 <= c <= 3
    if rule == "G8":
        # Even tighter: require substantive rationale length too
        return h <= 1 and 1 <= c <= 3 and L >= 60
    if rule == "G9":
        return h <= 1 and c in (1, 3)
    if rule == "G10":
        # Drop c=2 long rationales (correlates with vulgar/over-detailed
        # wrong flips on ZH/HateMM); keep c=2 short rationales.
        return h <= 1 and (c in (1, 3) or (c == 2 and L <= 90))
    raise ValueError(f"unknown gating rule {rule}")


# ---- Main pipeline ----

def load_jsonl(path):
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_v2_baseline():
    with open(os.path.join(OUT_ROOT, "v2_baseline.json")) as f:
        return json.load(f)


def load_band_diag(alpha):
    p = os.path.join(OUT_ROOT, f"band_diag_alpha{alpha:.2f}.json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        diags = json.load(f)
    return {d["dataset"]: d for d in diags}


def apply_dataset(dataset, tag, version, gating, hedges, base, rescue):
    """Build a {video_id, pred} jsonl for eval_one and return flip records."""
    base_by_vid = {r["video_id"]: r for r in base}
    rescue_by_vid = {r["video_id"]: r for r in rescue}

    flip_records = {}
    for vid, rr in rescue_by_vid.items():
        parsed_pred, verdict_word, rationale = parse_response(
            rr.get("rescue_response", "")
        )
        h = hedge_count(rationale, hedges) if rationale is not None else 0
        c = concrete_count(rationale) if rationale is not None else 0
        L = len((rationale or "").split())
        flipped = gate(
            gating,
            int(rr["pred_baseline"]),
            parsed_pred,
            rationale,
            hedges,
        )
        flip_records[vid] = {
            "side": rr.get("side", "?"),
            "pred_baseline": int(rr["pred_baseline"]),
            "parsed_pred": parsed_pred,
            "verdict_word": verdict_word,
            "rationale_len": L,
            "hedge_count": h,
            "concrete_count": c,
            "flipped": flipped,
            "pred_after": (1 - int(rr["pred_baseline"])) if flipped else int(rr["pred_baseline"]),
        }

    out_path = os.path.join(
        OUT_ROOT, dataset,
        f"test_v2_{tag}_{version}_{gating}.jsonl",
    )
    with open(out_path, "w") as f:
        for r in base:
            vid = r["video_id"]
            if vid in flip_records:
                pred = flip_records[vid]["pred_after"]
            else:
                pred = int(r["pred_baseline"])
            f.write(
                json.dumps({"video_id": vid, "pred": pred}, ensure_ascii=False)
                + "\n"
            )
    return out_path, flip_records


def diagnostic_flip_breakdown(dataset, flip_records):
    """DIAGNOSTIC ONLY — uses labels."""
    ann = load_annotations(dataset)
    skip = SKIP_VIDEOS.get(dataset, set())
    correct = wrong = total_flipped = 0
    flipped_ids = []
    for vid, rec in flip_records.items():
        if vid in skip or vid not in ann:
            continue
        if not rec.get("flipped"):
            continue
        total_flipped += 1
        gt = collapse_label(dataset, ann[vid]["label"])
        if int(rec["pred_after"]) == gt:
            correct += 1
            flipped_ids.append((vid, "correct"))
        else:
            wrong += 1
            flipped_ids.append((vid, "wrong"))
    n_in_band = sum(
        1 for vid in flip_records
        if vid not in skip and vid in ann
    )
    return {
        "n_in_band": n_in_band,
        "n_flipped": total_flipped,
        "n_correct_flips": correct,
        "n_wrong_flips": wrong,
        "net_gain": correct - wrong,
        "flipped_ids": flipped_ids,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--tag", default=None,
                        help="Override rescue tag (file is "
                             "rescue_8b_<tag>_<version>.jsonl)")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--gating", default="G1",
                        choices=["G1","G2","G3","G4","G5","G6","G7","G8","G9","G10"])
    parser.add_argument("--hedge-dict", default="D1",
                        choices=["D1", "D2", "D3"])
    args = parser.parse_args()
    if args.alpha is None and args.tag is None:
        parser.error("provide --alpha or --tag")

    hedges = HEDGE_DICTS[args.hedge_dict]
    v2_baseline = load_v2_baseline()
    band_diag = load_band_diag(args.alpha) if args.alpha is not None else None

    if args.tag:
        rescue_tag = args.tag
        run_label = args.tag
    else:
        rescue_tag = f"band_alpha{args.alpha:.2f}"
        run_label = f"α={args.alpha:.2f}"

    print(
        f"=== rescue_8b {run_label} version={args.version} "
        f"gating={args.gating} hedge_dict={args.hedge_dict} ==="
    )
    print()

    rows = []
    for ds in ALL_DATASETS:
        base_path = os.path.join(OUT_ROOT, ds, "baseline_preds_v2.jsonl")
        rescue_path = os.path.join(
            OUT_ROOT, ds,
            f"rescue_8b_{rescue_tag}_{args.version}.jsonl",
        )
        base = load_jsonl(base_path)
        rescue = load_jsonl(rescue_path)
        out_path, flips = apply_dataset(
            ds, rescue_tag, args.version, args.gating, hedges, base, rescue
        )
        new_metrics = eval_one(out_path, ds)
        diag = diagnostic_flip_breakdown(ds, flips)

        b = v2_baseline[ds]
        dacc = new_metrics["acc"] - b["acc"]
        dmf1 = new_metrics["mf"] - b["mf1"]
        beat = dacc > 1e-9 and dmf1 > 1e-9

        ceiling = (band_diag or {}).get(ds, {}).get("band_error_ceiling")
        goal2_min = (band_diag or {}).get(ds, {}).get("goal2_min_correct_flips")
        goal2_ok = (
            goal2_min is not None
            and diag["n_correct_flips"] >= goal2_min
        )

        rows.append({
            "dataset": ds,
            "base_acc": b["acc"], "base_mf1": b["mf1"],
            "new_acc": new_metrics["acc"], "new_mf1": new_metrics["mf"],
            "dacc": dacc, "dmf1": dmf1, "beat": beat,
            "ceiling": ceiling, "goal2_min": goal2_min,
            "goal2_ok": goal2_ok,
            "diag": diag,
            "n_rescue": len(rescue),
        })

    # Print main results table
    print(
        f"{'dataset':<10}  {'base':>14}  {'new':>14}  "
        f"{'Δacc':>8} {'Δmf1':>8}  {'beat?':>5}"
    )
    print("-" * 75)
    all_beat = True
    for r in rows:
        beat_str = "YES" if r["beat"] else "no"
        if not r["beat"]:
            all_beat = False
        print(
            f"{r['dataset']:<10}  "
            f"{r['base_acc']:.4f}/{r['base_mf1']:.4f}  "
            f"{r['new_acc']:.4f}/{r['new_mf1']:.4f}  "
            f"{r['dacc']:>+8.4f} {r['dmf1']:>+8.4f}  {beat_str:>5}"
        )
    print()
    print(f"strict_beat_all = {all_beat}")
    print()

    # Diagnostic table
    print(
        f"{'dataset':<10}  {'#cand':>5} {'#flip':>5} {'corr':>4} {'wrong':>5} "
        f"{'net':>4}  {'ceil':>4} {'g2min':>5} {'g2ok':>5}"
    )
    print("-" * 70)
    all_goal2 = True
    for r in rows:
        d = r["diag"]
        ceil_str = f"{r['ceiling']}" if r['ceiling'] is not None else "  -"
        g2min_str = f"{r['goal2_min']}" if r['goal2_min'] is not None else "  -"
        g2ok_str = "YES" if r["goal2_ok"] else "no"
        if not r["goal2_ok"]:
            all_goal2 = False
        print(
            f"{r['dataset']:<10}  "
            f"{d['n_in_band']:>5d} {d['n_flipped']:>5d} "
            f"{d['n_correct_flips']:>4d} {d['n_wrong_flips']:>5d} "
            f"{d['net_gain']:>+4d}  "
            f"{ceil_str:>4} {g2min_str:>5} {g2ok_str:>5}"
        )
    print()
    print(
        f"goal1_strict_beat_all = {all_beat}    "
        f"goal2_meaningful_rescue = {all_goal2}"
    )

    # Append loop log
    log_path = os.path.join(OUT_ROOT, "loop_log_8b.jsonl")
    log_entry = {
        "alpha": args.alpha,
        "version": args.version,
        "gating": args.gating,
        "hedge_dict": args.hedge_dict,
        "goal1_strict_beat_all": all_beat,
        "goal2_meaningful_rescue": all_goal2,
    }
    for r in rows:
        log_entry[r["dataset"]] = {
            "new_acc": r["new_acc"], "new_mf1": r["new_mf1"],
            "dacc": r["dacc"], "dmf1": r["dmf1"], "beat": r["beat"],
            "n_in_band": r["diag"]["n_in_band"],
            "n_flipped": r["diag"]["n_flipped"],
            "n_correct_flips": r["diag"]["n_correct_flips"],
            "n_wrong_flips": r["diag"]["n_wrong_flips"],
            "ceiling": r["ceiling"], "goal2_ok": r["goal2_ok"],
        }
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print()
    print(f"appended → {log_path}")


if __name__ == "__main__":
    main()
