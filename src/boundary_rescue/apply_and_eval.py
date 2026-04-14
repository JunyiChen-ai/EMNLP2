"""
Task C: apply rescue flips to baseline predictions and evaluate.

Flip decision is derived from the rescue's STRUCTURED FIELDS via a
per-dataset rule, not from the raw VERDICT line (the 2B model often
truncates before VERDICT or outputs a verbal verdict we would miss).

HateMM rule (strict group-hate definition):
  below: OVERTURN iff TARGET_GROUP not-none AND (slur|dehu|visual) not-none
  above: CONFIRM iff VIDEO_ROLE=PRODUCING AND TARGET_IS_GROUP=YES; else OVERTURN

MHClip rule (broader offensive + hateful definition):
  below: OVERTURN iff TARGET, OFFENSIVE_ELEMENT, and CATEGORY are all
         concretely filled in (TARGET not "none", OFFENSIVE_ELEMENT not
         "none", CATEGORY in canonical list)
  above: OVERTURN iff ANY_MOCKERY_OR_INSULT=NO AND ANY_HOSTILITY=NO AND
         VIDEO_ROLE in {FACTUAL, POSITIVE}; else CONFIRM

Selection is top-K nearest to threshold per side; non-candidate videos
keep pred_baseline.

No labels at any step of the decision pipeline. Labels are used only
at the eval step to compute acc/mF1 and the diagnostic flip breakdown.
"""

import argparse
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "our_method"))
sys.path.insert(0, os.path.join(_HERE, "..", "naive_baseline"))

from data_utils import load_annotations, SKIP_VIDEOS  # noqa: E402
from eval_generative_predictions import eval_one, collapse_label  # noqa: E402

PROJECT_ROOT = "/data/jehc223/EMNLP2"
OUT_ROOT = os.path.join(PROJECT_ROOT, "results", "boundary_rescue")
ALL_DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM"]

# Hardcoded strict-beat targets. ZH target is loaded from the pinned file.
EN_TARGET = {"acc": 0.7640, "mf1": 0.6532}
HM_TARGET = {"acc": 0.8047, "mf1": 0.7930}


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


_FIELD_RE = {}


def _field(text, name):
    if name not in _FIELD_RE:
        _FIELD_RE[name] = re.compile(rf"{name}\s*:\s*([^\n]+)")
    m = _FIELD_RE[name].search(text or "")
    return m.group(1).strip() if m else ""


def _is_none(s):
    s = (s or "").strip().strip('"\'').lower()
    return s in {"", "none", "n/a", "no", "nothing", "null"}


def decide_flip(dataset, side, response_text):
    """Return True iff this candidate should be flipped."""
    if dataset == "HateMM":
        if side == "below":
            tg = _field(response_text, "TARGET_GROUP")
            sl = _field(response_text, "EXPLICIT_SLURS")
            de = _field(response_text, "DEHUMANIZING_LANGUAGE")
            vi = _field(response_text, "VISUAL_HATE_SIGNAL")
            return (not _is_none(tg)) and (
                not _is_none(sl) or not _is_none(de) or not _is_none(vi)
            )
        else:
            role = _field(response_text, "VIDEO_ROLE").upper()
            tg = _field(response_text, "TARGET_IS_GROUP").upper()
            return not ("PRODUCING" in role and tg.startswith("YES"))
    else:  # MHClip
        if side == "below":
            tgt = _field(response_text, "TARGET")
            elem = _field(response_text, "OFFENSIVE_ELEMENT")
            cat = _field(response_text, "CATEGORY").upper()
            cat_ok = any(
                c in cat
                for c in [
                    "MOCKERY",
                    "SLUR",
                    "STEREOTYPE",
                    "BULLYING",
                    "CRUDE_HUMOR",
                    "HOSTILITY",
                    "HATE_SYMBOL",
                ]
            )
            return (not _is_none(tgt)) and (not _is_none(elem)) and cat_ok
        else:
            mock = _field(response_text, "ANY_MOCKERY_OR_INSULT").upper()
            host = _field(response_text, "ANY_HOSTILITY").upper()
            role = _field(response_text, "VIDEO_ROLE").upper()
            return (
                mock.startswith("NO")
                and host.startswith("NO")
                and ("FACTUAL" in role or "POSITIVE" in role)
            )


def apply_flips(dataset, version, candidates_version):
    base_path = os.path.join(OUT_ROOT, dataset, "baseline_preds.jsonl")
    cand_path = os.path.join(
        OUT_ROOT, dataset, f"candidates_{candidates_version}.jsonl"
    )
    rescue_path = os.path.join(OUT_ROOT, dataset, f"rescue_{version}.jsonl")
    out_path = os.path.join(OUT_ROOT, dataset, f"test_{version}.jsonl")

    base = load_jsonl(base_path)
    candidates = load_jsonl(cand_path) if os.path.isfile(cand_path) else []
    candidate_ids = {c["video_id"] for c in candidates}
    rescue = load_jsonl(rescue_path) if os.path.isfile(rescue_path) else []
    rescue_by_vid = {r["video_id"]: r for r in rescue}

    flip_records = {}
    for c in candidates:
        vid = c["video_id"]
        rr = rescue_by_vid.get(vid)
        if rr is None:
            continue
        side = rr["side"]
        flipped = decide_flip(dataset, side, rr["response_text"])
        flip_records[vid] = {
            "side": side,
            "flipped": flipped,
            "pred_baseline": int(rr["pred_baseline"]),
            "pred_after": (1 - int(rr["pred_baseline"])) if flipped else int(rr["pred_baseline"]),
            "response_text": rr["response_text"],
        }

    with open(out_path, "w") as f:
        for r in base:
            vid = r["video_id"]
            if vid in flip_records:
                pred_after = flip_records[vid]["pred_after"]
            else:
                pred_after = int(r["pred_baseline"])
            f.write(
                json.dumps(
                    {"video_id": vid, "pred": pred_after}, ensure_ascii=False
                )
                + "\n"
            )

    return out_path, flip_records


def diagnostic_flip_counts(dataset, flip_records):
    """DIAGNOSTIC ONLY — uses labels to break down correct/wrong flips."""
    ann = load_annotations(dataset)
    skip = SKIP_VIDEOS.get(dataset, set())
    correct = wrong = total_flipped = 0
    flipped_ids = []
    for vid, r in flip_records.items():
        if vid in skip:
            continue
        if vid not in ann:
            continue
        if not r.get("flipped"):
            continue
        total_flipped += 1
        gt = collapse_label(dataset, ann[vid]["label"])
        if int(r["pred_after"]) == gt:
            correct += 1
        else:
            wrong += 1
        flipped_ids.append(vid)
    return {
        "n_flipped_total": total_flipped,
        "n_flipped_correct": correct,
        "n_flipped_wrong": wrong,
        "net_gain": correct - wrong,
        "flipped_ids": flipped_ids,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--candidates-version",
        default=None,
        help="Candidates version (default: same as --version)",
    )
    args = parser.parse_args()
    if args.candidates_version is None:
        args.candidates_version = args.version

    # Load ZH pre-repro target (pinned in Task 0)
    zh_target_path = os.path.join(OUT_ROOT, "zh_prerepro_baseline.json")
    with open(zh_target_path) as f:
        zh_target = json.load(f)

    targets = {
        "MHClip_EN": EN_TARGET,
        "MHClip_ZH": {"acc": zh_target["acc"], "mf1": zh_target["mf1"]},
        "HateMM": HM_TARGET,
    }

    print(f"=== boundary_rescue {args.version} ===")
    print()
    print(
        f"{'dataset':<10}  "
        f"{'base acc':>9} {'base mf1':>9}  "
        f"{'new acc':>9} {'new mf1':>9}  "
        f"{'Δacc':>8} {'Δmf1':>8}  "
        f"{'beat?':>6}"
    )
    print("-" * 82)
    all_beat = True
    results = {}
    for ds in ALL_DATASETS:
        test_path, flip_records = apply_flips(
            ds, args.version, args.candidates_version
        )
        new_metrics = eval_one(test_path, ds)
        t = targets[ds]
        dacc = new_metrics["acc"] - t["acc"]
        dmf1 = new_metrics["mf"] - t["mf1"]
        beat = dacc > 1e-9 and dmf1 > 1e-9
        if not beat:
            all_beat = False
        print(
            f"{ds:<10}  "
            f"{t['acc']:>9.4f} {t['mf1']:>9.4f}  "
            f"{new_metrics['acc']:>9.4f} {new_metrics['mf']:>9.4f}  "
            f"{dacc:>+8.4f} {dmf1:>+8.4f}  "
            f"{'YES' if beat else 'no':>6}"
        )
        results[ds] = {
            "base_acc": t["acc"],
            "base_mf1": t["mf1"],
            "new_acc": new_metrics["acc"],
            "new_mf1": new_metrics["mf"],
            "dacc": dacc,
            "dmf1": dmf1,
            "beat": beat,
            "flip_records": flip_records,
        }

    print()
    print(f"strict_beat_all = {all_beat}")
    print()

    # --- DIAGNOSTIC ONLY (uses labels) ---
    print("=== diagnostic flip breakdown (uses labels — NOT part of method) ===")
    print()
    print(
        f"{'dataset':<10}  {'#cand':>5}  {'#flip':>5}  {'correct':>8} {'wrong':>6} {'net':>5}"
    )
    print("-" * 60)
    for ds in ALL_DATASETS:
        flip_records = results[ds]["flip_records"]
        diag = diagnostic_flip_counts(ds, flip_records)
        n_cand = len(flip_records)
        print(
            f"{ds:<10}  {n_cand:>5d}  {diag['n_flipped_total']:>5d}  "
            f"{diag['n_flipped_correct']:>8d} {diag['n_flipped_wrong']:>6d} "
            f"{diag['net_gain']:>+5d}"
        )

    # Append to loop log
    log_path = os.path.join(OUT_ROOT, "loop_log.jsonl")
    log_entry = {
        "version": args.version,
        "strict_beat_all": all_beat,
    }
    for ds in ALL_DATASETS:
        r = results[ds]
        log_entry[ds] = {
            "new_acc": r["new_acc"],
            "new_mf1": r["new_mf1"],
            "dacc": r["dacc"],
            "dmf1": r["dmf1"],
            "beat": r["beat"],
        }
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print()
    print(f"appended to {log_path}")


if __name__ == "__main__":
    main()
