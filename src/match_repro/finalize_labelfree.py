"""MATCH-HVD — intermediate label-free preview of the judgement output.

**Status note (2026-04-15 scope correction)**: MATCH-HVD is a
supervised method. Its authoritative baseline number comes from
**stage 3** — a classifier head trained on MFCC audio features +
ViViT video features + BERT text features (the judgement summary).
Stage 3 lives in a separate brief; this file is NOT the primary
MATCH deliverable.

What this file is: a **debug / preview tool** that parses the stage
2c judge summary text for a yes/no verdict, so we can spot-check
whether the vLLM-substituted judge is producing sensible rationales
without running the full stage 3 training loop. It is label-free by
construction (no label data tunes the phrase list), which is why it
sits here — useful for quick sanity checks, not for the paper
headline number.

Input : `results/match_qwen2vl_7b/<dataset>/judge.json` with
        schema `[{"id": vid, "summary": "<judge text>"}]`
Output: `results/match_qwen2vl_7b/<dataset>/test_match_peek.jsonl`
        with schema `{"video_id": vid, "pred": 0|1, "summary": "<...>"}`

The file suffix was renamed from `test_match.jsonl` to
`test_match_peek.jsonl` to avoid confusion with the stage 3 output
that eval_generative_predictions.py will eventually consume. The
parser rules themselves are unchanged from the earlier deliverable
and are preserved here for reproducibility of the preview.

Parser rules (no label tuning):
  1. Scan the lowercased summary for explicit verdict phrases in
     priority order. Negative phrases ("not hateful", "non-hateful",
     "should not be classified as hateful", ...) are checked before
     positive phrases to avoid substring false-matches. First hit wins.
  2. Word-count fallback: count standalone `\\bhateful\\b` minus
     occurrences of `non-hateful` / `non hateful` / `not hateful`.
     Net > 0 → pred=1; net ≤ 0 with any non-hate marker → pred=0.
  3. Unparseable / empty → pred=0 (conservative default).
"""

import argparse
import json
import logging
import os
import re

ALL_DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]
PROJECT_ROOT = "/data/jehc223/EMNLP2"

# Positive phrases (→ pred=1). Ordered longest-first so that
# "should not be classified as hateful" does not accidentally match
# "classified as hateful".
POSITIVE_PHRASES = [
    "should be considered hateful",
    "should be classified as hateful",
    "classify as hateful",
    "classified as hateful",
    "is hateful",
    "contains hate content",
    "contains hateful content",
    "content is hateful",
    "promotes hate",
]

# Negative phrases (→ pred=0).
NEGATIVE_PHRASES = [
    "should not be considered hateful",
    "should not be classified as hateful",
    "not hateful",
    "non-hateful",
    "non hateful",
    "does not contain hate",
    "does not promote hate",
    "is not hateful",
    "is non-hateful",
]


def parse_verdict(summary):
    if not summary:
        return 0
    text = summary.lower()

    # Priority 1: explicit verdict phrases. Negative phrases are
    # checked first because several positives are substrings of
    # negatives (e.g. "hateful" ⊂ "not hateful").
    for phrase in NEGATIVE_PHRASES:
        if phrase in text:
            return 0
    for phrase in POSITIVE_PHRASES:
        if phrase in text:
            return 1

    # Priority 2: word-count fallback. Count the token "hateful"
    # appearing as a standalone word vs the explicit non-hate markers.
    # `\bhateful\b` avoids matching inside "non-hateful" — we already
    # would have returned above if that substring was present, but
    # belt-and-braces is cheap.
    non_hateful = sum(
        len(re.findall(pattern, text))
        for pattern in (r"non-hateful", r"non hateful", r"not hateful")
    )
    hateful = len(re.findall(r"\bhateful\b", text)) - non_hateful
    if hateful > non_hateful:
        return 1
    if non_hateful > 0:
        return 0
    # Priority 3: unparseable → conservative default.
    return 0


def finalize_one_dataset(ds):
    in_path = os.path.join(
        PROJECT_ROOT, "results", "match_qwen2vl_7b", ds, "judge.json"
    )
    out_path = os.path.join(
        PROJECT_ROOT, "results", "match_qwen2vl_7b", ds, "test_match_peek.jsonl"
    )
    if not os.path.exists(in_path):
        logging.warning(f"[{ds}] missing {in_path}, skipping")
        return 0, 0

    with open(in_path) as f:
        judgements = json.load(f)

    n_hateful = 0
    n_nonhate = 0
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for entry in judgements:
            vid = entry.get("id")
            summary = entry.get("summary", "") or ""
            pred = parse_verdict(summary)
            if pred == 1:
                n_hateful += 1
            else:
                n_nonhate += 1
            rec = {"video_id": vid, "pred": pred, "summary": summary}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)
    logging.info(
        f"[{ds}] wrote {out_path}  ({n_hateful} hateful, {n_nonhate} non)"
    )
    return n_hateful, n_nonhate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.dataset and not args.all:
        parser.error("Provide --dataset or --all")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )

    datasets = ALL_DATASETS if args.all else [args.dataset]
    for ds in datasets:
        finalize_one_dataset(ds)


if __name__ == "__main__":
    main()
