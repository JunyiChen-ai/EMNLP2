"""LoReHM video reproduction — main driver (rework 2026-04-15).

Upstream: `external_repos/lorehm/main.py:63-99`. Control flow is
**upstream-verbatim**; the only substitutions are the localized
rework deviations:

  1. `image_file` is a **list of 16 PIL.Image frames** loaded from
     `frames_16/<vid>/frame_NNN.jpg` for NNN in 000..015. LLaVA-Next
     sees 16 `<image>` tokens in a single forward pass via its
     native multi-image path. No grid composite.
  2. `text` field is the video transcript, truncated to 500 chars.
  3. Prompts are meme→video substring rewrites of upstream's
     `BASIC_PROMPT` / `RSA_PROMPT` (see `prompts.py`), preserving
     every non-meme character byte-for-byte from upstream's
     `utils/constants.py:1-5`.
  4. `rel_sampl` is computed at our end via `retrieval.py` using
     Jina-CLIP-v2 pooled features; upstream's precomputed jsonls
     don't exist for video datasets.
  5. Model loaded at bf16, no quantization, sharded across 2 GPUs
     via `device_map="auto"` (fidelity upgrade).

Everything else — `parse_answer`, `get_rsa_label`, control flow,
MIA `"\nNote:\n{insights}\n"` append rule, LLaVA generation kwargs
— is upstream byte-for-byte.

Total LLaVA calls per test video:
  * Baseline: 1
  * With MIA: 1 (MIA prepends a `Note:` block to the query)
  * With RSA: 1 or 2 (second call fires only when
    `rsa_label != basic_predict`, exactly matching upstream
    `main.py:81-97`).

Output schema (per line, to `results/lorehm/<dataset>/test_lorehm.jsonl`):
    {
      "video_id": ...,
      "pred": 0 | 1,
      "label": 0 | 1,
      "basic_predict": 0 | 1,
      "rsa_label": 0 | 1 | null,
      "used_rsa_reask": true | false,
      "mia": true | false,
      "thought": "<model rationale>",
      "raw_response": "<full LLaVA output>"
    }
Compatible with `eval_generative_predictions.eval_one`.
"""

import argparse
import json
import logging
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from llava_runner import DEFAULT_MODEL_ID, LlavaRunner  # noqa: E402
from lorehm_video_dataset import ALL_DATASETS, build_video_items  # noqa: E402
from prompts import (  # noqa: E402
    BASIC_PROMPT,
    LABELS_STR,
    MEME_INSIGHTS,
    RSA_PROMPT,
)
from retrieval import (  # noqa: E402
    build_rel_sampl_for_dataset,
    get_rsa_label,
    load_rel_sampl,
    rel_sampl_output_path,
)

PROJECT_ROOT = "/data/jehc223/EMNLP2"

# Transcript truncation to 500 chars (unchanged from prior brief).
TRANSCRIPT_LIMIT = 500


def parse_answer(judge_result, sep="Answer:"):
    """Upstream `utils/utils.py:63-74`, verbatim (incl. the default-1
    fallback on parse failure).
    """
    answer = (
        judge_result.split(sep)[-1]
        .split(".")[0]
        .split(",")[0]
        .lower()
        .strip()
    )
    if "harmless" in answer:
        predict = 0
    elif "harmful" in answer:
        predict = 1
    else:
        answer = "skip"
        predict = 1
    return answer, predict


def _extract_thought(text: str) -> str:
    if "Thought:" in text:
        try:
            return (
                text.split("Thought:", 1)[1]
                .split("Answer:", 1)[0]
                .strip()
            )
        except Exception:
            return ""
    return ""


class _Args:
    """Upstream-compatible `args` object shape.

    `run_proxy(args)` expects attribute access on:
        args.query, args.image_file, args.temperature, args.num_beams,
        args.max_new_tokens, args.sep, args.top_p, args.conv_mode
    """

    def __init__(
        self,
        query: str,
        image_file,
        max_new_tokens: int = 1024,
    ):
        self.query = query
        self.image_file = image_file
        self.sep = ","
        self.temperature = 0
        self.top_p = None
        self.num_beams = 1
        self.max_new_tokens = max_new_tokens
        self.conv_mode = None


def score_one_video(
    item,
    runner: LlavaRunner,
    rsa_enabled: bool,
    rsa_k: int,
    mia_enabled: bool,
    mia_block: str,
    max_new_tokens: int,
):
    """Per-video LoReHM control flow, mirroring upstream
    `main.py:63-99` branch-by-branch.

    Step 1 — pass the 16 frames directly as a list (LLaVA-Next
    multi-image path), truncate transcript to 500 chars, build
    `BASIC_PROMPT.format(text)` with optional
    `"\\nNote:\\n{insights}\\n"` MIA appended, call LLaVA, parse
    `harmful | harmless` via `parse_answer`.

    Step 2 — if RSA enabled and `get_rsa_label(rel_sampl, k) !=
    basic_predict`, re-ask with `RSA_PROMPT.format(text,
    labels_str[rsa_label])` (same MIA appended), parse final answer.
    """
    image_list = item["frames"]  # list of 16 PIL.Image (RGB)
    raw_transcript = item["text"] or ""
    text = raw_transcript[:TRANSCRIPT_LIMIT]

    query = BASIC_PROMPT.format(text)
    if mia_enabled and mia_block:
        query = query + f"\nNote:\n{mia_block}\n"

    args = _Args(query=query, image_file=image_list, max_new_tokens=max_new_tokens)
    basic_response = runner.run_model(args)
    _ans, basic_predict = parse_answer(basic_response, "Answer:")

    used_rsa_reask = False
    rsa_label = None
    final_predict = basic_predict
    final_response = basic_response

    if rsa_enabled and item.get("rel_sampl") is not None:
        rsa_label = get_rsa_label(item["rel_sampl"], rsa_k)
        if basic_predict != rsa_label:
            reask_query = RSA_PROMPT.format(text, LABELS_STR[rsa_label])
            if mia_enabled and mia_block:
                reask_query = reask_query + f"\nNote:\n{mia_block}\n"
            args.query = reask_query
            reask_response = runner.run_model(args)
            _ans, final_predict = parse_answer(reask_response, "Answer:")
            final_response = reask_response
            used_rsa_reask = True

    return {
        "video_id": item["vid"],
        "pred": int(final_predict),
        "label": int(item["label"]),
        "basic_predict": int(basic_predict),
        "rsa_label": int(rsa_label) if rsa_label is not None else None,
        "used_rsa_reask": bool(used_rsa_reask),
        "mia": bool(mia_enabled),
        "thought": _extract_thought(final_response),
        "raw_response": final_response,
    }


def run_one_dataset(dataset: str, args, runner: LlavaRunner):
    """Per-dataset test-split loop, upstream `main.py:63-110`
    verbatim control flow.
    """
    # Load or build rel_sampl (v2 brief: build once, reuse from disk).
    rel_sampl_map = None
    if args.rsa:
        rel_sampl_map = load_rel_sampl(dataset)
        if not rel_sampl_map:
            logging.info(
                f"[{dataset}] no cached rel_sampl.json; building now via Jina-CLIP-v2"
            )
            rel_sampl_map, _, _, _ = build_rel_sampl_for_dataset(
                dataset, pool_topk=args.pool_topk, persist=True
            )
        else:
            logging.info(
                f"[{dataset}] loaded cached rel_sampl for {len(rel_sampl_map)} test videos"
            )

    items, missing = build_video_items(
        dataset, "test", rel_sampl_map=rel_sampl_map
    )
    logging.info(
        f"[{dataset}] {len(items)} test items  missing={missing}"
    )

    mia_block = None
    if args.mia:
        mia_block = MEME_INSIGHTS.get("llava-v1.6-34b", {}).get(dataset)
        if mia_block is None:
            logging.warning(
                f"[{dataset}] no MEME_INSIGHTS block; MIA disabled for this dataset"
            )

    out_dir = os.path.join(PROJECT_ROOT, "results", "lorehm", dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test_lorehm.jsonl")

    # Resume.
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("video_id"):
                        done.add(r["video_id"])
                except Exception:
                    pass
    logging.info(f"[{dataset}] {len(done)} already scored")

    t0 = time.time()
    n_processed = 0
    with open(out_path, "a") as f:
        for item in items:
            if item["vid"] in done:
                continue
            try:
                rec = score_one_video(
                    item,
                    runner,
                    rsa_enabled=args.rsa,
                    rsa_k=args.rsa_k,
                    mia_enabled=args.mia,
                    mia_block=mia_block,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as e:
                logging.error(f"  {item['vid']}: scoring failed: {e}")
                rec = {
                    "video_id": item["vid"],
                    "pred": -1,
                    "error": str(e)[:400],
                }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            n_processed += 1
            if n_processed % 10 == 0:
                rate = n_processed / (time.time() - t0 + 1e-9)
                logging.info(
                    f"  [{dataset}] {n_processed}/{len(items)}  "
                    f"{rate:.2f} vid/s"
                )
    logging.info(
        f"[{dataset}] done, {n_processed} new items, output {out_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="LoReHM video reproduction (LLaVA-v1.6-34b + RSA + MIA) — v2"
    )
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--rsa", action="store_true", default=True,
                        help="Enable RSA re-ask (upstream default)")
    parser.add_argument("--no-rsa", dest="rsa", action="store_false")
    parser.add_argument("--mia", action="store_true", default=True,
                        help="Enable MIA note block (upstream default)")
    parser.add_argument("--no-mia", dest="mia", action="store_false")
    parser.add_argument("--rsa-k", type=int, default=5,
                        help="Upstream default k=5 (main.py:20)")
    parser.add_argument("--pool-topk", type=int, default=50)
    parser.add_argument("--max-memory-per-gpu", default="75GiB")
    parser.add_argument("--cpu-memory", default="64GiB")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("Provide --dataset or --all")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    datasets = ALL_DATASETS if args.all else [args.dataset]
    logging.info(
        f"LoReHM repro rework: datasets={datasets}  model={args.model}  "
        f"rsa={args.rsa} k={args.rsa_k} pool={args.pool_topk}  "
        f"mia={args.mia}  bf16 device_map=auto  "
        f"max_memory_per_gpu={args.max_memory_per_gpu}  "
        f"transcript_limit={TRANSCRIPT_LIMIT}"
    )

    runner = LlavaRunner(
        model_id=args.model,
        max_memory_per_gpu=args.max_memory_per_gpu,
        cpu_memory=args.cpu_memory,
    )
    logging.info("LLaVA-v1.6-34b loaded (bf16, device_map=auto)")

    for ds in datasets:
        run_one_dataset(ds, args, runner)

    logging.info("All datasets done.")


if __name__ == "__main__":
    main()
