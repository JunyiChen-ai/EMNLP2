"""
Task B (v2): 8B-judge rescue with no-prior 2-field plain-text prompt.

For each video in `candidates_band_alpha{α}.jsonl`, ask Qwen3-VL-8B-
Instruct to independently judge the video. The 8B is NEVER told the
2B's first-pass decision. The prompt asks for exactly two fields:

  rationale: <one paragraph 3-6 sentences>
  verdict: <"hateful" or "normal", lowercase, one word>

The definition hint paragraph differs by dataset family (HateMM strict
group-hate vs MHClip broader offensive+hateful) — the only authorized
per-dataset variability per the user's directive.

Plain text decoding, temperature=0, max_tokens=2048, NO logprobs, NO
constrained-token list. Resume support: skip videos already in output.
"""

import argparse
import json
import logging
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "our_method"))

from data_utils import load_annotations, get_media_path  # noqa: E402

# Reuse the media loader pattern from score_holistic_2b.py:253-265.
sys.path.insert(0, os.path.join(_HERE, "..", "our_method"))
from score_holistic_2b import build_media_content  # noqa: E402

PROJECT_ROOT = "/data/jehc223/EMNLP2"
OUT_ROOT = os.path.join(PROJECT_ROOT, "results", "boundary_rescue")
ALL_DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM"]

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# ---- Definition hints (the only per-dataset variability) ----

HATEMM_DEF = (
    "A video is HATEFUL if it directly mocks, demeans, threatens, "
    "stereotypes, or dehumanizes an identifiable group based on race, "
    "ethnicity, religion, gender, sexuality, disability, nationality, "
    "or any similar protected characteristic. A video that merely shows "
    "or discusses hate for the purpose of reporting, critiquing, "
    "parodying, or educating is NOT hateful."
)

MHCLIP_DEF = (
    "A video is HATEFUL in this dataset if it (a) mocks, ridicules, "
    "demeans, or stereotypes an identifiable real person or group, OR "
    "(b) contains slurs, dehumanizing language, or hostile framing of "
    "any identity, gender, race, ethnicity, religion, sexuality, "
    "disability, nationality, or shared characteristic. The mockery "
    "or hostility must be aimed at a specific identifiable target — "
    "real or representative of a real group. "
    "IMPORTANT: the following are NOT hateful in this dataset, even "
    "if they contain shock value or strong language: fictional "
    "dialogue or scripted dramatic conflict between fictional "
    "characters; action-movie or video-game violence or threats; "
    "generic vulgar, crude, or sexual humor that does not target an "
    "identity; LGBTQ+ creators simply being themselves; news, "
    "documentary, parody, or commentary that depicts hate in order "
    "to critique or report on it."
)

# ---- Shared framing and output schema ----

PROMPT_TEMPLATE = """You are reviewing a video for content moderation. Watch the video, read its title, and read any provided transcript.

Title: {title}
Transcript: {transcript}

{definition}

Examine the video carefully. Then output your decision in EXACTLY this two-field format on two lines, with no additional fields, no extra prose, and no markdown:

rationale: <one paragraph (3–6 sentences) describing what you observed in the video and why you reached your verdict>
verdict: <one word, exactly "hateful" or "normal", lowercase>"""


def get_definition(dataset):
    return HATEMM_DEF if dataset == "HateMM" else MHCLIP_DEF


def load_candidates(dataset, alpha, candidates_file=None):
    if candidates_file is not None:
        path = os.path.join(OUT_ROOT, dataset, candidates_file)
    else:
        path = os.path.join(
            OUT_ROOT, dataset, f"candidates_band_alpha{alpha:.2f}.jsonl"
        )
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done_ids(out_path):
    if not os.path.exists(out_path):
        return set()
    done = set()
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("video_id"):
                    done.add(r["video_id"])
            except json.JSONDecodeError:
                pass
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--candidates-file", default=None,
                        help="Override candidates filename (relative to "
                             "results/boundary_rescue/<ds>/)")
    parser.add_argument("--out-tag", default=None,
                        help="Override output filename tag (used when "
                             "--candidates-file is set)")
    parser.add_argument("--version", default="v1")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_DATASETS,
        choices=ALL_DATASETS,
    )
    parser.add_argument("--all", action="store_true",
                        help="Process all 3 datasets")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--transcript-limit", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.alpha is None and args.candidates_file is None:
        parser.error("provide --alpha or --candidates-file")

    if args.all:
        datasets = ALL_DATASETS
    else:
        datasets = args.datasets

    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    if args.candidates_file:
        tag = args.out_tag or args.candidates_file.replace("candidates_", "").replace(".jsonl", "")
        logging.info(
            f"rescue_8b: candidates_file={args.candidates_file} tag={tag} "
            f"version={args.version} model={args.model} datasets={datasets}"
        )
    else:
        tag = f"band_alpha{args.alpha:.2f}"
        logging.info(
            f"rescue_8b: alpha={args.alpha:.2f} version={args.version} "
            f"model={args.model} datasets={datasets} "
            f"max_tokens={args.max_tokens}"
        )

    # Build the work plan: one big list of (dataset, candidate) pairs
    # so we hit a single vLLM init for the whole job.
    work = []
    out_paths = {}
    for ds in datasets:
        cands = load_candidates(ds, args.alpha, args.candidates_file)
        out_path = os.path.join(
            OUT_ROOT, ds,
            f"rescue_8b_{tag}_{args.version}.jsonl",
        )
        out_paths[ds] = out_path
        done_ids = load_done_ids(out_path)
        new_cands = [c for c in cands if c["video_id"] not in done_ids]
        logging.info(
            f"  {ds}: {len(cands)} candidates, {len(done_ids)} already done, "
            f"{len(new_cands)} to process"
        )
        for c in new_cands:
            work.append((ds, c))

    if not work:
        logging.info("All candidates already rescued. Nothing to do.")
        return

    # Lazy vLLM import
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=0.92,
        max_model_len=32768,
        limit_mm_per_prompt={"video": 1, "image": 8},
        allowed_local_media_path="/data/jehc223",
        mm_processor_kwargs={"max_pixels": 100352},
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
    )

    annotations_cache = {}

    def get_ann(ds):
        if ds not in annotations_cache:
            annotations_cache[ds] = load_annotations(ds)
        return annotations_cache[ds]

    t0 = time.time()
    n_processed = 0
    n_skipped_media = 0

    for batch_start in range(0, len(work), args.batch_size):
        batch = work[batch_start : batch_start + args.batch_size]
        batch_messages = []
        batch_meta = []  # (dataset, candidate)

        for ds, cand in batch:
            vid = cand["video_id"]
            ann = get_ann(ds).get(vid)
            if ann is None:
                logging.warning(f"  {ds}/{vid}: not in annotations, skipping")
                n_skipped_media += 1
                continue
            media = get_media_path(vid, ds)
            if media is None:
                logging.warning(f"  {ds}/{vid}: no media, skipping")
                n_skipped_media += 1
                continue
            media_path, media_type = media
            title = ann.get("title", "") or ""
            transcript = (ann.get("transcript", "") or "")[: args.transcript_limit]
            prompt_text = PROMPT_TEMPLATE.format(
                title=title,
                transcript=transcript,
                definition=get_definition(ds),
            )
            media_content = build_media_content(media_path, media_type)
            content = media_content + [{"type": "text", "text": prompt_text}]
            messages = [
                {"role": "system", "content": "You are a content moderation analyst. Answer based strictly on observable evidence."},
                {"role": "user", "content": content},
            ]
            batch_messages.append(messages)
            batch_meta.append((ds, cand))

        if not batch_messages:
            continue

        try:
            outputs = llm.chat(messages=batch_messages, sampling_params=sampling_params)
        except Exception as e:
            err_msg = str(e)
            logging.error(f"  Batch failed: {err_msg[:200]}, falling back to single")
            outputs = []
            for i, msgs in enumerate(batch_messages):
                try:
                    out_one = llm.chat(messages=[msgs], sampling_params=sampling_params)
                    outputs.append(out_one[0])
                except Exception as e2:
                    logging.error(
                        f"  {batch_meta[i][0]}/{batch_meta[i][1]['video_id']}: single failed: {str(e2)[:200]}"
                    )
                    outputs.append(None)

        # Write out one record per processed item, append-mode per dataset
        for (ds, cand), out in zip(batch_meta, outputs):
            if out is None:
                response_text = ""
                err = "single_failed"
            else:
                gen = out.outputs[0]
                response_text = gen.text or ""
                err = None
            rec = {
                "video_id": cand["video_id"],
                "dataset": ds,
                "score_2b": cand["score"],
                "posterior_hi": cand["posterior_hi"],
                "threshold": cand["threshold"],
                "pred_baseline": cand["pred_baseline"],
                "side": cand["side"],
                "alpha": args.alpha,
                "rescue_response": response_text,
                "error": err,
            }
            with open(out_paths[ds], "a") as fout:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                os.fsync(fout.fileno())
            n_processed += 1

        elapsed = time.time() - t0
        rate = n_processed / elapsed if elapsed > 0 else 0
        logging.info(
            f"  [{n_processed}/{len(work)}] rate={rate:.2f} cand/s"
        )

    logging.info(
        f"\nDone. processed={n_processed}, skipped_media={n_skipped_media}"
    )
    for ds in datasets:
        logging.info(f"  {ds} → {out_paths[ds]}")


if __name__ == "__main__":
    main()
