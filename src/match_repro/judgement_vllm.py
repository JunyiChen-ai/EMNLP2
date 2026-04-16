"""MATCH-HVD stage 2c (judgement) — local vLLM substitution.

Upstream: `external_repos/match_hvd/preprocess/judgement.py`
Upstream model: `"Pro/Qwen/Qwen2.5-VL-7B-Instruct"` served behind the
SiliconFlow OpenAI-compatible API (cloud endpoint).

**Substitution rule** (`feedback_api_to_vllm.md`, 2026-04-15):
when an upstream baseline calls a cloud API, substitute a local vLLM
run of the same model variant. Substituted weights are
`Qwen/Qwen2.5-VL-7B-Instruct` (the same open-weights model the API
serves). Everything else — prompt text, 2×2 grid construction,
`max_tokens=4096` — is upstream-exact. Specific upstream line
references for auditors:
  - `format_entries` body (per-entry text block + 2×2 grid build):
    `judgement.py:48-77`
  - Prompt template (`### Hateful Perspective:` etc.):
    `judgement.py:81-96`
  - API call block (now the vLLM substitution):
    `judgement.py:97-113`

**Inputs**: `time_unit_hate.json` / `time_unit_nonhate.json` produced
by stage 2b.5 (`time_unit_jina_clip.py`). These are per-frame entries
`{id, answer, frame, ocr, transcript}` — upstream's exact format. The
`frame` field names the jpg inside `<root>/frames_32/<vid>/`.

**Output**: `results/match_qwen2vl_7b/<dataset>/judge.json` with
schema `[{"id": vid, "summary": "<text>"}, ...]`. The filename
matches upstream stage 3 loader expectations —
`src/model/MATCH/data/HateMM_MATCH.py:19`, `MHClipEN_MATCH.py:20`,
and `MHClipZH_MATCH.py:20` all read `./data/<dataset>/judge.json`.
The schema is identical to upstream `judgement.py:122` output.
"""

import argparse
import base64
import io
import json
import logging
import os
import sys
import time
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "our_method"))
from data_utils import DATASET_ROOTS, SKIP_VIDEOS  # noqa: E402

PROJECT_ROOT = "/data/jehc223/EMNLP2"
ALL_DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_TOKENS = 4096  # upstream `judgement.py:111`


def load_and_group_data(filepath):
    """Upstream `judgement.py:22-28`, verbatim.

    Returns `{video_id -> list[entry]}`. After stage 2b.5, each value
    is the list of per-subclaim CLIP-matched frames (1 entry per
    non-empty `split_answer` slot, up to 4).
    """
    grouped = defaultdict(list)
    if not os.path.exists(filepath):
        return grouped
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        grouped[item["id"]].append(item)
    return grouped


def format_entries_upstream(entries, frames_root, video_id):
    """Verbatim port of upstream `format_entries` (`judgement.py:48-77`).

    The nested-function scope in upstream is flattened here — `entries`,
    `frames_root`, and `video_id` are passed explicitly instead of
    captured from the enclosing `judgement` function. The body is
    byte-for-byte upstream:

    - iterate per-entry, append a numbered Reason/Frame/OCR/Transcript block
    - build 2×2 grid from first 4 frames (upstream: `frame_uris[:4]`)
    - pad with black 128×128 if fewer than 4 entries
    - resize each frame to 128×128 via cv2
    - top_row = hstack(0, 1); bottom_row = hstack(2, 3); grid = vstack
    - BGR→RGB, PIL, PNG, base64
    """
    import cv2
    import numpy as np
    from PIL import Image

    formatted_text = ""
    frame_uris = []
    for i, entry in enumerate(entries):
        frame_path = os.path.join(frames_root, video_id, entry["frame"])
        frame_uris.append(frame_path)
        formatted_text += (
            f"**Reason {i+1}:** {entry['answer']}\n"
            f"  - **Frame:** {entry['frame']}\n"
            f"  - **OCR Text:** {entry['ocr'] if entry['ocr'] else 'N/A'}\n"
            f"  - **Transcript:** {entry['transcript'] if entry['transcript'] else 'N/A'}\n\n"
        )

    target_size = (128, 128)
    black_image = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    while len(frame_uris) < 4:
        frame_uris.append(None)

    images = [cv2.imread(img) if img else black_image for img in frame_uris[:4]]
    # cv2.imread returns None on missing files; mirror upstream's implicit
    # assumption that the path exists by replacing None with a black frame.
    images = [img if img is not None else black_image for img in images]
    images_resized = [cv2.resize(img, target_size) for img in images]
    top_row = np.hstack((images_resized[0], images_resized[1]))
    bottom_row = np.hstack((images_resized[2], images_resized[3]))
    grid_image = np.vstack((top_row, bottom_row))
    grid_pil = Image.fromarray(cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    grid_pil.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return formatted_text.strip(), base64_image


def build_prompt(hate_text, nonhate_text):
    """Verbatim from upstream `judgement.py:81-96`."""
    return (
        "You are an AI judge analyzing the presence of hateful content in videos. Below are two perspectives on the same video.\n"
        "Given the following clue and its associated evidence, determine whether the evidence supports, contradicts, or refines the clue. Provide a brief explanation grounded in the content of the evidence.\n"
        "Each perspective provides reasons, supporting video frames (images), OCR-detected text from the frame, and transcripts from the video's audio.\n"
        f"### Hateful Perspective:\n{hate_text if hate_text else 'No arguments provided.'}\n"
        f"### Non-Hateful Perspective:\n{nonhate_text if nonhate_text else 'No arguments provided.'}\n"
        "### Image Information:"
        "- The images have been processed into a **2\u00d72 grid** for each perspective."
        "- The **top image** corresponds to the hateful perspective."
        "- The **bottom image** corresponds to the non-hateful perspective."
        "### Instructions:\n"
        "- Analyze the hateful perspective, considering the language, tone, and potential for harm or discrimination.\n"
        "- Evaluate both perspective to determine if it effectively counters or mitigates concerns about hate content.\n"
        "- Provide a final judgment that synthesizes both perspectives, clearly explaining whether the content should be considered hateful.\n"
        "- Keep your response concise and within 350 words.\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    args = parser.parse_args()
    if not args.dataset and not args.all:
        parser.error("Provide --dataset or --all")

    datasets = ALL_DATASETS if args.all else [args.dataset]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.info(
        f"MATCH judgement (vLLM) model={args.model} datasets={datasets}"
    )

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 2},
        trust_remote_code=True,
    )
    # Upstream `judgement.py:108-112` does NOT set `temperature` on the
    # OpenAI-compat chat.completions.create call; it takes the endpoint
    # default (1.0). We omit temperature here so vLLM's
    # `SamplingParams` default (also 1.0) matches upstream. Do NOT
    # force `temperature=0.0` — that would be a sampling-behavior
    # deviation flagged by strict review.
    sampling = SamplingParams(max_tokens=MAX_TOKENS)
    logging.info("vLLM Qwen2.5-VL-7B-Instruct loaded")

    for ds in datasets:
        in_dir = os.path.join(
            PROJECT_ROOT, "results", "match_qwen2vl_7b", ds
        )
        # Stage 2b.5 outputs — produced by `time_unit_jina_clip.py`.
        hate_file = os.path.join(in_dir, "time_unit_hate.json")
        nonhate_file = os.path.join(in_dir, "time_unit_nonhate.json")
        out_path = os.path.join(in_dir, "judge.json")

        frames_root = os.path.join(DATASET_ROOTS[ds], "frames_32")

        if not (os.path.exists(hate_file) and os.path.exists(nonhate_file)):
            logging.warning(
                f"[{ds}] skip — stage 2b.5 outputs missing at {in_dir}. "
                "Run time_unit_jina_clip.py first."
            )
            continue
        if not os.path.isdir(frames_root):
            logging.warning(
                f"[{ds}] skip — frames_32 missing at {frames_root}"
            )
            continue

        hate_data = load_and_group_data(hate_file)
        nonhate_data = load_and_group_data(nonhate_file)

        done = set()
        results = []
        if os.path.exists(out_path):
            try:
                with open(out_path) as f:
                    results = json.load(f)
                done = {e["id"] for e in results if e.get("id")}
            except Exception:
                results = []
                done = set()

        all_ids = sorted(set(hate_data.keys()) | set(nonhate_data.keys()))
        logging.info(
            f"[{ds}] {len(all_ids)} video ids, {len(done)} already judged"
        )

        n_processed = 0
        t0 = time.time()
        for video_id in all_ids:
            if video_id in done:
                continue
            if video_id in SKIP_VIDEOS.get(ds, set()):
                continue

            hate_entries = hate_data.get(video_id, [])
            nonhate_entries = nonhate_data.get(video_id, [])

            hate_text, hate_images = format_entries_upstream(
                hate_entries, frames_root, video_id
            )
            nonhate_text, nonhate_images = format_entries_upstream(
                nonhate_entries, frames_root, video_id
            )

            prompt_text = build_prompt(hate_text, nonhate_text)

            # vLLM chat API with OpenAI-style image_url data URIs.
            # Matches upstream `judgement.py:97-106` byte-for-byte,
            # including `"detail": "low"` on both image_url entries
            # (upstream `:102-103`).
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{hate_images}",
                                "detail": "low",
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{nonhate_images}",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ]
            try:
                outputs = llm.chat(messages, sampling_params=sampling)
                summary = outputs[0].outputs[0].text.strip()
            except Exception as e:
                logging.error(f"  {video_id}: judge failed: {e}")
                summary = ""

            results.append({"id": video_id, "summary": summary})
            n_processed += 1

            if n_processed % 20 == 0:
                tmp = out_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, out_path)
                rate = n_processed / (time.time() - t0)
                logging.info(
                    f"  [{ds}] {n_processed} judged / {rate:.2f} vid/s"
                )

        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out_path)
        logging.info(f"[{ds}] done. {n_processed} new judgements")

    logging.info("All datasets done.")


if __name__ == "__main__":
    main()
