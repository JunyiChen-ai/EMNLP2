"""MATCH-HVD stage 2b.5 — Jina-CLIP frame alignment.

Port of upstream `external_repos/match_hvd/preprocess/time_unit.py`
with the three documented deviations listed in
`docs/baseline_briefs/match_hvd_2b5_jina_clip.md`:

1. `./MLLM/jina-clip-v2` (modelscope local cache) →
   `jinaai/jina-clip-v2` (HF Hub). Identical weights, different
   loader path. Loaded via
   `AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)`.

2. Frames directory: `<dataset_root>/frames_32/<vid>/` (produced by
   our `extract_frames_32.py`). Upstream reads `./data/HateMM/frames_32/`
   (modelscope-cached path); same layout.

3. Per-frame OCR signal dropped. Upstream `find_best_matching_frame`
   computes
     `frame_text = ocr_data.get(str(idx), "") + " " + transcript_splits[idx]`
   using a pre-extracted per-frame OCR JSON. We do not have that OCR
   file, so we substitute `frame_text = transcript_splits[idx]`. The
   `{"ocr": ...}` field of the emitted per-frame entry is left as the
   empty string. Upstream `judgement.py:58` already falls back to
   `'N/A'` for empty OCR, so downstream rendering is preserved.

Everything else is copied verbatim from upstream:
- `split_answer` (upstream lines 9-29) — regex split on CJK/ASCII
  sentence terminators, pad to 4 slots, even distribution when >4
  sentences.
- `load_hate_data` (upstream lines 32-37) — reads JSON produced by
  stage 2a/2b (`hate.json` / `nonhate.json`) and applies
  `split_answer` to each video's `answer`.
- `load_transcripts_from_annotations` — **thin wrapper** around our
  `data_utils.load_annotations(dataset)`, producing the same
  `{vid -> list[32]}` structure upstream builds from a jsonl via
  `load_transcripts`. The splitter logic inside is byte-for-byte
  upstream (lines 39-69): sentence regex `[。！？?.]` with pairing,
  32-segment interpolation by word-count breakpoints.
- `find_best_matching_frame` (upstream lines 72-101) — image / text
  embeddings via `model.encode_image` / `encode_text` at
  `truncate_dim=512`, similarity = image@answer + text@answer,
  argmax over frame index, returns `{id, answer, frame, ocr,
  transcript}`.
- `clip_run` (upstream lines 103-129) — resume via
  `processed_video_ids` set, write output JSON after each video.

Pre-flight note (for director; you do NOT run this yourself):
First run on a compute node will hit HF Hub for the model weights
(~1.2 GB). Pre-warm the cache on the login node before sbatch:
```
HUGGING_FACE_HUB_TOKEN=... python -c "from huggingface_hub import snapshot_download; snapshot_download('jinaai/jina-clip-v2', allow_patterns=['*.json','*.txt','*.bin','*.safetensors','tokenizer*','*.py'])"
```
"""

import argparse
import json
import logging
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "our_method"))
from data_utils import DATASET_ROOTS, SKIP_VIDEOS, load_annotations  # noqa: E402

PROJECT_ROOT = "/data/jehc223/EMNLP2"
ALL_DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]
DEFAULT_MODEL = "jinaai/jina-clip-v2"
TRUNCATE_DIM = 512  # upstream `time_unit.py:84`
NUM_SEGMENTS = 32  # upstream assumes 32 frames per video


def split_answer(answer):
    """Upstream `time_unit.py:9-29`, verbatim."""
    sentences = re.split(r"[。！？?.]", answer)
    sentences = [s.strip() for s in sentences if s.strip()]

    n = len(sentences)
    splits = [""] * 4

    if n == 0:
        return splits
    elif n <= 4:
        for i in range(n):
            splits[i] = sentences[i]
    else:
        avg_size = n // 4
        remainder = n % 4
        indices = [0]
        for i in range(4):
            size = avg_size + (1 if i < remainder else 0)
            indices.append(indices[-1] + size)
        for i in range(4):
            splits[i] = " ".join(sentences[indices[i]:indices[i + 1]])
    return splits


def load_hate_data(hate_json):
    """Upstream `time_unit.py:32-37`, verbatim."""
    with open(hate_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    hate_dict = {entry["id"]: split_answer(entry["answer"]) for entry in data}
    return hate_dict


def build_transcript_dict(dataset):
    """Our adaptation of upstream `load_transcripts` (`time_unit.py:39-71`).

    Upstream reads a jsonl of `{vid, transcript}`; we read from the
    canonical project-wide `data_utils.load_annotations(dataset)`.
    The per-video splitter body below is byte-for-byte upstream.
    """
    ann = load_annotations(dataset)
    transcript_dict = {}
    for vid, entry in ann.items():
        transcript = (entry.get("transcript", "") or "").strip()
        if not transcript:
            transcript_dict[vid] = [""] * NUM_SEGMENTS
            continue
        # --- begin upstream-verbatim splitter (`time_unit.py:49-69`) ---
        sentences = re.split(r"([。！？?.])", transcript)
        if len(sentences) > 1:
            sentences = [
                "".join(s).strip()
                for s in zip(sentences[::2], sentences[1::2])
            ]
        else:
            sentences = [transcript]
        sentences = [s for s in sentences if s]
        words = transcript.split()
        num_words = len(words)
        if num_words < NUM_SEGMENTS:
            transcript_splits = (sentences * NUM_SEGMENTS)[:NUM_SEGMENTS]
            transcript_dict[vid] = transcript_splits
            continue
        segment_size = num_words // NUM_SEGMENTS
        breakpoints = [i * segment_size for i in range(NUM_SEGMENTS)] + [
            num_words - 1
        ]
        transcript_splits = []
        word_index = 0
        for bp in breakpoints:
            while word_index < len(sentences) and bp >= len(
                " ".join(words[:word_index]).split()
            ):
                word_index += 1
            transcript_splits.append(sentences[max(0, word_index - 1)])
        transcript_dict[vid] = transcript_splits
        # --- end upstream-verbatim splitter ---
    return transcript_dict


def find_best_matching_frame(
    video_id, answer, frames_dir, transcript_dict, model, device
):
    """Upstream `time_unit.py:72-101`, verbatim except for:

    - `frames_dir` is our `<root>/frames_32/<vid>/` instead of
      upstream's `./data/HateMM/frames_32/<vid>/` (same layout).
    - `frame_text` drops the OCR contribution (documented deviation
      #3): upstream used
        `ocr_data.get(str(idx), "") + " " + transcript_splits[idx]`
      because we don't have per-frame OCR, we use
        `transcript_splits[idx]`.
    - The returned entry's `"ocr"` field is `""`.

    The embedding + similarity logic is otherwise byte-for-byte
    upstream.
    """
    from PIL import Image

    frames_path = os.path.join(frames_dir, video_id)
    if not os.path.exists(frames_path):
        return None

    frame_files = sorted(os.listdir(frames_path))
    transcript_splits = transcript_dict.get(video_id, [""] * NUM_SEGMENTS)

    best_frame, best_score = None, -float("inf")
    import torch
    for idx, frame_name in enumerate(frame_files):
        frame_path = os.path.join(frames_path, frame_name)
        image = Image.open(frame_path)
        # DEVIATION #3: upstream concatenated per-frame OCR here.
        frame_text = (
            transcript_splits[idx]
            if idx < len(transcript_splits)
            else ""
        )

        with torch.no_grad():
            image_emb = model.encode_image(image, truncate_dim=TRUNCATE_DIM)
            text_emb = model.encode_text(
                [frame_text], truncate_dim=TRUNCATE_DIM
            )
            answer_emb = model.encode_text(
                [answer], truncate_dim=TRUNCATE_DIM
            )

            similarity_image = (image_emb @ answer_emb.T).sum().item()
            similarity_text = (text_emb @ answer_emb.T).sum().item()
            similarity = similarity_image + similarity_text

        if similarity > best_score:
            best_score = similarity
            best_frame = {
                "id": video_id,
                "answer": answer,
                "frame": frame_name,
                "ocr": "",  # DEVIATION #3: no per-frame OCR available
                "transcript": (
                    transcript_splits[idx]
                    if idx < len(transcript_splits)
                    else ""
                ),
            }
    return best_frame


def clip_run(hate_json, frames_dir, transcript_dict, output_json, model, device):
    """Upstream `time_unit.py:103-129`, verbatim except:
    - No `ocr_dir` parameter (deviation #3)
    - `hate_dict` still comes from upstream `load_hate_data`
    """
    hate_dict = load_hate_data(hate_json)

    processed_video_ids = set()
    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            try:
                existing_results = json.load(f)
                processed_video_ids = {
                    entry["id"] for entry in existing_results
                }
            except json.JSONDecodeError:
                existing_results = []
    else:
        existing_results = []

    results = existing_results
    from tqdm import tqdm
    for video_id, answers in tqdm(hate_dict.items(), desc="Processing Videos"):
        if video_id in processed_video_ids:
            continue
        for answer in answers:
            if answer.strip():
                best_frame = find_best_matching_frame(
                    video_id, answer, frames_dir, transcript_dict, model, device
                )
                if best_frame:
                    results.append(best_frame)
        tmp = output_json + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, output_json)


def run_one_dataset(ds, model, device):
    root = DATASET_ROOTS[ds]
    in_dir = os.path.join(PROJECT_ROOT, "results", "match_qwen2vl_7b", ds)
    frames_dir = os.path.join(root, "frames_32")

    hate_json = os.path.join(in_dir, "hate.json")
    nonhate_json = os.path.join(in_dir, "nonhate.json")
    hate_out = os.path.join(in_dir, "time_unit_hate.json")
    nonhate_out = os.path.join(in_dir, "time_unit_nonhate.json")

    if not (os.path.exists(hate_json) and os.path.exists(nonhate_json)):
        logging.warning(
            f"[{ds}] skip — stage 2a/2b outputs missing at {in_dir}"
        )
        return

    if not os.path.isdir(frames_dir):
        logging.warning(
            f"[{ds}] skip — frames_32 missing at {frames_dir}; "
            "run extract_frames_32.py first"
        )
        return

    transcript_dict = build_transcript_dict(ds)
    logging.info(
        f"[{ds}] transcript_dict built for {len(transcript_dict)} videos"
    )

    os.makedirs(in_dir, exist_ok=True)
    logging.info(f"[{ds}] clip_run hate → {hate_out}")
    clip_run(hate_json, frames_dir, transcript_dict, hate_out, model, device)
    logging.info(f"[{ds}] clip_run nonhate → {nonhate_out}")
    clip_run(
        nonhate_json, frames_dir, transcript_dict, nonhate_out, model, device
    )
    logging.info(f"[{ds}] done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    if not args.dataset and not args.all:
        parser.error("Provide --dataset or --all")

    datasets = ALL_DATASETS if args.all else [args.dataset]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.info(f"MATCH stage 2b.5 (Jina-CLIP) datasets={datasets}")

    import torch
    from transformers import AutoModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info(f"Loading {args.model} on {device}")
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True
    ).to(device)
    model.eval()
    logging.info("jina-clip-v2 loaded")

    for ds in datasets:
        if ds in SKIP_VIDEOS and not SKIP_VIDEOS[ds]:
            pass
        run_one_dataset(ds, model, device)

    logging.info("All datasets done.")


if __name__ == "__main__":
    main()
