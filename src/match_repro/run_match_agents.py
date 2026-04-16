"""Run MATCH-HVD's stage 2a/2b agents on our 4 datasets — faithful to upstream.

Upstream: anonymous.4open.science/r/MATCH-HVD
Backbone: Qwen2-VL-7B-Instruct (HF transformers, not vLLM — per upstream
          `MATCH_HateMM_run.py`, `MATCH_Multi_en_run.py`,
          `MATCH_Multi_zh_run.py` which all use
          `Qwen2VLForConditionalGeneration.from_pretrained(...)` +
          `qwen_vl_utils.process_vision_info`).

Per-dataset dispatch (audited against the 3 upstream runners):

  * HateMM     → English prompt with `Audio Transcription` + `OCR`,
                 qwen_run indices `[0,2,4,6,8,10,12,14]`,
                 err_run indices  `[0,4,8,12]`.
                 Upstream: `MATCH_HateMM_run.py:39-141`, `:206-290`, `:295-296`.
  * MHClip_EN  → English prompt with `Audio Transcription` + `Title`
                 (upstream omits OCR from the prompt even though it
                 loads an OCR file — kept upstream-exact),
                 qwen_run indices `[0,1,2,3,5,7,9,11,13,14,15]`,
                 err_run indices  `[0,2,4,6,8,10,12,14]`.
                 Upstream: `MATCH_Multi_en_run.py:46-145`, `:209-293`, `:298-299`.
  * MHClip_ZH  → Chinese prompt with `音频转录` + `标题` (no OCR),
                 `ifhate` uses `"没有"` for the nonhate agent,
                 qwen_run indices `[0,2,4,6,8,10,12,14]`,
                 err_run indices  `[0,4,8,12]`.
                 Upstream: `MATCH_Multi_zh_run.py:39-134`, `:196-276`, `:280-281`.
  * ImpliHateVid → extension dataset (no upstream runner). Reuses the
                 HateMM English prompt. Our `ImpliHateVid` annotations
                 have no title or OCR, so both fields default to empty
                 strings. Flagged in README as an intentional extension.

Pipeline per video: 1 Qwen2-VL forward pass through the selected prompt.

Output schema (per agent per dataset): JSON array of
`{"id": vid, "answer": "<rationale>"}` matching upstream
`hate.json` / `nonhate.json`.

Resume support: skip video_ids already present in the output JSON.
"""

import argparse
import json
import logging
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "our_method"))
from data_utils import (  # noqa: E402
    DATASET_ROOTS,
    SKIP_VIDEOS,
    load_annotations,
)

PROJECT_ROOT = "/data/jehc223/EMNLP2"
ALL_DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]

DEFAULT_MODEL = "Qwen/Qwen2-VL-7B-Instruct"

# Per-dataset configuration, each row verbatim from the upstream
# runner that matches the dataset. ImpliHateVid reuses HateMM's
# configuration (documented as extension).
DATASET_CFG = {
    "HateMM": {
        "main_indices": [0, 2, 4, 6, 8, 10, 12, 14],
        "err_indices": [0, 4, 8, 12],
        "max_transcript": 1000,
        "max_ocr": 100,
        "max_title": 0,
        "lang": "en",
        "prompt_variant": "hatemm",
    },
    "MHClip_EN": {
        "main_indices": [0, 1, 2, 3, 5, 7, 9, 11, 13, 14, 15],
        "err_indices": [0, 2, 4, 6, 8, 10, 12, 14],
        "max_transcript": 1000,
        "max_ocr": 0,
        "max_title": 300,
        "lang": "en",
        "prompt_variant": "mhclip_en",
    },
    "MHClip_ZH": {
        "main_indices": [0, 2, 4, 6, 8, 10, 12, 14],
        "err_indices": [0, 4, 8, 12],
        "max_transcript": 1000,
        "max_ocr": 0,
        "max_title": 100,
        "lang": "zh",
        "prompt_variant": "mhclip_zh",
    },
    "ImpliHateVid": {
        "main_indices": [0, 2, 4, 6, 8, 10, 12, 14],
        "err_indices": [0, 4, 8, 12],
        "max_transcript": 1000,
        "max_ocr": 100,
        "max_title": 0,
        "lang": "en",
        "prompt_variant": "hatemm",
    },
}


# Verbatim from upstream `MATCH_HateMM_run.py` lines 83-92.
PROMPT_HATEMM = (
    "You are a hate video detector. Follow these steps:\n"
    "1. As a proposer claiming that the video {ifhate} contains hate signals, review the visual content."
    "Your task is to identify and highlight any subtle elements that support this claim.\n"
    "2. As a proposer claiming that the video {ifhate} contains hate signals, review the textual content based on the observations from the visual content."
    "Your task is to verify these findings and identify additional subtle elements that support the claim, providing comprehensive and convincing evidence.\n"
    "Audio Transcription: {transcription}\n"
    "OCR: {ocr}"
)

# Verbatim from upstream `MATCH_Multi_en_run.py` lines 89-98 (note the
# trailing `\n` after Title, which upstream includes explicitly, and
# the absence of an OCR line — upstream loads OCR but never appends it
# to the prompt).
PROMPT_MHCLIP_EN = (
    "You are a hate video detector. Follow these steps:\n"
    "1. As a proposer claiming that the video {ifhate} contains hate signals, review the visual content."
    "Your task is to identify and highlight any subtle elements that support this claim.\n"
    "2. As a proposer claiming that the video {ifhate} contains hate signals, review the textual content based on the observations from the visual content."
    "Your task is to verify these findings and identify additional subtle elements that support the claim, providing comprehensive and convincing evidence.\n"
    "Audio Transcription: {transcription}\n"
    "Title: {title}\n"
)

# Verbatim from upstream `MATCH_Multi_zh_run.py` lines 79-87.
PROMPT_MHCLIP_ZH = (
    "你是仇恨视频探测器。请按照以下步骤操作：\n"
    "1.作为声称视频{ifhate}包含仇恨信号的提议者，查看视频内容."
    "你的任务是找出并突出任何支持这一说法的微妙元素"
    "2.作为声称视频{ifhate}包含仇恨信号的提议者，根据视觉内容的观察结果审查文本内容。"
    "你的任务是核实这些发现，并找出支持该主张的其他微妙元素，提供全面而令人信服的证据。"
    "音频转录： {transcription}\n"
    "标题： {title}\n"
)


def build_prompt(variant, ifhate, transcription, title, ocr):
    if variant == "hatemm":
        return PROMPT_HATEMM.format(
            ifhate=ifhate, transcription=transcription, ocr=ocr
        )
    if variant == "mhclip_en":
        return PROMPT_MHCLIP_EN.format(
            ifhate=ifhate, transcription=transcription, title=title
        )
    if variant == "mhclip_zh":
        return PROMPT_MHCLIP_ZH.format(
            ifhate=ifhate, transcription=transcription, title=title
        )
    raise ValueError(f"unknown prompt variant: {variant}")


def ifhate_for(dataset, agent):
    """Return the `ifhate` token upstream uses for each (dataset, agent) pair.

    Upstream wiring:
      * `MATCH_HateMM_run.py:11` and `MATCH_Multi_en_run.py:11` use
        `""` for hate, `"does not"` for nonhate.
      * `MATCH_Multi_zh_run.py:11` uses `""` for hate, `"没有"` for
        nonhate (Chinese equivalent of "does not").
    """
    if agent == "hate":
        return ""
    # nonhate agent
    if DATASET_CFG[dataset]["lang"] == "zh":
        return "没有"
    return "does not"


def get_frame_uris(dataset, vid, indices):
    root = DATASET_ROOTS[dataset]
    folder = os.path.join(root, "frames_16", vid)
    paths = [os.path.join(folder, f"frame_{i:03d}.jpg") for i in indices]
    if not all(os.path.exists(p) for p in paths):
        return None
    return [os.path.abspath(p) for p in paths]


def load_existing(path):
    if not os.path.exists(path):
        return [], set()
    try:
        with open(path) as f:
            data = json.load(f)
        ids = {e["id"] for e in data if "id" in e}
        return data, ids
    except Exception:
        return [], set()


def save_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def truncate(s, n):
    if not s or n <= 0:
        return ""
    s = str(s)
    return s[:n] if len(s) > n else s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--agent", required=True, choices=["hate", "nonhate"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if not args.dataset and not args.all:
        parser.error("Provide --dataset or --all")

    datasets = ALL_DATASETS if args.all else [args.dataset]

    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.info(
        f"MATCH agent={args.agent} model={args.model} datasets={datasets}"
    )

    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    processor = AutoProcessor.from_pretrained(args.model)
    model.eval()
    logging.info("Qwen2-VL-7B-Instruct loaded")

    def run_one(frame_uris, text_prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frame_uris},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda:0")
        with torch.no_grad():
            out_ids = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens
            )
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
        return processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    for ds in datasets:
        cfg = DATASET_CFG[ds]
        ann = load_annotations(ds)
        split_path = os.path.join(
            DATASET_ROOTS[ds], "splits", f"{args.split}_clean.csv"
        )
        if not os.path.isfile(split_path):
            from data_utils import generate_clean_splits
            generate_clean_splits(ds)
        with open(split_path) as f:
            vids = [line.strip() for line in f if line.strip()]

        out_dir = os.path.join(
            PROJECT_ROOT, "results", "match_qwen2vl_7b", ds
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{args.agent}.json")
        existing, done = load_existing(out_path)
        logging.info(
            f"[{ds}] {len(vids)} videos, {len(done)} already done, "
            f"prompt={cfg['prompt_variant']}, main_indices={cfg['main_indices']}"
        )

        ifhate = ifhate_for(ds, args.agent)
        results = list(existing)
        n_processed = 0
        n_skipped = 0
        n_errors = 0
        t0 = time.time()
        for vid in vids:
            if vid in done:
                continue
            if vid in SKIP_VIDEOS.get(ds, set()):
                continue
            entry = ann.get(vid)
            if entry is None:
                n_skipped += 1
                continue
            transcription = truncate(
                entry.get("transcript", ""), cfg["max_transcript"]
            )
            title = truncate(entry.get("title", ""), cfg["max_title"])
            ocr = truncate(entry.get("ocr", ""), cfg["max_ocr"])

            text_prompt = build_prompt(
                cfg["prompt_variant"], ifhate, transcription, title, ocr
            )

            frame_uris = get_frame_uris(ds, vid, cfg["main_indices"])
            if frame_uris is None:
                logging.warning(f"  {vid}: missing frames_16/ main indices")
                results.append({"id": vid, "answer": ""})
                n_skipped += 1
                continue

            try:
                answer = run_one(frame_uris, text_prompt)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
                    logging.error(
                        f"  {vid}: {type(e).__name__}: {str(e)[:120]}"
                    )
                    answer = ""
                    n_errors += 1
                else:
                    torch.cuda.empty_cache()
                    logging.warning(
                        f"  {vid}: OOM, retrying with err_indices "
                        f"{cfg['err_indices']}"
                    )
                    err_uris = get_frame_uris(ds, vid, cfg["err_indices"])
                    if err_uris is None:
                        answer = ""
                        n_errors += 1
                    else:
                        try:
                            answer = run_one(err_uris, text_prompt)
                        except Exception as e2:
                            logging.error(
                                f"  {vid}: err_run failed: {e2}"
                            )
                            answer = ""
                            n_errors += 1
            except Exception as e:
                logging.error(
                    f"  {vid}: {type(e).__name__}: {str(e)[:120]}"
                )
                answer = ""
                n_errors += 1

            results.append({"id": vid, "answer": answer})
            n_processed += 1

            if n_processed % 25 == 0:
                save_atomic(out_path, results)
                rate = n_processed / (time.time() - t0)
                logging.info(
                    f"  [{ds}] {n_processed} new / {len(results)} total / "
                    f"{rate:.2f} vid/s / errs={n_errors}"
                )

        save_atomic(out_path, results)
        logging.info(
            f"[{ds}] done. processed={n_processed} skipped={n_skipped} "
            f"errs={n_errors}"
        )

    logging.info("All datasets done.")


if __name__ == "__main__":
    main()
