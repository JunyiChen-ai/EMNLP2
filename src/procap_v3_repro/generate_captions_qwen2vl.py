"""Pro-Cap V3 — Qwen2-VL-7B multi-image (16-frame) caption generator.

Upstream Pro-Cap (`external_repos/procap/codes/Pro-Cap-Generation.ipynb`)
runs 8 fixed VQA probes on a single meme image via BLIP-2 and writes
the answers to a per-dataset pickle file that the supervised PBM
trainer (`scr/dataset.py:114`) loads. We adapt to video by:

  1. Replacing the BLIP-2 single-image captioner with **Qwen2-VL-7B-
     Instruct via vLLM**, which natively handles multi-image inputs.
     (User directive 2026-04-15: model already cached locally from
     MATCH stage 2c; no extra download.)
  2. Passing **16 frames per video** via Qwen2-VL's chat template
     (`{"type": "image", "image": "file://..."}` per frame + final
     text entry). The model sees all 16 frames at once per probe call.
  3. Replacing `"in the image"` → `"in the video"` in the 8 upstream
     probe strings; everything else byte-for-byte from the notebook.
  4. Adding a 9th "gen" probe — `"describe the video in one sentence."` —
     equivalent to the upstream notebook's general caption cell.
     Stored under the `"gen"` key.

Output: per-dataset, per-split pickle file:
    results/procap_v3/<dataset>/captions_<split>.pkl
Schema: `{vid: {race, gender, animal, person, country, what_animal,
                 disabled, religion, gen}}` (all str values).

Resume support: if the output pkl already exists, videos already in
the dict are skipped. Per-N-videos checkpoint write + fsync.
"""

import argparse
import logging
import os
import pickle
import sys
import time

# Allow PIL to finish decoding JPGs that were written with a truncated
# tail (~25 bytes short). Same hardening applied to lorehm retrieval
# after job 8481 hit a corrupt frame in MHClip_ZH test frames_16.
from PIL import ImageFile  # noqa: E402
ImageFile.LOAD_TRUNCATED_IMAGES = True

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "our_method"))
from data_utils import DATASET_ROOTS, SKIP_VIDEOS  # noqa: E402

PROJECT_ROOT = "/data/jehc223/EMNLP2"
ALL_DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]
DEFAULT_MODEL = "Qwen/Qwen2-VL-7B-Instruct"
NUM_FRAMES = 16
CHECKPOINT_EVERY = 20

# 8 VQA probes — byte-for-byte from upstream
# `external_repos/procap/codes/Pro-Cap-Generation.ipynb` (cells at
# notebook lines 275, 366, 441, 516, 591, 666, 740, 814) with the only
# substring substitution being `"in the image"` -> `"in the video"`.
PROCAP_PROBES = [
    ("race",         "what is the race of the person in the video?"),
    ("gender",       "what is the gender of the person in the video?"),
    ("animal",       "is there an animal in the video?"),
    ("person",       "is there a person in the video?"),
    ("country",      "which country does the person in the video come from?"),
    ("what_animal",  "what animal is in the video?"),
    ("disabled",     "are there disabled people in the video?"),
    ("religion",     "what is the religion of the person in the video?"),
]

# 9th probe — equivalent to upstream notebook's general caption cell
# ("a photo of ..."); stored under "gen" key, matching the upstream
# `mode+'_generic.pkl'` file from `scr/dataset.py:96-98`.
GEN_PROBE = ("gen", "describe the video in one sentence.")


def load_split_video_ids(dataset, split):
    csv_path = os.path.join(
        DATASET_ROOTS[dataset], "splits", f"{split}_clean.csv"
    )
    if not os.path.isfile(csv_path):
        from data_utils import generate_clean_splits
        generate_clean_splits(dataset)
    with open(csv_path) as f:
        return [line.strip() for line in f if line.strip()]


def frame_paths_for(dataset, vid):
    """Return the absolute paths for frame_000.jpg .. frame_015.jpg.

    Returns None if any frame is missing.
    """
    folder = os.path.join(DATASET_ROOTS[dataset], "frames_16", vid)
    paths = [
        os.path.join(folder, f"frame_{i:03d}.jpg") for i in range(NUM_FRAMES)
    ]
    if not all(os.path.exists(p) for p in paths):
        return None
    return paths


def load_pil_frames(paths):
    from PIL import Image
    return [Image.open(p).convert("RGB") for p in paths]


def build_messages(frame_paths, probe_text):
    """Build a Qwen2-VL chat template messages list with 16 image
    entries followed by the probe text. Image references use
    `file://` URIs so vLLM can dispatch to its multi-modal loader.
    """
    content = [
        {"type": "image", "image": f"file://{os.path.abspath(p)}"}
        for p in frame_paths
    ]
    content.append({"type": "text", "text": probe_text})
    return [{"role": "user", "content": content}]


def output_path(dataset, split):
    out_dir = os.path.join(PROJECT_ROOT, "results", "procap_v3", dataset)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"captions_{split}.pkl")


def load_existing(out_path):
    if not os.path.exists(out_path):
        return {}
    try:
        with open(out_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        logging.warning(f"failed to read {out_path}, starting fresh")
        return {}


def save_checkpoint(out_path, captions):
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(captions, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)


def run_one_dataset(dataset, split, llm, processor, sampling, args):
    vids = load_split_video_ids(dataset, split)
    skip_set = SKIP_VIDEOS.get(dataset, set())
    out_path = output_path(dataset, split)
    captions = load_existing(out_path)
    logging.info(
        f"[{dataset}/{split}] {len(vids)} vids, {len(captions)} already done"
    )

    n_processed = 0
    t0 = time.time()
    for vid in vids:
        if vid in skip_set:
            continue
        if vid in captions:
            continue
        paths = frame_paths_for(dataset, vid)
        if paths is None:
            logging.warning(f"  {vid}: missing frames, skipping")
            continue
        try:
            pil_frames = load_pil_frames(paths)
        except Exception as e:
            logging.error(f"  {vid}: PIL load failed: {e}")
            continue

        record = {}
        for probe_key, probe_text in PROCAP_PROBES + [GEN_PROBE]:
            messages = build_messages(paths, probe_text)
            prompt = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            try:
                outputs = llm.generate(
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"image": pil_frames},
                    },
                    sampling_params=sampling,
                )
                text = outputs[0].outputs[0].text.strip()
            except Exception as e:
                logging.error(f"  {vid}/{probe_key}: vLLM call failed: {e}")
                text = ""
            record[probe_key] = text
        captions[vid] = record
        n_processed += 1

        if n_processed % CHECKPOINT_EVERY == 0:
            save_checkpoint(out_path, captions)
            rate = n_processed / (time.time() - t0)
            logging.info(
                f"  [{dataset}/{split}] {n_processed} new / {rate:.2f} vid/s"
            )

    save_checkpoint(out_path, captions)
    logging.info(
        f"[{dataset}/{split}] done. {n_processed} new captions written to {out_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Pro-Cap V3 multi-image caption generator (Qwen2-VL-7B vLLM)"
    )
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--split", choices=["train", "test"], required=True
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-tokens", type=int, default=128)
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
        f"Pro-Cap V3 captioner: model={args.model} datasets={datasets} "
        f"split={args.split} frames={NUM_FRAMES} probes={len(PROCAP_PROBES)+1}"
    )

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": NUM_FRAMES},
        # Fix 2026-04-16: cap per-image token budget so vLLM's Qwen2-VL
        # processor auto-resizes 1080x1920 mp4 frames down to a safe
        # ~448x448 ≈ 1024 visual tokens each. Without this cap, early
        # MHClip_EN/ZH videos with native high-res frames hung the
        # engine silently, writing empty strings.
        mm_processor_kwargs={"max_pixels": 200704},
        trust_remote_code=True,
    )
    # Greedy decoding for caption generation. Pro-Cap captions feed a
    # downstream supervised classifier; sampling variance only adds
    # noise to the training set without aiding the captioner's role.
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
    )
    logging.info("vLLM Qwen2-VL-7B-Instruct loaded")

    for ds in datasets:
        run_one_dataset(ds, args.split, llm, processor, sampling, args)

    logging.info("All datasets done.")


if __name__ == "__main__":
    main()
