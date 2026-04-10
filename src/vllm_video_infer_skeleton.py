#!/usr/bin/env python3
"""Minimal vLLM video MLLM inference skeleton.

Idea-agnostic starting point for any future single-dataset, fresh-vLLM inference
job. Replaces the deleted `mllm_tension_cache.py` / `tensiongate.py` /
`triview_gate_single.py` family. **Does not** encode any tension / relation /
counterfactual / tri-view methodology — only the reusable scaffolding:

- dataset config + annotation loading
- split-filtered target id selection
- video / frame-directory loading helpers
- vLLM init with OOM-fallback on max_num_seqs
- per-video prompt placeholder (PUT YOUR PROMPT HERE)
- JSON checkpoint with atomic save + resume
- progress logging compatible with TARGET_LOOP monitoring

Usage:
    python src/vllm_video_infer_skeleton.py \\
        --dataset MultiHateClip_EN \\
        --output_dir /data/jehc223/EMNLP2/results/mllm_skeleton \\
        --use_splits_only

Then customise:
    - `build_prompt()` — write the actual user prompt
    - `parse_response()` — parse the model output into your structured form
    - (optional) tweak `SamplingParams`, `num_frames`, `max_model_len`
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import sys
import time
from typing import Any

import numpy as np
import torch

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


# ── Dataset config ───────────────────────────────────────────────────────────

def get_dataset_config(dataset_name: str) -> dict:
    configs = {
        "HateMM": {
            "annotation": "/data/jehc223/EMNLP2/baseline/HVGuard/datasets/HateMM/data (base).json",
            "video_dir": "/data/jehc223/HateMM/video",
            "frames_dir": "/data/jehc223/HateMM/frames",
            "splits_dir": "/data/jehc223/HateMM/splits",
        },
        "MultiHateClip_CN": {
            "annotation": "/data/jehc223/EMNLP2/baseline/HVGuard/datasets/Multihateclip/Chinese/data.json",
            "video_dir": "/data/jehc223/Multihateclip/Chinese/video",
            "frames_dir": "/data/jehc223/Multihateclip/Chinese/frames",
            "splits_dir": "/data/jehc223/Multihateclip/Chinese/splits",
        },
        "MultiHateClip_EN": {
            "annotation": "/data/jehc223/EMNLP2/baseline/HVGuard/datasets/Multihateclip/English/data.json",
            "video_dir": "/data/jehc223/Multihateclip/English/video",
            "frames_dir": "/data/jehc223/Multihateclip/English/frames",
            "splits_dir": "/data/jehc223/Multihateclip/English/splits",
        },
    }
    return configs[dataset_name]


def load_annotations(annotation_path: str) -> dict:
    with open(annotation_path) as f:
        data = json.load(f)
    return {d["Video_ID"]: d for d in data}


def load_target_ids(cfg: dict, use_splits_only: bool) -> list:
    annotations = load_annotations(cfg["annotation"])
    if not use_splits_only:
        return list(annotations.keys())
    split_ids: list[str] = []
    for split in ["train", "valid", "test"]:
        split_path = os.path.join(cfg["splits_dir"], f"{split}.csv")
        with open(split_path, "r", encoding="utf-8") as f:
            split_ids.extend([line.strip() for line in f if line.strip()])
    return [vid for vid in dict.fromkeys(split_ids) if vid in annotations]


# ── Video loading ────────────────────────────────────────────────────────────

def find_video_file(video_dir: str, video_id: str) -> str | None:
    for ext in [".mp4", ".webm", ".avi", ".mkv", ".mov"]:
        path = os.path.join(video_dir, f"{video_id}{ext}")
        if os.path.exists(path):
            return path
    matches = glob.glob(os.path.join(video_dir, f"{video_id}.*"))
    return matches[0] if matches else None


def load_video_with_metadata(video_path: str, num_frames: int = 16) -> tuple:
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path)
    total = len(vr)
    if total == 0:
        raise ValueError(f"Empty video: {video_path}")
    fps = vr.get_avg_fps() or 24.0
    if total <= num_frames:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    frames = vr.get_batch(indices).asnumpy()
    metadata = {
        "fps": fps,
        "total_num_frames": total,
        "frames_indices": indices,
        "do_sample_frames": False,
    }
    return frames, metadata


def load_frames_from_dir_with_metadata(frames_dir: str, num_frames: int = 16) -> tuple:
    from PIL import Image
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    if not frame_files:
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not frame_files:
        raise ValueError(f"No frames found in {frames_dir}")
    total = len(frame_files)
    indices = (
        list(range(total)) if total <= num_frames
        else np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    )
    frames = [np.array(Image.open(frame_files[i]).convert("RGB")) for i in indices]
    metadata = {
        "fps": 1.0,
        "total_num_frames": total,
        "frames_indices": indices,
        "do_sample_frames": False,
    }
    return np.stack(frames), metadata


def load_video_or_frames(cfg: dict, vid: str, num_frames: int):
    """Try direct video first, then frame directory. Returns None if unavailable."""
    video_path = find_video_file(cfg["video_dir"], vid)
    if video_path:
        try:
            return load_video_with_metadata(video_path, num_frames=num_frames)
        except Exception:
            pass
    frames_dir = os.path.join(cfg["frames_dir"], vid)
    if os.path.isdir(frames_dir):
        try:
            return load_frames_from_dir_with_metadata(frames_dir, num_frames=num_frames)
        except Exception:
            pass
    return None


# ── Checkpoint ───────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str) -> dict:
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            return json.load(f)
    return {}


def save_checkpoint(ckpt_path: str, results: dict) -> None:
    tmp_path = ckpt_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, ckpt_path)


# ── Prompt + parsing (placeholder — fill in for your method) ─────────────────

def build_prompt(title: str, transcript: str) -> str:
    """PLACEHOLDER. Replace with the prompt for your actual method."""
    title = (title or "").strip() or "(no title)"
    transcript = (transcript or "").strip() or "(no transcript)"
    return (
        f"Title: {title[:200]}\n"
        f"Transcript: {transcript[:1500]}\n\n"
        "Describe what you see and hear in this video in 2-3 sentences."
    )


def parse_response(raw: str) -> dict:
    """PLACEHOLDER. Replace with structured parsing for your actual output schema."""
    return {"raw": (raw or "").strip()}


# ── vLLM init with OOM fallback ──────────────────────────────────────────────

def init_vllm(args):
    from vllm import LLM
    max_num_seqs = args.max_num_seqs
    while max_num_seqs >= 1:
        try:
            print(f"\nTrying LLM init: max_num_seqs={max_num_seqs}", flush=True)
            llm = LLM(
                model=MODEL_ID,
                tensor_parallel_size=1,
                max_model_len=args.max_model_len,
                max_num_seqs=max_num_seqs,
                gpu_memory_utilization=args.gpu_mem,
                mm_processor_cache_gb=args.mm_processor_cache_gb,
                trust_remote_code=True,
                limit_mm_per_prompt={"video": 1},
                dtype="auto",
                enforce_eager=True,
            )
            print("LLM initialized successfully!", flush=True)
            return llm
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "memory" in str(e).lower() or "cuda" in str(e).lower():
                new_seqs = max(1, max_num_seqs // 2)
                print(f"OOM, reducing max_num_seqs {max_num_seqs} -> {new_seqs}", flush=True)
                if new_seqs == max_num_seqs:
                    raise
                max_num_seqs = new_seqs
                torch.cuda.empty_cache()
                gc.collect()
            else:
                raise
    raise RuntimeError("Could not initialize LLM at any max_num_seqs")


# ── Main loop ────────────────────────────────────────────────────────────────

def _print_progress(done: int, total: int, start_time: float, dataset_name: str) -> None:
    elapsed = time.time() - start_time
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    print(
        f"  [{dataset_name}] {done}/{total} ({rate:.2f} vid/s, "
        f"elapsed {elapsed/60:.1f}m, ETA {eta/60:.1f}m)",
        flush=True,
    )


def run(args) -> None:
    from vllm import SamplingParams

    cfg = get_dataset_config(args.dataset)
    out_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    ckpt_path = os.path.join(out_dir, "infer_results.json")
    results = load_checkpoint(ckpt_path)
    print(f"Loaded {len(results)} existing results from checkpoint", flush=True)

    annotations = load_annotations(cfg["annotation"])
    target_ids = load_target_ids(cfg, args.use_splits_only)
    remaining = [vid for vid in target_ids if vid not in results]
    print(
        f"Total target: {len(target_ids)}, complete: {len(target_ids)-len(remaining)}, "
        f"remaining: {len(remaining)}",
        flush=True,
    )
    if not remaining:
        print("Nothing to do.", flush=True)
        return

    llm = init_vllm(args)
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, top_p=1.0)

    start = time.time()
    done = 0
    for vid in remaining:
        ann = annotations[vid]
        title = ann.get("Title", "")
        transcript = ann.get("Transcript", "")
        prompt_text = build_prompt(title, transcript)

        video_data = load_video_or_frames(cfg, vid, args.num_frames)

        if video_data is not None:
            prompt = (
                "<|im_start|>system\nYou are an expert video analyst.<|im_end|>\n"
                "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>\n"
                f"{prompt_text}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            inp = {"prompt": prompt, "multi_modal_data": {"video": video_data}}
        else:
            prompt = (
                "<|im_start|>system\nYou are an expert video analyst.<|im_end|>\n"
                f"<|im_start|>user\n{prompt_text}\n"
                "Note: visual content is unavailable for this item.<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            inp = {"prompt": prompt}

        try:
            out = llm.generate([inp], sampling_params=sp)
            raw = out[0].outputs[0].text.strip()
            results[vid] = parse_response(raw)
        except Exception as e:
            print(f"  Inference failed for {vid}: {str(e)[:120]}", flush=True)
            results[vid] = {"raw": "", "error": str(e)[:200]}

        done += 1
        if done % args.save_every == 0:
            _print_progress(done, len(remaining), start, args.dataset)
            save_checkpoint(ckpt_path, results)

    save_checkpoint(ckpt_path, results)
    _print_progress(done, len(remaining), start, args.dataset)
    print(f"\nDone. Total cached: {len(results)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Idea-agnostic vLLM video MLLM inference skeleton."
    )
    parser.add_argument("--dataset", required=True,
                        choices=["HateMM", "MultiHateClip_CN", "MultiHateClip_EN"])
    parser.add_argument("--output_dir", default="/data/jehc223/EMNLP2/results/mllm_skeleton")
    parser.add_argument("--use_splits_only", action="store_true",
                        help="Restrict to ids that appear in train/valid/test split CSVs.")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--max_num_seqs", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--gpu_mem", type=float, default=0.90)
    parser.add_argument("--mm_processor_cache_gb", type=float, default=0.0)
    parser.add_argument("--save_every", type=int, default=10)
    args = parser.parse_args()

    try:
        run(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user; checkpoint preserved.", flush=True)
        sys.exit(130)


if __name__ == "__main__":
    main()
