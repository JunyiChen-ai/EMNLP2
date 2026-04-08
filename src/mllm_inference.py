#!/usr/bin/env python3
"""MLLM video inference using Qwen3-VL-8B via vLLM.

Generates structured analysis for hate video detection.
Supports checkpointing/resume, OOM fallback, and frame-based fallback for corrupted videos.
"""

import argparse
import json
import os
import sys
import time
import gc
import glob
from pathlib import Path

import numpy as np
import torch

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def load_video_with_metadata(video_path: str, num_frames: int = 16) -> tuple:
    """Load video, uniformly sample frames, return (frames, metadata) for Qwen3-VL."""
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path)
    total = len(vr)
    if total == 0:
        raise ValueError(f"Empty video: {video_path}")
    fps = vr.get_avg_fps()
    if fps <= 0:
        fps = 24.0  # default fallback
    if total <= num_frames:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    frames = vr.get_batch(indices).asnumpy()  # (N, H, W, C)
    metadata = {
        "fps": fps,
        "total_num_frames": total,
        "frames_indices": indices,
        "do_sample_frames": False,  # already pre-sampled
    }
    return frames, metadata


def load_frames_from_dir_with_metadata(frames_dir: str, num_frames: int = 16) -> tuple:
    """Load pre-extracted frames from directory as fallback, return (frames, metadata)."""
    from PIL import Image
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    if not frame_files:
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not frame_files:
        raise ValueError(f"No frames found in {frames_dir}")

    total = len(frame_files)
    if total <= num_frames:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()

    frames = []
    for i in indices:
        img = Image.open(frame_files[i]).convert("RGB")
        frames.append(np.array(img))
    frames_array = np.stack(frames)
    # Assume ~1 fps for pre-extracted frames (typical extraction rate)
    metadata = {
        "fps": 1.0,
        "total_num_frames": total,
        "frames_indices": indices,
        "do_sample_frames": False,
    }
    return frames_array, metadata


def build_analysis_prompt(title: str, transcript: str) -> str:
    """Build structured analysis prompt for hate detection."""
    title_text = title.strip() if title and title.strip() else "(no title)"
    transcript_text = transcript.strip() if transcript and transcript.strip() else "(no transcript)"

    return f"""Analyze this video for hateful or offensive content. Consider visual, textual, and cross-modal cues.

Title: {title_text}
Transcript: {transcript_text}

Provide a structured analysis:
1. VISUAL CONTENT: Describe key visual elements, symbols, gestures, text overlays, and any potentially harmful imagery.
2. SPEECH/TEXT ANALYSIS: Analyze the transcript and title for hate speech, slurs, derogatory language, coded language, or dog whistles.
3. CROSS-MODAL INTERACTION: How do visual and textual elements interact? Does the visual context change the meaning of the text or vice versa? Is there sarcasm, irony, or implicit meaning that only emerges from combining modalities?
4. HATE ASSESSMENT: Based on all evidence, is this content hateful/offensive or normal? What specific elements make it hateful, or what makes it benign despite potentially sensitive topics?
5. CONFIDENCE: How confident are you in your assessment? What ambiguities exist?

Be thorough but concise. Focus on evidence-based reasoning."""


def load_checkpoint(ckpt_path: str) -> dict:
    """Load checkpoint of already-processed video IDs."""
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            return json.load(f)
    return {}


def save_checkpoint(ckpt_path: str, results: dict):
    """Save checkpoint atomically."""
    tmp_path = ckpt_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, ckpt_path)


def get_dataset_config(dataset_name: str) -> dict:
    """Get dataset-specific paths and configuration."""
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
    """Load annotations indexed by Video_ID."""
    with open(annotation_path) as f:
        data = json.load(f)
    return {d["Video_ID"]: d for d in data}


def find_video_file(video_dir: str, video_id: str) -> str:
    """Find video file with various extensions."""
    for ext in [".mp4", ".webm", ".avi", ".mkv", ".mov"]:
        path = os.path.join(video_dir, f"{video_id}{ext}")
        if os.path.exists(path):
            return path
    # Try glob
    matches = glob.glob(os.path.join(video_dir, f"{video_id}.*"))
    if matches:
        return matches[0]
    return None


def run_inference(args):
    from vllm import LLM, SamplingParams

    dataset_name = args.dataset
    cfg = get_dataset_config(dataset_name)
    num_frames = args.num_frames
    output_dir = os.path.join(args.output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    ckpt_path = os.path.join(output_dir, "mllm_results.json")
    results = load_checkpoint(ckpt_path)
    print(f"Loaded {len(results)} existing results from checkpoint")

    # Load annotations
    annotations = load_annotations(cfg["annotation"])
    print(f"Loaded {len(annotations)} annotations")

    # Get all video IDs (process all splits)
    all_ids = list(annotations.keys())
    remaining = [vid for vid in all_ids if vid not in results]
    print(f"Total: {len(all_ids)}, Already done: {len(results)}, Remaining: {len(remaining)}")

    if not remaining:
        print("All videos already processed!")
        return

    # Initialize vLLM
    max_num_seqs = args.max_num_seqs
    llm = None

    while max_num_seqs >= 1:
        try:
            print(f"\nTrying LLM init: max_num_seqs={max_num_seqs}, num_frames={num_frames}")
            llm = LLM(
                model=MODEL_ID,
                tensor_parallel_size=1,
                max_model_len=args.max_model_len,
                max_num_seqs=max_num_seqs,
                gpu_memory_utilization=args.gpu_mem,
                trust_remote_code=True,
                limit_mm_per_prompt={"video": 1},
                dtype="auto",
                enforce_eager=True,
            )
            print("LLM initialized successfully!")
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            err_msg = str(e).lower()
            if "out of memory" in err_msg or "cuda" in err_msg or "memory" in err_msg:
                print(f"OOM during init, reducing: max_num_seqs {max_num_seqs} -> {max(1, max_num_seqs // 2)}")
                max_num_seqs = max(1, max_num_seqs // 2)
                if num_frames > 4:
                    num_frames = max(4, num_frames // 2)
                    print(f"  Also reducing num_frames to {num_frames}")
                torch.cuda.empty_cache()
                gc.collect()
            else:
                raise

    if llm is None:
        print("ERROR: Could not initialize LLM")
        sys.exit(1)

    sp = SamplingParams(max_tokens=1024, temperature=0.0, top_p=1.0)

    # Process in chunks
    chunk_size = max(max_num_seqs, 4)
    total_processed = 0
    start_time = time.time()

    for chunk_start in range(0, len(remaining), chunk_size):
        chunk_ids = remaining[chunk_start: chunk_start + chunk_size]
        inputs = []
        input_vids = []

        for vid in chunk_ids:
            ann = annotations[vid]
            title = ann.get("Title", "")
            transcript = ann.get("Transcript", "")
            text_prompt = build_analysis_prompt(title, transcript)

            # Try to load video with metadata
            video_path = find_video_file(cfg["video_dir"], vid)
            video_data = None  # tuple of (frames, metadata)

            if video_path:
                try:
                    video_data = load_video_with_metadata(video_path, num_frames=num_frames)
                except Exception as e:
                    print(f"  Video load failed for {vid}: {e}")

            # Fallback to pre-extracted frames
            if video_data is None:
                frames_dir = os.path.join(cfg["frames_dir"], vid)
                if os.path.isdir(frames_dir):
                    try:
                        video_data = load_frames_from_dir_with_metadata(frames_dir, num_frames=num_frames)
                        print(f"  Using frame fallback for {vid}")
                    except Exception as e:
                        print(f"  Frame fallback also failed for {vid}: {e}")

            if video_data is None:
                # No video/frames available - generate text-only analysis
                print(f"  WARNING: No video/frames for {vid}, text-only fallback")
                results[vid] = {
                    "analysis": f"[TEXT-ONLY ANALYSIS - no video available]\n"
                                f"Title: {title}\nTranscript: {transcript}\n"
                                f"Based on text only, cannot determine visual content.",
                    "title": title,
                    "transcript": transcript,
                    "label": annotations[vid].get("Label", ""),
                }
                continue

            # Qwen3-VL prompt format - pass video as (frames, metadata) tuple
            prompt = (
                "<|im_start|>system\nYou are an expert content moderator analyzing videos for hateful content.<|im_end|>\n"
                "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>\n"
                f"{text_prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"video": video_data},
            })
            input_vids.append(vid)

        if not inputs:
            continue

        # Batch inference with OOM fallback
        try:
            outputs = llm.generate(inputs, sampling_params=sp)
        except Exception as e:
            print(f"Batch inference failed: {str(e)[:200]}")
            print("Falling back to one-by-one processing...")
            torch.cuda.empty_cache()
            outputs = []
            for i, single_input in enumerate(inputs):
                try:
                    out = llm.generate([single_input], sampling_params=sp)
                    outputs.extend(out)
                except Exception as e2:
                    print(f"  Single inference failed for {input_vids[i]}: {str(e2)[:150]}")
                    # Try with fewer frames
                    try:
                        frames_arr, meta = single_input["multi_modal_data"]["video"]
                        if len(frames_arr) > 4:
                            step = max(1, len(frames_arr) // 4)
                            reduced = frames_arr[::step][:4]
                            new_indices = list(range(0, len(frames_arr), step))[:4]
                            new_meta = {**meta, "frames_indices": new_indices}
                            single_input["multi_modal_data"]["video"] = (reduced, new_meta)
                            out = llm.generate([single_input], sampling_params=sp)
                            outputs.extend(out)
                            print(f"  Succeeded with reduced frames for {input_vids[i]}")
                        else:
                            outputs.append(None)
                    except Exception:
                        outputs.append(None)

        # Collect results
        for i, vid in enumerate(input_vids):
            if i < len(outputs) and outputs[i] is not None:
                raw = outputs[i].outputs[0].text
                results[vid] = {
                    "analysis": raw.strip(),
                    "title": annotations[vid].get("Title", ""),
                    "transcript": annotations[vid].get("Transcript", ""),
                    "label": annotations[vid].get("Label", ""),
                }
            else:
                results[vid] = {
                    "analysis": "ERROR: inference failed",
                    "title": annotations[vid].get("Title", ""),
                    "transcript": annotations[vid].get("Transcript", ""),
                    "label": annotations[vid].get("Label", ""),
                }

        total_processed += len(chunk_ids)
        elapsed = time.time() - start_time
        rate = total_processed / elapsed if elapsed > 0 else 0
        print(f"  [{dataset_name}] {total_processed}/{len(remaining)} "
              f"({rate:.1f} vid/s, elapsed {elapsed:.0f}s)")

        # Save checkpoint every chunk
        save_checkpoint(ckpt_path, results)

    print(f"\nDone! Total results: {len(results)}")
    save_checkpoint(ckpt_path, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["HateMM", "MultiHateClip_CN", "MultiHateClip_EN"])
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--max_num_seqs", type=int, default=16)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--gpu_mem", type=float, default=0.92)
    parser.add_argument("--output_dir", default="/data/jehc223/EMNLP2/results/mllm")
    args = parser.parse_args()
    run_inference(args)
