#!/usr/bin/env python3
"""MLLM scoring inference: structured JSON output for feature extraction.

Instead of encoding long text analysis, this prompt asks the MLLM to output
structured scores that can be directly used as classifier features.
"""

import argparse
import json
import os
import sys
import time
import gc
import glob
import re
from pathlib import Path

import numpy as np
import torch

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def load_video_with_metadata(video_path, num_frames=16):
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path)
    total = len(vr)
    if total == 0:
        raise ValueError(f"Empty video: {video_path}")
    fps = vr.get_avg_fps()
    if fps <= 0:
        fps = 24.0
    if total <= num_frames:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    frames = vr.get_batch(indices).asnumpy()
    metadata = {
        "fps": fps, "total_num_frames": total,
        "frames_indices": indices, "do_sample_frames": False,
    }
    return frames, metadata


def load_frames_fallback(frames_dir, num_frames=16):
    from PIL import Image
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    if not frame_files:
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not frame_files:
        raise ValueError(f"No frames in {frames_dir}")
    total = len(frame_files)
    indices = np.linspace(0, total - 1, min(num_frames, total), dtype=int).tolist()
    frames = np.stack([np.array(Image.open(frame_files[i]).convert("RGB")) for i in indices])
    return frames, {"fps": 1.0, "total_num_frames": total, "frames_indices": indices, "do_sample_frames": False}


def build_scoring_prompt(title, transcript):
    title = title.strip() if title and title.strip() else "(none)"
    transcript = transcript.strip() if transcript and transcript.strip() else "(none)"

    return f"""Analyze this video and its metadata for hateful/offensive content.

Title: {title}
Transcript: {transcript}

Rate each dimension from 0-10. Output ONLY a JSON object with these exact keys:
{{
  "hate_speech_score": <0-10, text-level hate speech severity>,
  "visual_hate_score": <0-10, visual hate indicators like symbols, violence>,
  "cross_modal_score": <0-10, how visual and text interact to create hate>,
  "implicit_hate_score": <0-10, implicit/coded hate content>,
  "overall_hate_score": <0-10, overall hateful content level>,
  "confidence": <0-10, confidence in assessment>,
  "classification": "<hateful or normal>",
  "key_evidence": "<brief 1-sentence summary of main evidence>"
}}

Output ONLY valid JSON, nothing else. /no_think"""


def parse_json_response(text):
    """Extract JSON from model output, handling common issues."""
    # Try to find JSON block
    text = text.strip()

    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try with relaxed parsing (single quotes, trailing commas)
    try:
        cleaned = text.replace("'", '"')
        cleaned = re.sub(r',\s*}', '}', cleaned)
        match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass

    # Fallback: extract what we can
    result = {
        "hate_speech_score": 5, "visual_hate_score": 5,
        "cross_modal_score": 5, "implicit_hate_score": 5,
        "overall_hate_score": 5, "confidence": 5,
        "classification": "unknown", "key_evidence": "parse_failed"
    }
    # Try to extract individual scores
    for key in ["hate_speech_score", "visual_hate_score", "cross_modal_score",
                 "implicit_hate_score", "overall_hate_score", "confidence"]:
        match = re.search(rf'"{key}":\s*(\d+)', text)
        if match:
            result[key] = int(match.group(1))
    cls_match = re.search(r'"classification":\s*"(hateful|normal)"', text, re.I)
    if cls_match:
        result["classification"] = cls_match.group(1).lower()

    return result


def get_dataset_config(dataset_name):
    configs = {
        "HateMM": {
            "annotation": "/data/jehc223/EMNLP2/baseline/HVGuard/datasets/HateMM/data (base).json",
            "video_dir": "/data/jehc223/HateMM/video",
            "frames_dir": "/data/jehc223/HateMM/frames",
        },
        "MultiHateClip_CN": {
            "annotation": "/data/jehc223/EMNLP2/baseline/HVGuard/datasets/Multihateclip/Chinese/data.json",
            "video_dir": "/data/jehc223/Multihateclip/Chinese/video",
            "frames_dir": "/data/jehc223/Multihateclip/Chinese/frames",
        },
        "MultiHateClip_EN": {
            "annotation": "/data/jehc223/EMNLP2/baseline/HVGuard/datasets/Multihateclip/English/data.json",
            "video_dir": "/data/jehc223/Multihateclip/English/video",
            "frames_dir": "/data/jehc223/Multihateclip/English/frames",
        },
    }
    return configs[dataset_name]


def find_video(video_dir, vid):
    for ext in [".mp4", ".webm", ".avi", ".mkv", ".mov"]:
        p = os.path.join(video_dir, f"{vid}{ext}")
        if os.path.exists(p):
            return p
    matches = glob.glob(os.path.join(video_dir, f"{vid}.*"))
    return matches[0] if matches else None


def run_inference(args):
    from vllm import LLM, SamplingParams

    dataset = args.dataset
    cfg = get_dataset_config(dataset)
    num_frames = args.num_frames
    output_dir = os.path.join(args.output_dir, dataset)
    os.makedirs(output_dir, exist_ok=True)

    ckpt_path = os.path.join(output_dir, "mllm_scores.json")
    results = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            results = json.load(f)
    print(f"Loaded {len(results)} existing results")

    with open(cfg["annotation"]) as f:
        annotations = {d["Video_ID"]: d for d in json.load(f)}

    remaining = [vid for vid in annotations if vid not in results]
    print(f"Total: {len(annotations)}, Done: {len(results)}, Remaining: {len(remaining)}")

    if not remaining:
        print("All done!")
        return

    # Init vLLM
    max_num_seqs = args.max_num_seqs
    llm = None
    while max_num_seqs >= 1:
        try:
            llm = LLM(
                model=MODEL_ID, tensor_parallel_size=1,
                max_model_len=args.max_model_len, max_num_seqs=max_num_seqs,
                gpu_memory_utilization=args.gpu_mem, trust_remote_code=True,
                limit_mm_per_prompt={"video": 1}, dtype="auto", enforce_eager=True,
            )
            print(f"LLM init OK: max_num_seqs={max_num_seqs}")
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "memory" in str(e).lower():
                max_num_seqs = max(1, max_num_seqs // 2)
                torch.cuda.empty_cache()
                gc.collect()
            else:
                raise

    if llm is None:
        sys.exit(1)

    sp = SamplingParams(max_tokens=512, temperature=0.0, top_p=1.0)
    chunk_size = max(max_num_seqs, 4)
    start = time.time()
    done = 0

    for ci in range(0, len(remaining), chunk_size):
        chunk = remaining[ci:ci + chunk_size]
        inputs, input_vids = [], []

        for vid in chunk:
            ann = annotations[vid]
            prompt_text = build_scoring_prompt(ann.get("Title", ""), ann.get("Transcript", ""))

            video_path = find_video(cfg["video_dir"], vid)
            vdata = None
            if video_path:
                try:
                    vdata = load_video_with_metadata(video_path, num_frames)
                except:
                    pass
            if vdata is None:
                fdir = os.path.join(cfg["frames_dir"], vid)
                if os.path.isdir(fdir):
                    try:
                        vdata = load_frames_fallback(fdir, num_frames)
                    except:
                        pass

            if vdata is None:
                # Text-only fallback scores
                results[vid] = {
                    "scores": {"hate_speech_score": 5, "visual_hate_score": 5,
                               "cross_modal_score": 5, "implicit_hate_score": 5,
                               "overall_hate_score": 5, "confidence": 3,
                               "classification": "unknown", "key_evidence": "no_video"},
                    "raw": "no_video_available",
                }
                continue

            prompt = (
                "<|im_start|>system\nYou are a content moderation expert. Output only valid JSON.<|im_end|>\n"
                "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>\n"
                f"{prompt_text}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            inputs.append({"prompt": prompt, "multi_modal_data": {"video": vdata}})
            input_vids.append(vid)

        if not inputs:
            done += len(chunk)
            continue

        # Batch inference
        try:
            outputs = llm.generate(inputs, sampling_params=sp)
        except:
            torch.cuda.empty_cache()
            outputs = []
            for si in inputs:
                try:
                    out = llm.generate([si], sampling_params=sp)
                    outputs.extend(out)
                except:
                    outputs.append(None)

        for i, vid in enumerate(input_vids):
            if i < len(outputs) and outputs[i] is not None:
                raw = outputs[i].outputs[0].text.strip()
                scores = parse_json_response(raw)
                results[vid] = {"scores": scores, "raw": raw}
            else:
                results[vid] = {
                    "scores": {"hate_speech_score": 5, "visual_hate_score": 5,
                               "cross_modal_score": 5, "implicit_hate_score": 5,
                               "overall_hate_score": 5, "confidence": 3,
                               "classification": "unknown", "key_evidence": "inference_failed"},
                    "raw": "ERROR",
                }

        done += len(chunk)
        elapsed = time.time() - start
        print(f"  [{dataset}] {done}/{len(remaining)} ({done/elapsed:.1f} vid/s)")

        # Checkpoint
        tmp = ckpt_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f, ensure_ascii=False)
        os.replace(tmp, ckpt_path)

    print(f"Done! {len(results)} results saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["HateMM", "MultiHateClip_CN", "MultiHateClip_EN"])
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--max_num_seqs", type=int, default=16)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--gpu_mem", type=float, default=0.92)
    parser.add_argument("--output_dir", default="/data/jehc223/EMNLP2/results/mllm")
    args = parser.parse_args()
    run_inference(args)
