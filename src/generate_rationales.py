"""
Generate structured rationales for hateful video detection via vLLM.

Supports sharded inference: split data across N processes (1 GPU each),
avoiding vLLM's dp mode which may hang. OOM-aware: starts with max batch
size and halves on failure.

Usage (single process):
  python src/generate_rationales.py --dataset HateMM --prompt-family diagnostic

Usage (sharded, called from wrapper):
  CUDA_VISIBLE_DEVICES=0 python src/generate_rationales.py --dataset HateMM --prompt-family diagnostic --shard-id 0 --num-shards 2
  CUDA_VISIBLE_DEVICES=1 python src/generate_rationales.py --dataset HateMM --prompt-family diagnostic --shard-id 1 --num-shards 2
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── dataset configs ──────────────────────────────────────────────────
DATASET_CONFIGS = {
    "HateMM": {
        "annotation": "datasets/HateMM/annotation(new).json",
        "video_dir": "datasets/HateMM/video",
        "video_ext": ".mp4",
        "id_field": "Video_ID",
        "title_field": "Title",
        "transcript_field": "Transcript",
        "label_field": "Label",
        "label_map": {"Hate": "hateful", "Non Hate": "normal"},
    },
    "MHClip_EN": {
        "annotation": "datasets/MHClip_EN/annotation(new).json",
        "video_dir": "datasets/MHClip_EN/video",
        "video_ext": ".webm",
        "id_field": "Video_ID",
        "title_field": "Title",
        "transcript_field": "Transcript",
        "label_field": "Label",
        "label_map": {"Hateful": "hateful", "Offensive": "hateful", "Normal": "normal"},
    },
    "MHClip_ZH": {
        "annotation": "datasets/MHClip_ZH/annotation(new).json",
        "video_dir": "datasets/MHClip_ZH/video",
        "video_ext": ".webm",
        "id_field": "Video_ID",
        "title_field": "Title",
        "transcript_field": "Transcript",
        "label_field": "Label",
        "label_map": {"Hateful": "hateful", "Offensive": "hateful", "Normal": "normal"},
    },
}

# ═══════════════════════════════════════════════════════════════════════
# PROMPT FAMILY A: HVGuard-style Generic CoT
# ═══════════════════════════════════════════════════════════════════════
HVGUARD_SYSTEM = """You are a content moderation specialist. Your task is to identify any instances of hate speech, violent imagery, discriminatory actions, or any other content that could be considered harmful, abusive, or offensive."""

HVGUARD_USER = """This is a video that may contain harmful content.

Title: {title}
Transcript: {transcript}

Step 1 - Visual Content: Describe the video content. Pay attention to any special characters, symbols, or scenes.
Step 2 - Textual Meaning: Analyze the meaning of the title and transcript. Note any homophonic memes, puns, slang, or coded language.
Step 3 - Combined Analysis: Combine visual, textual, and audio elements to detect and flag any hateful content. Answer the implicit meanings and whether this video expresses hateful content.

Output format:
[Visual Description]: <your description>
[Textual Analysis]: <your analysis>
[Implicit Meanings]: <your analysis of implicit meanings>
[Hate or Not]: <Hateful or Normal>
[Reasoning]: <2-3 sentence explanation>"""

# ═══════════════════════════════════════════════════════════════════════
# PROMPT FAMILY B: MARS/RAMF-style Adversarial
# ═══════════════════════════════════════════════════════════════════════
MARS_SYSTEM = """You are an expert content analyst specializing in online media moderation. You will analyze video content through multiple perspectives to provide a thorough and balanced assessment."""

MARS_USER = """Analyze this video through three stages:

Title: {title}
Transcript: {transcript}

**Stage 1 - Objective Description**: Describe what you observe in the video, including visual content, speech, text overlays, and audio elements. Be comprehensive and detailed. NO analysis, interpretations, or value judgments. Simply report what you observe.

**Stage 2 - Hate Hypothesis**: Under the ASSUMPTION that this content IS hateful/offensive, identify all evidence supporting this interpretation. Consider: language targeting identity groups, dehumanizing imagery/language, threats, stereotypes, discriminatory content.

**Stage 3 - Non-Hate Hypothesis**: Under the ASSUMPTION that this content is NOT hateful, identify evidence supporting a benign interpretation. Consider: personal disputes (not group targeting), artistic/satirical/educational context, quoting/reporting, cultural expression.

Output as JSON:
{{
  "objective_description": "<factual description>",
  "hate_evidence": "<evidence supporting hate interpretation>",
  "hate_reasoning": "<reasoning for hate hypothesis>",
  "nonhate_evidence": "<evidence supporting non-hate interpretation>",
  "nonhate_reasoning": "<reasoning for non-hate hypothesis>",
  "final_verdict": "<hateful or normal>",
  "confidence": "<low/medium/high>",
  "key_factors": "<what tips the balance>"
}}"""

# ═══════════════════════════════════════════════════════════════════════
# PROMPT FAMILY C: Compact Diagnostic Prompt (Evidence-to-Implication)
#
# Design principle: Minimal evidence-to-implication decomposition.
# Instead of exhaustive field enumeration, focus on 4 diagnostic
# sections (6 fields total) that map directly to known failure modes.
#
# Inspiration sources:
#   - VERA (CVPR 2025): decompose subtle concepts into focused guiding
#     questions targeting recognizable diagnostic patterns, not diffuse
#     narration. Motivates evidence_timeline + salient_cue.
#   - Social Bias Frames (Sap et al., ACL 2020): target identification
#     and implied social meaning are central to hate detection.
#     Motivates target + implied_message.
#   - Latent Hatred (ElSherief et al., EMNLP 2021): implicit hate
#     requires surfacing what is suggested, not just what is said.
#     Motivates implied_message.
#   - DEFAME (ICML 2025): explicit evidence checking before verdict
#     reduces hallucinated judgments. Motivates benign_alternative.
#   - DisCLIP (AAAI 2025): description quality > quantity for
#     downstream classification. Motivates compact design.
# ═══════════════════════════════════════════════════════════════════════
DIAGNOSTIC_SYSTEM = """You are a forensic content analyst. Your job is to extract the specific evidence that determines whether video content is hateful. Be precise and grounded. Separate observation from interpretation."""

DIAGNOSTIC_USER = """Analyze this video for hateful content.

Title: {title}
Transcript: {transcript}

**1. OBSERVED EVIDENCE**
- Evidence timeline: List 2-4 key observations in temporal order. Quote on-screen text or speech directly. Ground each observation in what you see/hear.
- Salient cue: The single most diagnostic piece of evidence for a hate judgment.

**2. SOCIAL FRAME**
- Target: Who or what group is referenced or attacked? (specific group / individual / none)
- Implied message: What negative stereotype, devaluation, threat, or exclusion is implied? If none, state "none".

**3. AMBIGUITY CHECK**
- Benign alternative: What is the strongest non-hateful interpretation? What specific evidence weakens a hate judgment?

**4. DECISION**
- Verdict: hateful or normal
- Decision basis: One sentence tying verdict to evidence, target, and implication."""


def get_prompt(prompt_family, title, transcript):
    """Return (system_msg, user_msg) for the given prompt family."""
    if prompt_family == "hvguard":
        return HVGUARD_SYSTEM, HVGUARD_USER.format(title=title, transcript=transcript)
    elif prompt_family == "mars":
        return MARS_SYSTEM, MARS_USER.format(title=title, transcript=transcript)
    elif prompt_family == "diagnostic":
        return DIAGNOSTIC_SYSTEM, DIAGNOSTIC_USER.format(title=title, transcript=transcript)
    else:
        raise ValueError(f"Unknown prompt family: {prompt_family}")


def load_dataset(dataset_name: str, project_root: str):
    cfg = DATASET_CONFIGS[dataset_name]
    ann_path = os.path.join(project_root, cfg["annotation"])
    with open(ann_path) as f:
        annotations = json.load(f)
    return annotations, cfg


def build_messages(sample, cfg, prompt_family, project_root):
    """Build chat messages for one sample."""
    vid_id = sample[cfg["id_field"]]
    title = sample.get(cfg["title_field"], "") or ""
    transcript = sample.get(cfg["transcript_field"], "") or ""

    # Truncate to keep total prompt within max_model_len
    # Video tokens ~1000-4000, system+user prompt ~500 tokens, max_tokens=1500
    # So transcript budget: ~8192 - 4000 - 500 - 1500 = ~2192 tokens ≈ 1000 chars
    if len(transcript) > 1000:
        transcript = transcript[:1000] + "..."

    system_msg, user_msg = get_prompt(prompt_family, title, transcript)
    messages = [{"role": "system", "content": system_msg}]

    # Build user content with video
    video_path = os.path.join(
        project_root, cfg["video_dir"], vid_id + cfg["video_ext"]
    )
    user_content = []
    if os.path.exists(video_path):
        user_content.append({
            "type": "video_url",
            "video_url": {"url": f"file://{video_path}"},
        })
    else:
        user_content.append({
            "type": "text",
            "text": "[Video file not available]",
        })
    user_content.append({
        "type": "text",
        "text": user_msg,
    })
    messages.append({"role": "user", "content": user_content})
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--prompt-family", required=True, choices=["hvguard", "mars", "diagnostic"])
    parser.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--batch-size", type=int, default=8, help="Starting batch size (auto-halves on OOM)")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--project-root", default="/data/jehc223/EMNLP2")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index (0-based)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples per run (0=all). For chunked processing to avoid cluster timeouts.")
    args = parser.parse_args()

    project_root = args.project_root
    if args.output_dir is None:
        args.output_dir = os.path.join(
            project_root, "rationales", args.dataset, args.prompt_family
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    annotations, cfg = load_dataset(args.dataset, project_root)
    print(f"Loaded {len(annotations)} samples from {args.dataset}")

    # ── Sharding: split data across processes ────────────────────────
    if args.num_shards > 1:
        # Deterministic split by index
        annotations = [a for i, a in enumerate(annotations) if i % args.num_shards == args.shard_id]
        print(f"Shard {args.shard_id}/{args.num_shards}: {len(annotations)} samples")

    # Check ALL shard output files for resume (so shards don't redo each other's work)
    existing_ids = set()
    for shard_idx in range(args.num_shards):
        suffix = f"_shard{shard_idx}" if args.num_shards > 1 else ""
        check_path = os.path.join(args.output_dir, f"rationales{suffix}.jsonl")
        if os.path.exists(check_path):
            with open(check_path) as f:
                for line in f:
                    if line.strip():
                        try:
                            obj = json.loads(line)
                            existing_ids.add(obj["video_id"])
                        except json.JSONDecodeError:
                            pass
    # Also check merged file
    merged_path = os.path.join(args.output_dir, "rationales.jsonl")
    if os.path.exists(merged_path):
        with open(merged_path) as f:
            for line in f:
                if line.strip():
                    try:
                        existing_ids.add(json.loads(line)["video_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    if existing_ids:
        print(f"Resuming: {len(existing_ids)} already done across all shards")

    # Filter out already processed
    todo = [s for s in annotations if s[cfg["id_field"]] not in existing_ids]
    if not todo:
        print("All samples already processed!")
        return
    # ── Chunk limit (for clusters that kill long jobs) ─────────
    if args.max_samples > 0 and len(todo) > args.max_samples:
        todo = todo[:args.max_samples]
        print(f"Chunked to {args.max_samples} samples (resume to continue)")

    print(f"Remaining: {len(todo)} samples (prompt: {args.prompt_family})")

    # ── Output file for this shard ───────────────────────────────────
    shard_suffix = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    out_path = os.path.join(args.output_dir, f"rationales{shard_suffix}.jsonl")

    # Import vLLM
    from vllm import LLM, SamplingParams

    # GPU memory budget: tp=1 on A100-80GB is tight for 32B model.
    # Use 0.95 utilization to maximize KV cache headroom.
    gpu_util = 0.95
    # Reduce max_model_len to fit within available KV cache
    max_model_len = 8192

    print(f"Loading {args.model} with TP={args.tp} (shard {args.shard_id}), gpu_util={gpu_util}, max_len={max_model_len}...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_util,
        allowed_local_media_path="/data/jehc223",
        limit_mm_per_prompt={"video": 1},
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )

    # ── OOM-aware batch processing ───────────────────────────────────
    mm_kwargs = {"max_pixels": 360 * 420 * 4}  # limit video frame pixels
    current_batch_size = args.batch_size
    t_start = time.time()
    total_done = 0
    idx = 0

    while idx < len(todo):
        batch = todo[idx : idx + current_batch_size]

        conversations = []
        batch_ids = []
        batch_labels = []
        for sample in batch:
            vid_id = sample[cfg["id_field"]]
            label = cfg["label_map"].get(sample[cfg["label_field"]], sample[cfg["label_field"]])
            messages = build_messages(sample, cfg, args.prompt_family, project_root)
            conversations.append(messages)
            batch_ids.append(vid_id)
            batch_labels.append(label)

        # Try batch, halve on OOM
        results = None
        try:
            outputs = llm.chat(
                messages=conversations,
                sampling_params=sampling_params,
                mm_processor_kwargs=mm_kwargs,
            )
            results = [(o.outputs[0].text, None) for o in outputs]
        except Exception as e:
            err_str = str(e).lower()
            if "out of memory" in err_str or "oom" in err_str or "cuda" in err_str:
                if current_batch_size > 1:
                    old_bs = current_batch_size
                    current_batch_size = max(1, current_batch_size // 2)
                    print(f"OOM with batch_size={old_bs}, reducing to {current_batch_size}. Retrying...")
                    try:
                        import torch; torch.cuda.empty_cache()
                    except Exception:
                        pass
                    continue  # retry with smaller batch (don't advance idx)
                else:
                    # batch_size=1 still fails — try reducing pixels
                    print(f"OOM even at batch_size=1, reducing max_pixels. Error: {e}")
                    mm_kwargs["max_pixels"] = max(160 * 200, mm_kwargs["max_pixels"] // 2)
                    continue
            else:
                # Non-OOM error: try individual samples
                print(f"Batch error: {e}. Falling back to individual processing.")
                results = []
                for conv in conversations:
                    try:
                        single_out = llm.chat(
                            messages=[conv],
                            sampling_params=sampling_params,
                            mm_processor_kwargs=mm_kwargs,
                        )
                        results.append((single_out[0].outputs[0].text, None))
                    except Exception as e2:
                        results.append((None, str(e2)))

        if results is None:
            continue  # retrying with smaller batch

        # Save results
        with open(out_path, "a") as f:
            for (text, err), vid_id, label in zip(results, batch_ids, batch_labels):
                record = {
                    "video_id": vid_id,
                    "label": label,
                    "prompt_family": args.prompt_family,
                    "dataset": args.dataset,
                }
                if err:
                    record["rationale_raw"] = f"ERROR: {err}"
                    record["parse_success"] = False
                else:
                    record["rationale_raw"] = text
                    if args.prompt_family == "mars":
                        try:
                            cleaned = text.strip()
                            if cleaned.startswith("```"):
                                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
                            if cleaned.endswith("```"):
                                cleaned = cleaned[:-3]
                            record["rationale_parsed"] = json.loads(cleaned.strip())
                            record["parse_success"] = True
                        except (json.JSONDecodeError, IndexError):
                            record["parse_success"] = False
                    else:
                        record["parse_success"] = True
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        total_done += len(batch)
        idx += len(batch)
        elapsed = time.time() - t_start
        rate = total_done / elapsed if elapsed > 0 else 0
        eta = (len(todo) - total_done) / rate if rate > 0 else 0
        print(f"  Shard{args.shard_id} [{total_done}/{len(todo)}] bs={current_batch_size} {rate:.2f} samples/s, ETA: {eta/60:.0f}min")

    print(f"Done! {total_done} samples processed. Results: {out_path}")


if __name__ == "__main__":
    main()
