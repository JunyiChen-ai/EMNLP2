"""
Faithful reproduction of MARS (Multi-stage Adversarial ReaSoning) using
Qwen3-VL-2B-Instruct via vLLM.

Paper: "Training-Free and Interpretable Hateful Video Detection via
Multi-stage Adversarial Reasoning", Multimodal Intelligence Lab (MIL),
University of Exeter, Jan 2026. arxiv: 2601.15115
Repo:  https://github.com/Multimodal-Intelligence-Lab-MIL/MARS
Commit of our reference: 72e1618943d36edb26ba0a24d0b4b417978d38e6

Faithfulness scope
  Verbatim: all 4 stage prompt strings, the `\\n\\nTranscript: ...` append
  rule, the ```json fence-strip + json.loads parser, the 16-frame uniform
  sampling, the stochastic decoding (temperature=0.7, top_p=0.9,
  max_new_tokens=4096), and the final-label source
  (`final_decision.label`).

  Adapted to our infra: Qwen3-VL-2B-Instruct backbone (paper used
  Qwen2.5-VL 32B / Llama4 / GPT5-mini / Gemini2.5-Flash); inference through
  vLLM instead of HuggingFace transformers; images loaded by URL instead
  of PIL; Slurm-friendly per-video resume.

Prompts below are VERBATIM from
third_party/MARS/code/MARS/Qwen.py@72e1618943d36edb26ba0a24d0b4b417978d38e6
lines 137-219. Do not paraphrase — "faithful" means byte-exact.
"""

import argparse
import base64
import glob
import io
import json
import logging
import os
import re
import sys
import time
from typing import Dict, List, Optional

import numpy as np

_OUR_METHOD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "our_method")
sys.path.insert(0, _OUR_METHOD)
from data_utils import DATASET_ROOTS, get_media_path, load_annotations  # noqa: E402

PROJECT_ROOT = "/data/jehc223/EMNLP2"
ALL_DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM"]
NUM_FRAMES = 16  # MARS default, paper says stable at 8/16/32


# ============================================================================
# VERBATIM PROMPTS FROM MARS/code/MARS/Qwen.py (commit 72e1618)
# Do not edit. Faithfulness depends on byte-exactness.
# ============================================================================

MARS_TURN_1 = """You are a content analysis expert. Provide a detailed, objective description of this video content.

Task: Describe what you observe in the video frames and transcript:
1. objective_visual_description: Describe what you see - objects, people, actions, settings, text, symbols, colors, expressions, gestures, etc.

Rules:
- Be comprehensive and detailed
- NO analysis, interpretations, or value judgments
- Simply report what you observe
- Return ONLY valid JSON with one key: ["objective_visual_description"]"""

MARS_TURN_2 = """You are a hateful content detection specialist. Analyze the video and transcript content under the ASSUMPTION that it contains hateful or offensive content.

HATE SPEECH DEFINITION: Any communication that attacks or uses pejorative/discriminatory language targeting a person or group based on their religion, ethnicity, nationality, race, colour, descent, gender or other identity factor.
OFFENSIVE CONTENT: Contains hurtful, derogatory, or socially inappropriate content, even if it does not constitute direct hate speech.
IMPORTANT: Simple conflicts, arguments, insults, or violent behaviour between individuals may not constitute hate speech.

Task: Identify evidence supporting hateful(offensive) interpretation:
1. evidence: Describe specific evidence supporting the hateful nature of the content
2. reasoning: Explain why it is hateful content.
3. strength: Your confidence in this hate interpretation [0-1]

Focus on:
- Language targeting religion, ethnicity, race, gender, nationality, etc.
- Group-based dehumanizing language or symbols
- Identity-based discriminatory attacks (not personal insults)
- Incitement against identity groups

Return ONLY valid JSON with keys: ["evidence", "reasoning", "strength"]"""

MARS_TURN_3 = """You are a content analyst. Analyze this content under the ASSUMPTION that it does NOT constitute hate speech and offensive content.

HATE SPEECH DEFINITION: Any communication that attacks or uses pejorative/discriminatory language targeting a person or group based on their religion, ethnicity, nationality, race, colour, descent, gender or other identity factor.
OFFENSIVE CONTENT: Contains hurtful, derogatory, or socially inappropriate content, even if it does not constitute direct hate speech.
IMPORTANT: Simple conflicts, arguments, insults, or violent behaviour between individuals may not constitute hate speech.

Task: Identify evidence supporting non-hate(non-offensive) interpretation:
1. evidence: Describe specific evidence supporting the non-hateful nature of the content
2. reasoning: Explain why this content does not hateful
3. strength: Your confidence in this non-hate interpretation [0-1]

Consider:
- Is this a personal dispute rather than group targeting?
- Are insults directed at individuals rather than identity groups?
- Is there artistic, satirical, or educational context?
- Does the content lack group-based discriminatory language?

Return ONLY valid JSON with keys: ["evidence", "reasoning", "strength"]"""


def build_turn_4_prompt(turn_1_parsed, turn_2_parsed, turn_3_parsed) -> str:
    """Stage 4 prompt: verbatim from MARS Qwen.py:194-219. Uses f-string to
    embed the prior stages' parsed responses via json.dumps, matching MARS."""
    return f"""You are a senior content moderation specialist making the final determination.

OBJECTIVE DESCRIPTION: {json.dumps(turn_1_parsed or {}, ensure_ascii=False)}

HATE ARGUMENT: {json.dumps(turn_2_parsed or {}, ensure_ascii=False)}

NON-HATE ARGUMENT: {json.dumps(turn_3_parsed or {}, ensure_ascii=False)}

Task: Conduct critical analysis and make final determination:

1. evidence_comparison: Compare the quality and strength of both arguments, identifying which evidence is more compelling
2. contextual_analysis: Analyze how the content's context (platform, audience, intent) affects interpretation
3. harm_assessment: Evaluate potential real-world impact and harm to targeted groups
4. final_decision:
   - label: 0 (non-hate) or 1 (hate/offensive)
   - confidence: Overall confidence in decision [0-1]
   - key_factors: The decisive elements that determined your judgment
   - reasoning: 2-3 sentences explaining your decision

Rules:
- Weigh evidence objectively, not just confidence scores
- Consider both explicit and subtle indicators
- Prioritize potential for real-world harm
- Base decision on strongest evidence, not balanced arguments

Return ONLY valid JSON with key "final_decision" containing the above structure"""


def append_transcript(base_prompt: str, transcript: str) -> str:
    """MARS Qwen.py:224-228. Verbatim appending rule."""
    if (transcript or "").strip():
        return base_prompt + f"\n\nTranscript: {transcript}"
    return base_prompt + "\n\nTranscript: No transcript available."


def clean_json_response(response_text: str, default_key: Optional[str] = None):
    """Parse MARS stage output as JSON, with fallbacks for 2B-scale models.

    Primary path: MARS Qwen.py:316-333. Strip ```json fences, then json.loads.

    Fallbacks (added for 2B-downgraded reproduction — NOT in original MARS):
      F1: `[...]` → `{...}`. Our 2B model frequently emits
          `["key": value]` because the prompt says
          `one key: ["objective_visual_description"]` — 2B reads the
          square brackets in the format hint as the literal output
          delimiter. Swap `[` / `]` → `{` / `}` and retry.
      F2: first balanced `{...}` block via brace-depth scan.
      F3: first balanced `[...]` block, then apply F1.
      F4: if `default_key` given, wrap the entire cleaned text as
          `{default_key: <raw text>}`. Last-ditch: never leave the model's
          content on the floor just because it forgot the braces.
    """
    cleaned = (response_text or "").strip()
    if not cleaned:
        if default_key is not None:
            return {default_key: ""}, True
        return response_text, False

    # Strip markdown fences
    fenced = cleaned
    if cleaned.startswith("```json") and cleaned.endswith("```"):
        fenced = cleaned[7:-3].strip()
    elif cleaned.startswith("```") and cleaned.endswith("```"):
        fenced = cleaned[3:-3].strip()

    # Primary parse
    try:
        return json.loads(fenced), True
    except json.JSONDecodeError:
        pass

    # F1: outer [] → {}
    if fenced.startswith("[") and fenced.endswith("]"):
        try:
            fixed = "{" + fenced[1:-1].strip() + "}"
            return json.loads(fixed), True
        except json.JSONDecodeError:
            pass

    # F2: balanced {...} scan
    start = fenced.find("{")
    if start != -1:
        depth = 0
        end = -1
        for i in range(start, len(fenced)):
            c = fenced[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end != -1:
            try:
                return json.loads(fenced[start:end]), True
            except json.JSONDecodeError:
                pass

    # F3: balanced [...] scan then swap
    start = fenced.find("[")
    if start != -1:
        depth = 0
        end = -1
        for i in range(start, len(fenced)):
            c = fenced[i]
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end != -1:
            chunk = fenced[start:end]
            try:
                fixed = "{" + chunk[1:-1].strip() + "}"
                return json.loads(fixed), True
            except json.JSONDecodeError:
                pass

    # F4: last-ditch — wrap raw text as {default_key: text}
    if default_key is not None:
        return {default_key: fenced}, True

    return response_text, False


def extract_stage4_label_from_text(raw_text: str) -> Optional[int]:
    """Last-ditch Stage 4 label extraction from free text.

    Tries several regex patterns in order of specificity. Returns 0, 1, or
    None if nothing resembles a verdict.
    """
    if not raw_text:
        return None
    t = raw_text.lower()

    # Pattern 1: `"label": 1` / `"label": 0` / `'label': 1` etc.
    m = re.search(r'["\']?label["\']?\s*[:=]\s*["\']?([01])["\']?', t)
    if m:
        return int(m.group(1))

    # Pattern 2: `label is 1` / `label = 0`
    m = re.search(r'\blabel\s+(?:is|=)\s*([01])\b', t)
    if m:
        return int(m.group(1))

    # Pattern 3: `final_decision.*label.*[01]` (looser)
    m = re.search(r'final[_\s]*decision.*?label[^\d]{0,10}([01])', t, re.DOTALL)
    if m:
        return int(m.group(1))

    # Pattern 4: explicit verdict words near the end
    tail = t[-500:]  # last few hundred chars
    if re.search(r'\bhate(ful)?\b', tail) and not re.search(r'\bnot\s+hate(ful)?\b', tail):
        if not re.search(r'\bnon[-_\s]?hate(ful)?\b', tail):
            return 1
    if re.search(r'\b(not\s+hate|non[-_\s]?hate|non[-_\s]?offensive|does\s+not\s+constitute)', tail):
        return 0

    return None


# ============================================================================
# Infrastructure (our vLLM + data_utils plumbing)
# ============================================================================

def sample_frames(media_path: str, media_type: str, num_frames: int = NUM_FRAMES):
    """Collect up to num_frames image URLs uniformly sampled, matching
    MARS.sample_frames which uses step = total_frames / num_frames indexing."""
    if media_type == "video":
        # MARS uses pre-extracted frame folders; when the source is an mp4,
        # we rely on vLLM's internal video-url handling. See the main loop
        # for which path is taken per dataset.
        return [{"type": "video_url", "video_url": {"url": f"file://{media_path}"}}]
    jpgs = sorted(glob.glob(os.path.join(media_path, "*.jpg")) +
                  glob.glob(os.path.join(media_path, "*.jpeg")) +
                  glob.glob(os.path.join(media_path, "*.png")))
    if not jpgs:
        return []
    if len(jpgs) <= num_frames:
        selected = jpgs
    else:
        step = len(jpgs) / num_frames
        indices = [int(i * step) for i in range(num_frames)]
        selected = [jpgs[i] for i in indices]
    return [{"type": "image_url", "image_url": {"url": f"file://{p}"}} for p in selected]


def build_user_message(media_content, prompt_text: str):
    """Single-turn user message. MARS uses a fresh conversation per stage,
    so every call is a standalone user turn with images + text."""
    return [{
        "role": "user",
        "content": media_content + [{"type": "text", "text": prompt_text}],
    }]


def extract_final_label(stage4_parsed) -> Optional[int]:
    """Pull final_decision.label from MARS stage 4 output. Paper says the
    meta-prompt directly emits the verdict; label ∈ {0, 1}."""
    if not isinstance(stage4_parsed, dict):
        return None
    fd = stage4_parsed.get("final_decision")
    if not isinstance(fd, dict):
        # Some outputs may put the keys at the top level if the model ignores
        # the wrapper. Try a fallback.
        if "label" in stage4_parsed:
            fd = stage4_parsed
        else:
            return None
    label = fd.get("label")
    try:
        label_int = int(label)
        return 1 if label_int == 1 else 0 if label_int == 0 else None
    except (TypeError, ValueError):
        return None


def resume_done_ids(out_path: str):
    done = set()
    if os.path.exists(out_path):
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


def load_split_ids(dataset: str, split: str):
    root = DATASET_ROOTS[dataset]
    split_path = os.path.join(root, "splits", f"{split}_clean.csv")
    if not os.path.isfile(split_path):
        from data_utils import generate_clean_splits
        generate_clean_splits(dataset)
    with open(split_path) as f:
        return [line.strip() for line in f if line.strip()]


def mars_one_video(vid: str, transcript: str, media_content, llm, sampling_params) -> Dict:
    """Run the 4-stage MARS pipeline on a single video. Each stage is an
    independent vLLM call with the same images + a fresh user prompt.

    Each clean_json_response call passes a stage-specific `default_key` so
    that F4 (last-ditch text-as-value wrap) produces a usable structure.
    """
    turn_outputs = {}

    # --- Stage 1: objective description
    p1 = append_transcript(MARS_TURN_1, transcript)
    msg1 = build_user_message(media_content, p1)
    out1 = llm.chat(messages=[msg1], sampling_params=sampling_params)
    raw1 = out1[0].outputs[0].text
    parsed1, ok1 = clean_json_response(raw1, default_key="objective_visual_description")
    turn_outputs["turn_1"] = {"raw": raw1, "parsed": parsed1 if ok1 else None, "parse_ok": ok1}

    # --- Stage 2: hate hypothesis (3 keys; wrap plain text as `evidence`)
    p2 = append_transcript(MARS_TURN_2, transcript)
    msg2 = build_user_message(media_content, p2)
    out2 = llm.chat(messages=[msg2], sampling_params=sampling_params)
    raw2 = out2[0].outputs[0].text
    parsed2, ok2 = clean_json_response(raw2, default_key="evidence")
    turn_outputs["turn_2"] = {"raw": raw2, "parsed": parsed2 if ok2 else None, "parse_ok": ok2}

    # --- Stage 3: non-hate hypothesis (3 keys; wrap plain text as `evidence`)
    p3 = append_transcript(MARS_TURN_3, transcript)
    msg3 = build_user_message(media_content, p3)
    out3 = llm.chat(messages=[msg3], sampling_params=sampling_params)
    raw3 = out3[0].outputs[0].text
    parsed3, ok3 = clean_json_response(raw3, default_key="evidence")
    turn_outputs["turn_3"] = {"raw": raw3, "parsed": parsed3 if ok3 else None, "parse_ok": ok3}

    # --- Stage 4 (meta-synthesis → final label)
    p4_base = build_turn_4_prompt(
        turn_outputs["turn_1"]["parsed"],
        turn_outputs["turn_2"]["parsed"],
        turn_outputs["turn_3"]["parsed"],
    )
    p4 = append_transcript(p4_base, transcript)
    msg4 = build_user_message(media_content, p4)
    out4 = llm.chat(messages=[msg4], sampling_params=sampling_params)
    raw4 = out4[0].outputs[0].text
    # Stage 4 is special: F4 doesn't help because we need an integer label,
    # not wrap-as-value. So try the JSON path first; if that fails, fall
    # back to a regex extractor over the raw text.
    parsed4, ok4 = clean_json_response(raw4)
    turn_outputs["turn_4"] = {"raw": raw4, "parsed": parsed4 if ok4 else None, "parse_ok": ok4}

    pred = extract_final_label(parsed4) if ok4 else None
    if pred is None:
        # Last-ditch: regex over raw stage 4 text
        pred = extract_stage4_label_from_text(raw4)
    rec = {
        "video_id": vid,
        "turn_1_raw": raw1,
        "turn_2_raw": raw2,
        "turn_3_raw": raw3,
        "turn_4_raw": raw4,
        "turn_1_parsed": turn_outputs["turn_1"]["parsed"],
        "turn_2_parsed": turn_outputs["turn_2"]["parsed"],
        "turn_3_parsed": turn_outputs["turn_3"]["parsed"],
        "turn_4_parsed": turn_outputs["turn_4"]["parsed"],
        "parse_ok": {
            "turn_1": ok1, "turn_2": ok2, "turn_3": ok3, "turn_4": ok4,
        },
        "pred": pred if pred is not None else -1,
    }
    return rec


def score_dataset(dataset: str, split: str, llm, sampling_params):
    annotations = load_annotations(dataset)
    split_ids = load_split_ids(dataset, split)
    logging.info(f"[{dataset}] {len(split_ids)} videos in {split}_clean")

    out_dir = os.path.join(PROJECT_ROOT, "results", "mars_2b", dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{split}_mars.jsonl")

    done_ids = resume_done_ids(out_path)
    if done_ids:
        logging.info(f"[{dataset}] Resume: {len(done_ids)} already done")
    remaining = [v for v in split_ids if v not in done_ids]
    if not remaining:
        logging.info(f"[{dataset}] all done")
        return out_path

    t0 = time.time()
    n_processed = 0
    n_skipped = 0

    with open(out_path, "a") as f:
        for vid in remaining:
            ann = annotations.get(vid)
            if ann is None:
                logging.warning(f"  {vid}: not in annotations, skipping")
                n_skipped += 1
                continue
            media = get_media_path(vid, dataset)
            if media is None:
                logging.warning(f"  {vid}: no media, skipping")
                n_skipped += 1
                continue
            media_path, media_type = media
            transcript = ann.get("transcript", "") or ""

            media_content = sample_frames(media_path, media_type, NUM_FRAMES)
            if not media_content:
                logging.warning(f"  {vid}: no frames, skipping")
                n_skipped += 1
                continue

            try:
                rec = mars_one_video(vid, transcript, media_content, llm, sampling_params)
            except Exception as e:
                err = str(e)
                logging.error(f"  {vid}: MARS pipeline failed: {err[:200]}")
                rec = {"video_id": vid, "pred": -1, "error": err[:500]}
                n_skipped += 1

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            n_processed += 1

            elapsed = time.time() - t0
            rate = n_processed / elapsed if elapsed > 0 else 0
            if n_processed % 5 == 0 or n_processed == 1:
                logging.info(f"  [{dataset}] [{len(done_ids)+n_processed}/{len(split_ids)}] "
                             f"{rate:.2f} vid/s, pred={rec.get('pred')}")

    logging.info(f"[{dataset}] done. {n_processed} scored, {n_skipped} skipped.")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Faithful MARS reproduction (Qwen3-VL-2B)")
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--seed", type=int, default=42,
                        help="vLLM sampling seed (MARS uses temperature=0.7 sampling; "
                             "seed gives us run-to-run reproducibility)")
    args = parser.parse_args()

    if not args.all and not args.dataset:
        parser.error("Provide --dataset or --all")

    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "mars_2b.log")),
            logging.StreamHandler(),
        ],
    )

    datasets = ALL_DATASETS if args.all else [args.dataset]
    logging.info(f"MARS reproduction: datasets={datasets} split={args.split} "
                 f"model={args.model} seed={args.seed} frames={NUM_FRAMES}")

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=0.92,
        max_model_len=40960,
        limit_mm_per_prompt={"image": NUM_FRAMES, "video": 1},
        allowed_local_media_path="/data/jehc223",
        mm_processor_kwargs={"max_pixels": 100352},
        seed=args.seed,
    )

    # MARS Qwen.py:282-290: max_new_tokens=4096, do_sample=True,
    # temperature=0.7, top_p=0.9. Verbatim.
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=4096,
        seed=args.seed,
    )

    for ds in datasets:
        score_dataset(ds, args.split, llm, sampling_params)

    logging.info("All datasets done.")


if __name__ == "__main__":
    main()
