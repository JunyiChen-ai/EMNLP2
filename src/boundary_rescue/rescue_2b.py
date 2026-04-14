"""
Task B: 2nd-call boundary rescue via asymmetric review prompts.

For each candidate video (from `candidates_{version}.jsonl`), the MLLM
is told what the first-pass decision was and asked to deliberate and
return a VERDICT: CONFIRM or VERDICT: OVERTURN.

Two prompts, one per side:
- `pred_baseline == 0` → FN-Hunt review (look harder for missed hate)
- `pred_baseline == 1` → FP-Hunt review (is it about hate, not producing?)

Plain-text decoding. No log-probabilities. Temperature 0. Max 512 tokens
so deliberation fits.

Output: `results/boundary_rescue/{dataset}/rescue_{version}.jsonl`
Schema: {video_id, side, pred_baseline, response_text, verdict,
         pred_after, flipped}
"""

import argparse
import json
import logging
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "our_method"))

from data_utils import get_media_path, load_annotations  # noqa: E402

PROJECT_ROOT = "/data/jehc223/EMNLP2"
OUT_ROOT = os.path.join(PROJECT_ROOT, "results", "boundary_rescue")
ALL_DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM"]


SYSTEM_REVIEW = (
    "You are a content moderator conducting a careful second-pass review. "
    "Your job is to extract observable evidence from a video and produce "
    "a structured report. Most initial assessments are correct — only "
    "overturn an initial assessment when the observable evidence clearly "
    "contradicts it."
)


# --- HateMM prompts (strict group-hate definition) ---------------------------
# HateMM labels align with the strict definition ("produces group-targeted
# hate"), so the HateMM rescue uses that definition. Failure mode on HateMM
# is FPs where videos report/critique/react to hate rather than producing it.

HATEMM_FN_HUNT_PROMPT = """Video title: {title}
Transcript: {transcript}

An initial assessment of this video produced:

    INITIAL ASSESSMENT: NOT HATEFUL

Your task is to check whether the observable evidence in the video contradicts the initial assessment. Fill in the report below using ONLY evidence you can actually observe in the video, transcript, or title. Do not speculate. If you did not observe something, write "none".

Definition (strict): a video is HATEFUL iff it itself mocks, demeans, threatens, stereotypes, or dehumanizes an identifiable protected group (race, ethnicity, religion, gender, sexuality, disability, nationality, immigration status, age, or similar).

Output your response in EXACTLY this format (one field per line):

TARGET_GROUP: <name the specific protected group being targeted in a demeaning way, or "none">
EXPLICIT_SLURS: <quote the slur(s) from the transcript, or "none">
DEHUMANIZING_LANGUAGE: <quote the dehumanizing phrase(s) from the transcript, or "none">
VISUAL_HATE_SIGNAL: <describe the specific flag/symbol/gesture, or "none">
RATIONALE: <one short sentence tying the evidence above to the verdict>
VERDICT: <OVERTURN if you found concrete demeaning treatment of a protected group — at least one non-"none" field above AND that field clearly shows group-targeted hate. CONFIRM in all other cases (including when the evidence is ambiguous, when it targets an individual rather than a group, or when it does not specifically target a protected characteristic).>"""


HATEMM_FP_HUNT_PROMPT = """Video title: {title}
Transcript: {transcript}

An initial assessment of this video produced:

    INITIAL ASSESSMENT: HATEFUL

Your task is to check whether the observable evidence in the video contradicts the initial assessment. Fill in the report below using ONLY evidence you can actually observe in the video, transcript, or title. Do not speculate. If you did not observe something, write "none".

Definition (strict): a video is HATEFUL iff it ITSELF mocks, demeans, threatens, stereotypes, or dehumanizes an identifiable protected group. A video that merely shows or discusses hate for the purpose of reporting, critiquing, parodying, educating, or reacting is NOT hateful.

Output your response in EXACTLY this format (one field per line):

OBSERVED_HATE: <describe the hateful content shown, or "none">
VIDEO_ROLE: <one of: PRODUCING (the video itself attacks a protected group), REPORTING (news / documentary about hate), CRITIQUING (condemning, parodying, or mocking hate), REACTING (commentary / analysis of hate content), UNRELATED (video is not about hate at all)>
TARGET_IS_GROUP: <YES if a specific protected group is being attacked by the video itself; NO if the target is a public figure, institution, idea, individual person, or there is no clear group target>
RATIONALE: <one short sentence tying the fields above to the verdict>
VERDICT: <CONFIRM if VIDEO_ROLE is PRODUCING AND TARGET_IS_GROUP is YES; OVERTURN otherwise>"""


# --- MHClip prompts (broader "hateful OR offensive" definition) --------------
# MHClip's labels collapse Hateful+Offensive → 1, which is broader than strict
# group-hate: it includes mockery, stereotyping, bullying, rudeness, crude
# humor, and distasteful content even when the target isn't a protected group.
# Same philosophy as HateMM (extract structured evidence, only overturn on
# clear counter-evidence), different *definition* to match MHClip labels.

MHCLIP_FN_HUNT_PROMPT = """Video title: {title}
Transcript: {transcript}

An initial assessment of this video produced:

    INITIAL ASSESSMENT: NOT OFFENSIVE

Your task is to check whether the observable evidence in the video contradicts the initial assessment. Fill in the report below using ONLY evidence you can actually observe in the video, transcript, or title. Do not speculate. If you did not observe something, write "none".

Definition (broad): a video is OFFENSIVE or HATEFUL if it contains ANY of the following — mockery, ridicule, stereotyping, dehumanization, bullying, demeaning jokes, crude insults, slurs, coded slurs, hostile or derogatory framing, targeted harassment, or distasteful humor about any person, group, category, identity, or shared characteristic. The target does NOT have to be a protected group: mocking men, mocking women, mocking marriage, mocking any subculture, or ridiculing any identifiable target all count.

Output your response in EXACTLY this format (one field per line):

OBSERVED_CONTENT: <describe the content, one sentence>
TARGET: <who or what is being mocked / ridiculed / demeaned / insulted, or "none">
OFFENSIVE_ELEMENT: <the specific phrase, joke, stereotype, or visual; or "none">
CATEGORY: <one of: MOCKERY, SLUR, STEREOTYPE, BULLYING, CRUDE_HUMOR, HOSTILITY, HATE_SYMBOL, NONE>
RATIONALE: <one short sentence tying the evidence above to the verdict>
VERDICT: <OVERTURN if CATEGORY is not NONE and OFFENSIVE_ELEMENT is not "none" (the video does contain offensive/hateful content the initial assessment missed). CONFIRM otherwise.>"""


MHCLIP_FP_HUNT_PROMPT = """Video title: {title}
Transcript: {transcript}

An initial assessment of this video produced:

    INITIAL ASSESSMENT: OFFENSIVE

Your task is to check whether the observable evidence in the video contradicts the initial assessment. Fill in the report below using ONLY evidence you can actually observe in the video, transcript, or title. Do not speculate. If you did not observe something, write "none".

Definition (broad): a video is OFFENSIVE or HATEFUL if it contains ANY of the following — mockery, ridicule, stereotyping, dehumanization, bullying, demeaning jokes, crude insults, slurs, coded slurs, hostile or derogatory framing, targeted harassment, or distasteful humor about any person, group, category, identity, or shared characteristic. The target does NOT have to be a protected group. The bar to overturn an OFFENSIVE assessment is HIGH — only overturn if the video has genuinely no hostile, mocking, demeaning, bullying, or distasteful content of any kind.

Output your response in EXACTLY this format (one field per line):

OBSERVED_CONTENT: <describe the content, one sentence>
ANY_MOCKERY_OR_INSULT: <YES if the video contains any mocking, teasing, demeaning, or insulting content (even mild); NO only if the video is fully neutral / wholesome / informational>
ANY_HOSTILITY: <YES if the video contains hostility, anger, derision, or hostile framing toward anyone or anything; NO only if the tone is fully neutral or positive>
VIDEO_ROLE: <one of: OFFENSIVE_CONTENT (the video itself mocks/insults/bullies/demeans), FACTUAL (neutral reporting, tutorial, or informational with no mockery), POSITIVE (wholesome, uplifting, or neutral entertainment)>
RATIONALE: <one short sentence tying the fields above to the verdict>
VERDICT: <OVERTURN only if ANY_MOCKERY_OR_INSULT is NO AND ANY_HOSTILITY is NO AND VIDEO_ROLE is FACTUAL or POSITIVE. CONFIRM in all other cases (including any borderline / mixed / ambiguous content).>"""


def get_prompts(dataset):
    """Return (FN_hunt, FP_hunt) prompts for a dataset."""
    if dataset == "HateMM":
        return HATEMM_FN_HUNT_PROMPT, HATEMM_FP_HUNT_PROMPT
    else:
        return MHCLIP_FN_HUNT_PROMPT, MHCLIP_FP_HUNT_PROMPT


VERDICT_RE = re.compile(
    r"VERDICT\s*[:\-]?\s*(CONFIRM|OVERTURN)",
    re.IGNORECASE,
)


def parse_verdict(text):
    """Return 'CONFIRM' / 'OVERTURN' / 'UNCLEAR'.

    Prefers the LAST occurrence (deliberation may name both words
    before the final verdict line).
    """
    if not text:
        return "UNCLEAR"
    matches = VERDICT_RE.findall(text)
    if not matches:
        return "UNCLEAR"
    return matches[-1].upper()


def build_media_content(media_path, media_type):
    if media_type == "video":
        return [{"type": "video_url", "video_url": {"url": f"file://{media_path}"}}]
    import glob as globmod
    import numpy as np

    jpgs = sorted(globmod.glob(os.path.join(media_path, "*.jpg")))
    if len(jpgs) > 8:
        indices = np.linspace(0, len(jpgs) - 1, 8, dtype=int)
        jpgs = [jpgs[i] for i in indices]
    return [{"type": "image_url", "image_url": {"url": f"file://{p}"}} for p in jpgs]


def load_candidates(dataset, version):
    path = os.path.join(OUT_ROOT, dataset, f"candidates_{version}.jsonl")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def resume_done_ids(out_path):
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


def rescue_dataset(
    dataset,
    version,
    llm,
    sampling_params,
    transcript_limit,
):
    candidates = load_candidates(dataset, version)
    annotations = load_annotations(dataset)

    out_dir = os.path.join(OUT_ROOT, dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"rescue_{version}.jsonl")

    done_ids = resume_done_ids(out_path)
    remaining = [c for c in candidates if c["video_id"] not in done_ids]
    logging.info(
        f"[{dataset}] candidates={len(candidates)} already_done={len(done_ids)} "
        f"to_run={len(remaining)}"
    )

    if not remaining:
        return out_path

    fn_prompt, fp_prompt = get_prompts(dataset)

    t0 = time.time()
    with open(out_path, "a") as f:
        for c in remaining:
            vid = c["video_id"]
            side = c["side"]
            pred_baseline = int(c["pred_baseline"])

            ann = annotations.get(vid)
            if ann is None:
                logging.warning(f"  [{dataset}] {vid}: not in annotations, skipping")
                continue
            media = get_media_path(vid, dataset)
            if media is None:
                logging.warning(f"  [{dataset}] {vid}: no media, skipping")
                continue
            media_path, media_type = media

            title = (ann.get("title", "") or "")
            transcript = (ann.get("transcript", "") or "")[:transcript_limit]

            if side == "below":
                prompt_text = fn_prompt.format(title=title, transcript=transcript)
                probe_role = "FN_Hunt"
            else:
                prompt_text = fp_prompt.format(title=title, transcript=transcript)
                probe_role = "FP_Hunt"

            media_content = build_media_content(media_path, media_type)
            content = media_content + [{"type": "text", "text": prompt_text}]
            messages = [
                {"role": "system", "content": SYSTEM_REVIEW},
                {"role": "user", "content": content},
            ]

            try:
                out = llm.chat(messages=[messages], sampling_params=sampling_params)
                response_text = out[0].outputs[0].text
                err = None
            except Exception as e:
                response_text = ""
                err = str(e)[:300]
                logging.error(f"  [{dataset}] {vid}: chat failed: {err}")

            verdict = parse_verdict(response_text)
            if verdict == "OVERTURN":
                pred_after = 1 - pred_baseline
                flipped = True
            else:
                pred_after = pred_baseline
                flipped = False

            rec = {
                "video_id": vid,
                "dataset": dataset,
                "side": side,
                "probe": probe_role,
                "pred_baseline": pred_baseline,
                "pred_after": pred_after,
                "flipped": flipped,
                "verdict": verdict,
                "response_text": response_text,
                "score": float(c["score"]),
                "threshold": float(c["threshold"]),
            }
            if err:
                rec["error"] = err
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

            elapsed = time.time() - t0
            logging.info(
                f"  [{dataset}] {vid} ({side}) verdict={verdict} flipped={flipped} "
                f"({elapsed:.1f}s)"
            )

    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="S1_v1")
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--transcript-limit", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
    )
    args = parser.parse_args()

    if not args.all and not args.dataset:
        parser.error("Provide --dataset or --all")

    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(log_dir, f"boundary_rescue_{args.version}.log")
            ),
            logging.StreamHandler(),
        ],
    )

    datasets = ALL_DATASETS if args.all else [args.dataset]
    logging.info(f"Boundary rescue: version={args.version} datasets={datasets}")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=0.92,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"video": 1, "image": 8},
        allowed_local_media_path="/data/jehc223",
        mm_processor_kwargs={"max_pixels": 100352},
    )
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
    )

    for ds in datasets:
        rescue_dataset(
            ds, args.version, llm, sampling_params, args.transcript_limit
        )

    logging.info("All datasets done.")


if __name__ == "__main__":
    main()
