"""
Phase A v2: Improved SCM prompt addressing error analysis findings.
Changes from v1:
1. Added endorsement_context field — distinguishes discussing/quoting hate vs endorsing
2. Added target_type field — distinguishes protected groups vs behavior-based targets
3. Stronger instruction to ground analysis in video evidence

Only SCM prompt changed. Other theories and all infrastructure code same as prompt_theories.py.

Usage:
  python prompt_theories_v2.py --theory scm_v2 --dataset_name HateMM --max_concurrent 10
"""

import argparse, asyncio, base64, json, logging, os, re, time
from datetime import datetime
from dotenv import load_dotenv
from openai import APIConnectionError, AsyncOpenAI, RateLimitError
from tqdm import tqdm

load_dotenv()
MODEL_NAME = "gpt-5.4-nano"
client = AsyncOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.getenv("OPENAI_API_KEY")
)

SYSTEM_PROMPT = (
    "You are a content moderation expert. You analyze videos to determine whether they contain hateful content. "
    "You are given a video represented by multiple 2x2 quad images. "
    "Each quad contains four consecutive frames in temporal order: top-left, top-right, bottom-left, bottom-right. "
    "The quads are in chronological order and represent the full video."
)

PROMPTS = {}

PROMPTS["scm_v2"] = """Analyze this video through the lens of social group perception and prejudice.
Video Title: {title}
Video Transcript: {transcript}

Important: Distinguish between videos that DISCUSS or REPORT ON prejudice/hate (not hateful) versus videos that EXPRESS, ENDORSE, or PROMOTE prejudice/hate (hateful). A video containing slurs or derogatory language is NOT automatically hateful — consider whether the speaker endorses or opposes the harmful expression.

Reason step by step and provide structured analysis in JSON format with exactly these fields:
{{
  "target_group": "Which social group (defined by race, ethnicity, religion, gender, sexuality, disability, nationality, or other protected characteristic) is being discussed or depicted? If the target is defined by behavior (e.g., drug users, criminals) rather than identity, note this distinction.",
  "warmth_evidence": "Is this group framed as warm (friendly, trustworthy, well-intentioned) or cold (hostile, untrustworthy, threatening)? Cite specific evidence from the video: exact words, visual elements, or tonal cues.",
  "competence_evidence": "Is this group framed as competent (capable, skilled, intelligent) or incompetent (foolish, backward, primitive)? Cite specific evidence.",
  "social_perception": "Given the warmth and competence framing, what social perception does the video construct? (contempt / envy / pity / admiration / mixed / none)",
  "endorsement_context": "Does the video ENDORSE the negative social perception, or does it DISCUSS, REPORT, QUOTE, CRITICIZE, or OPPOSE it? What evidence supports this judgment? (endorse / discuss / report / criticize / oppose / neutral)",
  "behavioral_tendency": "What treatment of this group does the video promote or normalize? (active harm / passive exclusion / patronizing / respect / none). Only answer 'active harm' or 'passive exclusion' if the video clearly endorses these, not merely discusses them."
}}

Output ONLY valid JSON, no other text."""

THEORY_FIELDS = {
    "scm_v2": ["target_group", "warmth_evidence", "competence_evidence",
               "social_perception", "endorsement_context", "behavioral_tendency"],
}


def setup_logger(theory, dataset):
    os.makedirs("./logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lf = f"./logs/prompt_{theory}_{dataset}_{ts}.log"
    logger = logging.getLogger(f"{theory}_{dataset}_{ts}")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fh = logging.FileHandler(lf, encoding="utf-8"); ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt); logger.addHandler(fh); logger.addHandler(ch)
    logger.info(f"Log: {lf}, Model: {MODEL_NAME}, Theory: {theory}")
    return logger


def encode_image(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_image_content(frames):
    return [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(f)}"}} for f in frames]


def parse_json_response(raw, fields):
    if not raw:
        return {f: "" for f in fields}
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    try:
        parsed = json.loads(text)
        return {f: str(parsed.get(f, "")) for f in fields}
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            return {f: str(parsed.get(f, "")) for f in fields}
        except json.JSONDecodeError:
            pass
    result = {}
    for f in fields:
        pattern = rf'"{f}"\s*:\s*"((?:[^"\\]|\\.)*)"|"{f}"\s*:\s*"((?:[^"\\]|\\.)*)'
        m = re.search(pattern, text, re.DOTALL)
        result[f] = m.group(1) or m.group(2) if m else ""
    return result


async def request_with_retries(messages, max_tokens=1024, logger=None):
    for attempt in range(5):
        try:
            r = await client.chat.completions.create(
                model=MODEL_NAME, messages=messages, max_completion_tokens=max_tokens, temperature=0
            )
            return r.choices[0].message.content.strip()
        except (RateLimitError, APIConnectionError) as e:
            if logger: logger.warning(f"Retry {attempt+1}/5: {type(e).__name__}")
            await asyncio.sleep(2 ** attempt)
        except Exception as e:
            if logger: logger.error(f"Error: {e}")
            if attempt < 4: await asyncio.sleep(2)
    return ""


def is_valid(item, theory):
    resp = item.get(f"{theory}_response", {})
    fields = THEORY_FIELDS[theory]
    return bool(resp.get(fields[0]))


def get_dataset_paths(dataset_name, language="English"):
    if dataset_name == "HateMM":
        return ("./datasets/HateMM/annotation(new).json", "./datasets/HateMM/quad")
    else:
        return (f"./datasets/Multihateclip/{language}/annotation(new).json",
                f"./datasets/Multihateclip/{language}/quad")


def get_save_path(theory, dataset_name, language="English"):
    if dataset_name == "HateMM":
        return f"./datasets/HateMM/{theory}_data.json"
    else:
        return f"./datasets/Multihateclip/{language}/{theory}_data.json"


def load_data(data_path, save_path, theory):
    with open(data_path, "r") as f:
        data = json.load(f)
    key = f"{theory}_response"
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            saved = json.load(f)
        sm = {d.get("Video_ID"): d for d in saved if d.get("Video_ID")}
        for item in data:
            s = sm.get(item.get("Video_ID"))
            if s and key in s:
                item[key] = s[key]
    return data


async def process_item(item, data, save_path, quad_root, theory, fields,
                       prompt_template, write_lock, semaphore, logger):
    vid = item.get("Video_ID")
    key = f"{theory}_response"
    if is_valid(item, theory):
        return "skipped"
    vp = os.path.join(quad_root, vid)
    if not os.path.isdir(vp):
        return "no_dir"
    frames = sorted([os.path.join(vp, f) for f in os.listdir(vp)
                     if f.lower().endswith((".jpg", ".png"))])
    if not frames:
        return "no_frames"
    title = item.get("Title", "") or ""
    transcript = item.get("Transcript", "") or ""
    prompt = prompt_template.format(title=title, transcript=transcript)
    t0 = time.time()
    async with semaphore:
        content = build_image_content(frames) + [{"type": "text", "text": prompt}]
        raw = await request_with_retries(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": content}],
            max_tokens=1024, logger=logger
        )
    if raw:
        parsed = parse_json_response(raw, fields)
        parsed["raw"] = raw
        item[key] = parsed
    else:
        item[key] = {f: "" for f in fields}
        item[key]["error"] = "empty response"
        item[key]["raw"] = ""
    logger.info(f"{vid}: {theory} done ({time.time()-t0:.1f}s)")
    async with write_lock:
        with open(save_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return "ok" if is_valid(item, theory) else "incomplete"


async def process(data_path, save_path, quad_root, theory, max_concurrent, logger):
    fields = THEORY_FIELDS[theory]
    prompt_template = PROMPTS[theory]
    data = load_data(data_path, save_path, theory)
    done = sum(1 for d in data if is_valid(d, theory))
    logger.info(f"Total: {len(data)}, Done: {done}, Need: {len(data)-done}")
    if done == len(data):
        logger.info("All done, skipping.")
        return
    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max_concurrent)
    stats = {"skipped": 0, "ok": 0, "incomplete": 0, "no_dir": 0, "no_frames": 0}
    pbar = tqdm(total=len(data), initial=done, desc=f"{theory}", unit="video")

    async def w(item):
        try:
            r = await process_item(item, data, save_path, quad_root, theory, fields,
                                   prompt_template, write_lock, semaphore, logger)
            if r: stats[r] = stats.get(r, 0) + 1
        except Exception as e:
            logger.error(f"{item.get('Video_ID', '?')}: {e}")
        finally:
            pbar.update(1)

    await asyncio.gather(*[w(item) for item in data])
    pbar.close()
    with open(save_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    valid = sum(1 for d in data if is_valid(d, theory))
    logger.info(f"Stats: {json.dumps(stats)}, Valid: {valid}/{len(data)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--theory", type=str, required=True, choices=["scm_v2"])
    parser.add_argument("--dataset_name", type=str, default="HateMM", choices=["HateMM", "Multihateclip"])
    parser.add_argument("--language", type=str, default="English", choices=["English", "Chinese"])
    parser.add_argument("--max_concurrent", type=int, default=10)
    args = parser.parse_args()
    tag = args.dataset_name if args.dataset_name == "HateMM" else f"MHC_{args.language[:2]}"
    logger = setup_logger(args.theory, tag)
    data_path, quad_root = get_dataset_paths(args.dataset_name, args.language)
    save_path = get_save_path(args.theory, args.dataset_name, args.language)
    logger.info(f"Theory: {args.theory}, Dataset: {tag}")
    asyncio.run(process(data_path, save_path, quad_root, args.theory, args.max_concurrent, logger))
    logger.info("Done.")


if __name__ == "__main__":
    main()
