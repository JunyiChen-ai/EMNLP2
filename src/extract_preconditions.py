"""
Step 2: Precondition Extraction.

Decomposes each objectified rule into a chain of independently verifiable
preconditions using LLM.

Usage:
  python src/extract_preconditions.py --constitution youtube --model Qwen/Qwen3-VL-8B-Instruct
"""

import argparse
import json
import logging
import os
import re
import sys

PROJECT_ROOT = "/data/jehc223/EMNLP2"

EXTRACTION_PROMPT = """Given a content moderation policy for videos, extract the pre-conditions for violating the policy. A video violates the policy only when ALL pre-conditions are satisfied. Each pre-condition should be a single verifiable statement that can be checked by observing the video's visual frames, title text, and spoken transcript.

##Example 1:
##Policy: Content that dehumanizes individuals or groups by calling them subhuman, comparing them to animals, insects, pests, disease, or any other non-human entity based on protected group status.
##Extracted pre-conditions:
1. A specific identity group (defined by race, ethnicity, religion, gender, sexual orientation, disability, or nationality) is referenced or depicted in the video.
2. The video contains language or imagery that compares members of that group to animals, insects, pests, disease, subhuman entities, or other non-human entities.

##Example 2:
##Policy: Content that uses racial, religious, or other slurs and stereotypes that incite or promote hatred, in speech, text, or imagery.
##Extracted pre-conditions:
1. The video contains racial, religious, or other identity-based slurs or stereotypes in its speech, on-screen text, or imagery.
2. The slurs or stereotypes are used in a way that incites or promotes hatred rather than quoting, condemning, or educating.

Now given a new policy, perform this task.
Policy: {objectified_rule}"""


def parse_preconditions(text):
    """Parse numbered list of preconditions from LLM output."""
    lines = text.strip().split("\n")
    preconditions = []
    for line in lines:
        line = line.strip()
        # Match lines starting with number + dot/paren
        m = re.match(r'^\d+[\.\)]\s*(.+)', line)
        if m:
            precond = m.group(1).strip()
            # Remove leading/trailing brackets if present
            precond = precond.strip("[]")
            if precond:
                preconditions.append(precond)
    return preconditions


def run_llm(llm, sampling_params, prompt):
    """Run a text-only LLM call and return the generated text."""
    messages = [{"role": "user", "content": prompt}]
    outputs = llm.chat(messages=[messages], sampling_params=sampling_params)
    return outputs[0].outputs[0].text.strip()


def main():
    parser = argparse.ArgumentParser(description="Extract preconditions from objectified rules")
    parser.add_argument("--constitution", required=True, choices=["youtube", "bilibili"])
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    args = parser.parse_args()

    # Setup logging
    log_dir = os.path.join(args.project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"extract_preconditions_{args.constitution}.log")),
            logging.StreamHandler(),
        ],
    )

    # Load objectified rules
    obj_path = os.path.join(args.project_root, "constitution", f"objectified_{args.constitution}.json")
    if not os.path.exists(obj_path):
        logging.error(f"Objectified rules not found: {obj_path}. Run objectify_rules.py first.")
        sys.exit(1)

    with open(obj_path) as f:
        objectified_rules = json.load(f)

    logging.info(f"Loaded {len(objectified_rules)} objectified rules from {args.constitution}")

    # Init vLLM
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=512)

    # Extract preconditions for each rule
    results = []
    for rule in objectified_rules:
        rule_id = rule["rule_id"]
        obj_text = rule["objectified"]
        logging.info(f"Extracting preconditions for {rule_id}: {obj_text[:80]}...")

        prompt = EXTRACTION_PROMPT.format(objectified_rule=obj_text)
        response = run_llm(llm, sampling_params, prompt)
        preconditions = parse_preconditions(response)

        if not preconditions:
            logging.warning(f"  No preconditions parsed for {rule_id}. Raw response: {response[:300]}")
            # Fallback: use the full objectified rule as a single precondition
            preconditions = [obj_text]

        logging.info(f"  {rule_id}: {len(preconditions)} preconditions extracted")
        for i, p in enumerate(preconditions):
            logging.info(f"    C{i+1}: {p}")

        results.append({
            "rule_id": rule_id,
            "name": rule.get("name", ""),
            "objectified_rule": obj_text,
            "preconditions": preconditions,
            "raw_response": response,
        })

    # Save output
    out_dir = os.path.join(args.project_root, "constitution")
    out_path = os.path.join(out_dir, f"preconditions_{args.constitution}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"Saved preconditions to {out_path}")

    # Summary
    total_preconditions = sum(len(r["preconditions"]) for r in results)
    logging.info(f"Total: {len(results)} rules, {total_preconditions} preconditions")


if __name__ == "__main__":
    main()
