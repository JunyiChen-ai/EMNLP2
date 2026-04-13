"""
Step 1: Rules Objectification.

Reads platform hate speech rules from constitution JSON, rewrites each rule
via LLM-as-Optimizer loop until objectiveness score >= 9/10.

Usage:
  python src/objectify_rules.py --constitution youtube --model Qwen/Qwen3-VL-8B-Instruct
"""

import argparse
import json
import logging
import os
import re
import sys
import time

PROJECT_ROOT = "/data/jehc223/EMNLP2"

SCORE_PROMPT = """Please act as an impartial judge and evaluate the objectiveness for the provided guideline. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 10 (10 is the most objective).

Guideline: {rule_text}"""

REWRITE_PROMPT = """The following content moderation rule scored {score}/10 on objectiveness. Rewrite it to be more objective and actionable, while preserving its original intent. The rule should be verifiable from observing a video's visual content, audio, and text.

Original rule: {rule_text}
Score: {score}/10
Feedback: {explanation}

Rewritten rule:"""


def parse_score(text):
    """Extract integer score from LLM response. Handles '8/10', '8 out of 10', 'score: 8', etc."""
    # Try "X/10" pattern
    m = re.search(r'(\d+)\s*/\s*10', text)
    if m:
        return int(m.group(1))
    # Try "X out of 10"
    m = re.search(r'(\d+)\s+out\s+of\s+10', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Try "score: X" or "rating: X"
    m = re.search(r'(?:score|rating)\s*[:=]\s*(\d+)', text, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        if 1 <= val <= 10:
            return val
    # Last resort: find the last standalone digit in 1-9 range (exclude 10 to avoid "out of 10" residue)
    digits = re.findall(r'\b(\d)\b', text)
    if digits:
        return int(digits[-1])
    return None


def run_llm(llm, sampling_params, prompt):
    """Run a text-only LLM call and return the generated text."""
    messages = [{"role": "user", "content": prompt}]
    outputs = llm.chat(messages=[messages], sampling_params=sampling_params)
    return outputs[0].outputs[0].text.strip()


def objectify_one_rule(llm, scoring_params, rewrite_params, rule_text, max_iter=10, target_score=9):
    """Objectify a single rule via iterative rewriting until score >= target."""
    if max_iter < 1:
        return rule_text, 0, []

    current_text = rule_text
    history = []

    for iteration in range(max_iter):
        # Score the current rule (deterministic)
        score_prompt = SCORE_PROMPT.format(rule_text=current_text)
        response = run_llm(llm, scoring_params, score_prompt)
        score = parse_score(response)

        if score is None:
            logging.warning(f"  Could not parse score from response: {response[:200]}")
            score = 0

        logging.info(f"  Iteration {iteration}: score={score}/10")
        history.append({
            "iteration": iteration,
            "text": current_text,
            "score": score,
            "feedback": response,
        })

        if score >= target_score:
            logging.info(f"  Reached target score {score} >= {target_score}")
            return current_text, score, history

        # Rewrite (stochastic)
        rewrite_prompt = REWRITE_PROMPT.format(
            rule_text=current_text,
            score=score,
            explanation=response,
        )
        current_text = run_llm(llm, rewrite_params, rewrite_prompt)
        # Clean up: remove quotes if LLM wraps the rewrite
        current_text = current_text.strip().strip('"').strip("'")

    logging.warning(f"  Did not reach target after {max_iter} iterations. Best: {score}/10")
    # Return the version with the highest score
    best = max(history, key=lambda h: h["score"])
    return best["text"], best["score"], history


def main():
    parser = argparse.ArgumentParser(description="Objectify hate speech rules")
    parser.add_argument("--constitution", required=True, choices=["youtube", "bilibili"])
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--target-score", type=int, default=9)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    args = parser.parse_args()

    # Setup logging
    log_dir = os.path.join(args.project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"objectify_{args.constitution}.log")),
            logging.StreamHandler(),
        ],
    )

    # Load constitution
    const_path = os.path.join(args.project_root, "constitution", f"{args.constitution}.json")
    with open(const_path) as f:
        constitution = json.load(f)

    rules = constitution["rules"]
    logging.info(f"Loaded {len(rules)} rules from {args.constitution}")

    # Init vLLM
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )
    scoring_params = SamplingParams(temperature=0, max_tokens=512)
    rewrite_params = SamplingParams(temperature=0.7, max_tokens=512)

    # Objectify each rule
    results = []
    for rule in rules:
        rule_id = rule["rule_id"]
        original = rule["text"]
        logging.info(f"Processing {rule_id}: {original[:80]}...")

        objectified, score, history = objectify_one_rule(
            llm, scoring_params, rewrite_params, original,
            max_iter=args.max_iter,
            target_score=args.target_score,
        )

        results.append({
            "rule_id": rule_id,
            "name": rule.get("name", ""),
            "original": original,
            "objectified": objectified,
            "objectiveness_score": score,
            "iterations": len(history),
            "history": history,
        })

    # Save output
    out_dir = os.path.join(args.project_root, "constitution")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"objectified_{args.constitution}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"Saved {len(results)} objectified rules to {out_path}")

    # Summary
    for r in results:
        logging.info(f"  {r['rule_id']}: score={r['objectiveness_score']}, iterations={r['iterations']}")


if __name__ == "__main__":
    main()
