"""
TMBD Pivot: Observation-grounded precondition re-decomposition.

Instead of discovering new rules, uses observation data from training videos
to rewrite preconditions for existing rules. The observation data shows what
the MLLM actually sees when identifying rule violations — preconditions are
rewritten to match the MLLM's own vocabulary and detection patterns.

Key design choice: disjunctive rules are split into variants (branches)
rather than forced into one AND chain.

Usage:
  python src/refine_preconditions.py --dataset MHClip_EN  # needs GPU/vLLM
  python src/refine_preconditions.py --dataset MHClip_ZH
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict

PROJECT_ROOT = "/data/jehc223/EMNLP2"

DATASET_CONSTITUTION = {
    "MHClip_EN": "youtube",
    "MHClip_ZH": "bilibili",
}

# Re-decomposition prompt: observation-grounded, implicit-aware
REDECOMPOSE_PROMPT = """You are rewriting the pre-conditions for a content moderation rule based on REAL examples of how hateful content appears in videos.

Rule: {rule_name}
Rule description: {objectified_rule}

Here are real observations from videos that violate this rule:
{observation_examples}

The mechanisms observed include: {mechanisms}

Based on these real observations, extract verifiable pre-conditions that would catch these patterns. A video violates the rule when ALL pre-conditions of at least ONE branch are satisfied.

IMPORTANT GUIDELINES:
- Pre-conditions must describe OBSERVABLE multimodal patterns (visible in frames, readable in text, spoken in transcript)
- Do NOT require "explicit", "unambiguous", or "direct" evidence — these mechanisms can be implicit, contextual, or conveyed through humor/irony/framing
- The first pre-condition should identify a target group through ANY observable cue (visual, textual, or spoken)
- If the rule covers MULTIPLE distinct mechanisms (e.g., slurs AND stereotypes), create SEPARATE BRANCHES — each branch is a different way to violate the rule
- Each branch should be 2-3 pre-conditions connected by AND logic

Format your response as:
BRANCH 1: [mechanism name]
1. [pre-condition]
2. [pre-condition]

BRANCH 2: [mechanism name]  (if applicable)
1. [pre-condition]
2. [pre-condition]
"""

# Fallback for rules with no/few observations
FALLBACK_PROMPT = """You are rewriting the pre-conditions for a content moderation rule. The current pre-conditions are too strict — they require "explicit" or "unambiguous" evidence that misses implicit, contextual, or humor-based violations.

Rule: {rule_name}
Rule description: {objectified_rule}

Current pre-conditions (too strict):
{current_preconditions}

Rewrite these pre-conditions to also catch IMPLICIT violations. Guidelines:
- Do NOT require "explicit", "unambiguous", or "direct" evidence
- Include implicit mechanisms: irony, sarcasm, humor-based stereotyping, coded language, contextual framing
- The first pre-condition should identify a target group through ANY observable cue
- If the rule covers multiple mechanisms, create SEPARATE BRANCHES

Format:
BRANCH 1: [mechanism]
1. [pre-condition]
2. [pre-condition]

BRANCH 2: [mechanism] (if applicable)
1. [pre-condition]
2. [pre-condition]
"""


def load_observations(obs_path):
    records = []
    with open(obs_path) as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def group_observations_by_rule(observations):
    """Group hateful observations by their assigned rule IDs."""
    rule_obs = defaultdict(list)
    for obs in observations:
        if obs.get("verdict") != "HATEFUL_OR_OFFENSIVE":
            continue
        for rule_id in obs.get("covered_rule_ids", []):
            rule_obs[rule_id].append(obs)
    return dict(rule_obs)


def parse_branches(response_text):
    """Parse BRANCH format from LLM response into structured branches."""
    branches = []
    current_branch = None
    current_preconditions = []

    for line in response_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Check for branch header
        branch_match = re.match(r'^BRANCH\s+\d+[:\s]*(.+)?', line, re.IGNORECASE)
        if branch_match:
            # Save previous branch
            if current_branch is not None and current_preconditions:
                branches.append({
                    "branch_name": current_branch,
                    "preconditions": current_preconditions,
                })
            current_branch = branch_match.group(1).strip() if branch_match.group(1) else f"branch_{len(branches)+1}"
            current_preconditions = []
            continue

        # Check for numbered precondition
        pc_match = re.match(r'^\d+[\.\)]\s*(.+)', line)
        if pc_match:
            current_preconditions.append(pc_match.group(1).strip())

    # Save last branch
    if current_branch is not None and current_preconditions:
        branches.append({
            "branch_name": current_branch,
            "preconditions": current_preconditions,
        })

    # Fallback: if no branches parsed, treat entire response as one branch
    if not branches:
        all_pcs = []
        for line in response_text.strip().split("\n"):
            pc_match = re.match(r'^\d+[\.\)]\s*(.+)', line.strip())
            if pc_match:
                all_pcs.append(pc_match.group(1).strip())
        if all_pcs:
            branches.append({
                "branch_name": "default",
                "preconditions": all_pcs,
            })

    return branches


def main():
    parser = argparse.ArgumentParser(description="Observation-grounded precondition re-decomposition")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONSTITUTION))
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--min-observations", type=int, default=3,
                        help="Min observations per rule to use observation-grounded prompt")
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    args = parser.parse_args()

    platform = DATASET_CONSTITUTION[args.dataset]
    root = args.project_root

    # Logging
    log_dir = os.path.join(root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"refine_{args.dataset}.log")),
            logging.StreamHandler(),
        ],
    )

    # Load existing constitution
    precond_path = os.path.join(root, "constitution", f"preconditions_{platform}.json")
    with open(precond_path) as f:
        rules = json.load(f)
    logging.info(f"Loaded {len(rules)} rules from {platform}")

    # Load observations
    obs_path = os.path.join(root, "results", "observations", args.dataset, "train.jsonl")
    if not os.path.exists(obs_path):
        logging.error(f"Observations not found: {obs_path}")
        sys.exit(1)
    observations = load_observations(obs_path)
    logging.info(f"Loaded {len(observations)} observations")

    # Group observations by rule
    rule_obs = group_observations_by_rule(observations)
    for rule_id, obs_list in sorted(rule_obs.items()):
        logging.info(f"  {rule_id}: {len(obs_list)} observations")

    # Load LLM
    logging.info("Loading vLLM for precondition re-decomposition...")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=0.92,
        max_model_len=4096,
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=1024)

    # Re-decompose each rule
    refined_rules = []

    for rule in rules:
        rule_id = rule["rule_id"]
        # Normalize rule_id for observation matching (observations use R1-R9 not YT-R1)
        obs_key = rule_id.replace("YT-", "").replace("BL-", "")
        obs_list = rule_obs.get(obs_key, [])

        if len(obs_list) >= args.min_observations:
            # Observation-grounded re-decomposition
            examples = []
            for o in obs_list[:15]:  # Cap examples
                cue = o.get("observable_cues", "")
                mech = o.get("mechanism", "unknown")
                tgt = o.get("target_group", "unknown")
                if cue:
                    examples.append(f"- Target: {tgt}, Mechanism: {mech}, Cues: {cue}")

            from collections import Counter
            mech_counter = Counter(o.get("mechanism", "unknown") for o in obs_list)
            mechanisms = ", ".join(f"{m} ({c})" for m, c in mech_counter.most_common(5))

            prompt = REDECOMPOSE_PROMPT.format(
                rule_name=rule.get("name", rule_id),
                objectified_rule=rule.get("objectified_rule", "")[:500],
                observation_examples="\n".join(examples),
                mechanisms=mechanisms,
            )
            logging.info(f"\n{rule_id}: observation-grounded re-decomposition ({len(obs_list)} observations)")
        else:
            # Fallback: rewrite without observations
            current_pcs = "\n".join(f"{i+1}. {pc}" for i, pc in enumerate(rule.get("preconditions", [])))
            prompt = FALLBACK_PROMPT.format(
                rule_name=rule.get("name", rule_id),
                objectified_rule=rule.get("objectified_rule", "")[:500],
                current_preconditions=current_pcs,
            )
            logging.info(f"\n{rule_id}: fallback re-decomposition (only {len(obs_list)} observations)")

        messages = [
            {"role": "system", "content": "You are a content moderation policy expert specializing in detecting both explicit and implicit hate speech in online videos."},
            {"role": "user", "content": prompt},
        ]

        try:
            outputs = llm.chat(messages=[messages], sampling_params=sampling_params)
            response = outputs[0].outputs[0].text.strip()
        except Exception as e:
            logging.error(f"  {rule_id}: LLM call failed: {e}")
            # Keep original preconditions
            refined_rules.append(rule)
            continue

        branches = parse_branches(response)
        logging.info(f"  {rule_id}: {len(branches)} branches parsed")

        if not branches:
            logging.warning(f"  {rule_id}: no branches parsed, keeping original")
            refined_rules.append(rule)
            continue

        # Create rule entries — one per branch
        # If single branch: same rule_id, updated preconditions
        # If multiple branches: rule_id + suffix (e.g., YT-R5a, YT-R5b)
        if len(branches) == 1:
            refined_rule = dict(rule)
            refined_rule["preconditions"] = branches[0]["preconditions"]
            refined_rule["branch_name"] = branches[0]["branch_name"]
            refined_rule["raw_response"] = response
            refined_rules.append(refined_rule)
            for i, pc in enumerate(branches[0]["preconditions"]):
                logging.info(f"    {i+1}. {pc[:80]}...")
        else:
            for j, branch in enumerate(branches):
                suffix = chr(ord('a') + j)
                refined_rule = dict(rule)
                refined_rule["rule_id"] = f"{rule_id}{suffix}"
                refined_rule["name"] = f"{rule.get('name', rule_id)} ({branch['branch_name']})"
                refined_rule["preconditions"] = branch["preconditions"]
                refined_rule["branch_name"] = branch["branch_name"]
                refined_rule["raw_response"] = response
                refined_rules.append(refined_rule)
                logging.info(f"  Branch {suffix}: {branch['branch_name']}")
                for i, pc in enumerate(branch["preconditions"]):
                    logging.info(f"    {i+1}. {pc[:80]}...")

    # Save refined constitution
    out_dir = os.path.join(root, "constitution")
    out_path = os.path.join(out_dir, f"preconditions_{platform}_refined.json")
    with open(out_path, "w") as f:
        json.dump(refined_rules, f, indent=2, ensure_ascii=False)
    logging.info(f"\nSaved {len(refined_rules)} refined rules (from {len(rules)} original) to {out_path}")

    # Also create the _merged file that downstream scripts expect
    merged_path = os.path.join(out_dir, f"preconditions_{platform}_merged.json")
    with open(merged_path, "w") as f:
        json.dump(refined_rules, f, indent=2, ensure_ascii=False)
    logging.info(f"Also saved as merged: {merged_path}")


if __name__ == "__main__":
    main()
