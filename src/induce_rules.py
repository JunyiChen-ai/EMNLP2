"""
TMBD Steps B-C: Gap filtering + rule induction from observations.

Reads structured observations from observe_training.py output.
Filters for discovery set (positive verdict, no covered rules, uncovered mechanism).
Aggregates by mechanism x protected_attribute x evidence_modality.
Induces new rules from recurrent uncovered triples.

Then extracts preconditions using an implicit-mechanism-preserving prompt.

Usage:
  python src/induce_rules.py --dataset MHClip_EN
  python src/induce_rules.py --dataset MHClip_ZH
  python src/induce_rules.py --dataset MHClip_EN --extract-preconditions  # needs GPU/vLLM
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict

PROJECT_ROOT = "/data/jehc223/EMNLP2"

DATASET_CONSTITUTION = {
    "MHClip_EN": "youtube",
    "MHClip_ZH": "bilibili",
}

# Mechanisms covered by existing rules (mapped from constitution)
COVERED_MECHANISMS = {
    "youtube": {
        "incitement_violence", "incitement_hatred", "dehumanization",
        "glorification_violence", "explicit_slurs", "stereotyping_as_fact",
        "claims_inferiority", "supremacism", "conspiracy", "denial_violence",
    },
    "bilibili": {
        "discrimination_violence", "attacks_disparagement", "verbal_abuse",
        "hateful_ideologies", "mocking_death_disability", "disaster_jokes",
        "gender_based_hate", "inciting_confrontation",
    },
}

# Rule induction prompt — text-only LLM call
RULE_INDUCTION_PROMPT = """You are a content moderation policy expert. Based on the following observations from real videos, generalize a new content moderation rule.

These observations describe hateful or offensive content that is NOT covered by existing platform rules. The content uses IMPLICIT mechanisms (not explicit slurs or direct threats).

Observations (mechanism: {mechanism}, targeting: {attribute}):
{observations}

Write a single, clear content moderation rule that would cover this type of content. The rule should:
1. Name the specific implicit mechanism (e.g., ironic mockery, coded language)
2. Describe what observable cues to look for in video frames, title, and transcript
3. NOT require "explicit", "unambiguous", or "direct" language — these mechanisms are inherently indirect
4. Be specific enough to avoid flagging legitimate content

Respond with ONLY the rule text (1-3 sentences), no preamble."""

# Implicit-mechanism-preserving precondition extraction
IMPLICIT_PRECONDITION_PROMPT = """Given a content moderation rule for implicit/indirect hateful content in videos, extract verifiable pre-conditions. A video violates the rule when ALL pre-conditions are satisfied.

IMPORTANT: These rules target IMPLICIT hate mechanisms. Pre-conditions must NOT require "explicit", "unambiguous", or "direct" language. Instead, each pre-condition should describe an OBSERVABLE multimodal pattern — something visible in frames, audible in transcript text, or readable in title/text — that indicates the implicit mechanism.

The first pre-condition must always identify: (a) a protected target group AND (b) an observable multimodal cue that the group is being referenced (visual depiction, textual mention, or spoken reference in transcript).

Example for implicit mechanism:
Policy: Content that uses ironic or sarcastic framing to mock or ridicule a specific group.
Pre-conditions:
1. A specific identity group is referenced or depicted through visual, textual, or spoken cues in the transcript.
2. The video's tone, framing, or juxtaposition contains markers of irony or sarcasm (e.g., exaggerated praise, contradictory visual-text pairing, mocking language in transcript).
3. The ironic or sarcastic framing is directed at the referenced group in a way that ridicules, belittles, or demeans them.

Now extract pre-conditions for this rule:
Policy: {rule_text}"""


def load_observations(obs_path):
    """Load observations from JSONL."""
    records = []
    with open(obs_path) as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def filter_discovery_set(observations, platform):
    """Step B: Filter for discovery set."""
    covered = COVERED_MECHANISMS[platform]
    discovery = []
    for obs in observations:
        if obs.get("verdict") != "HATEFUL_OR_OFFENSIVE":
            continue
        if obs.get("covered_rule_ids") and len(obs["covered_rule_ids"]) > 0:
            continue
        mech = obs.get("mechanism", "none").lower().strip()
        if mech in ("none", "unknown", ""):
            continue
        # Check if mechanism is uncovered
        # Handle OTHER:describe format
        base_mech = mech.split(":")[0] if mech.startswith("other") else mech
        if base_mech in covered:
            continue
        discovery.append(obs)
    return discovery


def aggregate_triples(discovery_set, min_count=3):
    """Step C part 1: Aggregate by mechanism x attribute x modality."""
    triples = defaultdict(list)
    for obs in discovery_set:
        mech = obs.get("mechanism", "unknown").lower().strip()
        attr = obs.get("protected_attribute", "none").lower().strip()
        mod = obs.get("evidence_modality", "unknown").lower().strip()
        key = (mech, attr, mod)
        triples[key].append(obs)

    # Filter by min count
    recurrent = {k: v for k, v in triples.items() if len(v) >= min_count}
    return recurrent


def induce_rules_from_triples(recurrent_triples, llm, sampling_params):
    """Step C part 2: Induce new rules from recurrent triples using text-only LLM."""
    new_rules = []
    rule_idx = 1

    for (mech, attr, mod), observations in sorted(recurrent_triples.items(), key=lambda x: -len(x[1])):
        # Collect observable_cues for the prompt
        cues = []
        for obs in observations[:10]:  # Cap at 10 examples
            cue = obs.get("observable_cues", "")
            tgt = obs.get("target_group", "unknown")
            if cue:
                cues.append(f"- Target: {tgt}. Cues: {cue}")

        if not cues:
            continue

        obs_text = "\n".join(cues)
        prompt = RULE_INDUCTION_PROMPT.format(
            mechanism=mech,
            attribute=attr,
            observations=obs_text,
        )

        messages = [
            {"role": "system", "content": "You are a content moderation policy expert."},
            {"role": "user", "content": prompt},
        ]

        try:
            outputs = llm.chat(messages=[messages], sampling_params=sampling_params)
            rule_text = outputs[0].outputs[0].text.strip()
        except Exception as e:
            logging.warning(f"  Rule induction failed for ({mech}, {attr}, {mod}): {e}")
            continue

        new_rules.append({
            "rule_id": f"D-R{rule_idx}",
            "name": f"Discovered: {mech} ({attr})",
            "objectified_rule": rule_text,
            "mechanism": mech,
            "protected_attribute": attr,
            "evidence_modality": mod,
            "n_observations": len(observations),
            "source_triple": f"{mech} × {attr} × {mod}",
        })
        logging.info(f"  Induced D-R{rule_idx}: {mech} × {attr} × {mod} (n={len(observations)})")
        logging.info(f"    Rule: {rule_text[:100]}...")
        rule_idx += 1

    return new_rules


def extract_preconditions_implicit(new_rules, llm, sampling_params):
    """Extract preconditions using implicit-mechanism-preserving prompt."""
    for rule in new_rules:
        prompt = IMPLICIT_PRECONDITION_PROMPT.format(rule_text=rule["objectified_rule"])
        messages = [
            {"role": "system", "content": "You are a content moderation policy expert."},
            {"role": "user", "content": prompt},
        ]

        try:
            outputs = llm.chat(messages=[messages], sampling_params=sampling_params)
            response = outputs[0].outputs[0].text.strip()
        except Exception as e:
            logging.warning(f"  Precondition extraction failed for {rule['rule_id']}: {e}")
            rule["preconditions"] = [rule["objectified_rule"]]
            continue

        # Parse numbered preconditions
        import re
        lines = response.strip().split("\n")
        preconditions = []
        for line in lines:
            line = line.strip()
            match = re.match(r'^\d+[\.\)]\s*(.+)', line)
            if match:
                preconditions.append(match.group(1).strip())

        if not preconditions:
            # Fallback: use full response as single precondition
            preconditions = [response[:500]]

        rule["preconditions"] = preconditions
        rule["raw_response"] = response
        logging.info(f"  {rule['rule_id']}: {len(preconditions)} preconditions")
        for i, pc in enumerate(preconditions):
            logging.info(f"    {i+1}. {pc[:80]}...")

    return new_rules


def main():
    parser = argparse.ArgumentParser(description="TMBD Steps B-C: Gap filter + rule induction")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONSTITUTION))
    parser.add_argument("--min-count", type=int, default=3, help="Min observations per triple")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
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
            logging.FileHandler(os.path.join(log_dir, f"induce_{args.dataset}.log")),
            logging.StreamHandler(),
        ],
    )

    # Load observations
    obs_path = os.path.join(root, "results", "observations", args.dataset, "train.jsonl")
    if not os.path.exists(obs_path):
        logging.error(f"Observations not found: {obs_path}. Run observe_training.py first.")
        sys.exit(1)
    observations = load_observations(obs_path)
    logging.info(f"Loaded {len(observations)} observations for {args.dataset}")

    # Stats
    n_hateful = sum(1 for o in observations if o.get("verdict") == "HATEFUL_OR_OFFENSIVE")
    n_normal = sum(1 for o in observations if o.get("verdict") == "NORMAL")
    logging.info(f"  HATEFUL_OR_OFFENSIVE: {n_hateful}, NORMAL: {n_normal}")

    # Step B: Filter discovery set
    discovery = filter_discovery_set(observations, platform)
    logging.info(f"\nDiscovery set: {len(discovery)} videos (positive, no covered rules, uncovered mechanism)")

    if not discovery:
        logging.warning("No gap observations found. Coverage-gap hypothesis may be wrong.")
        # Save empty result + write merged as copy of original
        out_dir = os.path.join(root, "constitution")
        out_path = os.path.join(out_dir, f"discovered_{platform}.json")
        with open(out_path, "w") as f:
            json.dump([], f)
        original_path = os.path.join(out_dir, f"preconditions_{platform}.json")
        with open(original_path) as f:
            original_rules = json.load(f)
        merged_path = os.path.join(out_dir, f"preconditions_{platform}_merged.json")
        with open(merged_path, "w") as f:
            json.dump(original_rules, f, indent=2, ensure_ascii=False)
        logging.info(f"No discovery — merged file is copy of original: {merged_path}")
        return

    # Distribution of uncovered mechanisms
    mech_counter = Counter(o.get("mechanism", "unknown") for o in discovery)
    logging.info(f"\nUncovered mechanism distribution:")
    for mech, count in mech_counter.most_common():
        logging.info(f"  {mech}: {count}")

    attr_counter = Counter(o.get("protected_attribute", "none") for o in discovery)
    logging.info(f"\nProtected attribute distribution:")
    for attr, count in attr_counter.most_common():
        logging.info(f"  {attr}: {count}")

    # Step C part 1: Aggregate triples
    recurrent = aggregate_triples(discovery, min_count=args.min_count)
    logging.info(f"\nRecurrent uncovered triples (count >= {args.min_count}): {len(recurrent)}")
    for (mech, attr, mod), obs_list in sorted(recurrent.items(), key=lambda x: -len(x[1])):
        logging.info(f"  {mech} × {attr} × {mod}: {len(obs_list)} videos")

    if not recurrent:
        logging.warning(f"No recurrent triples with count >= {args.min_count}.")
        logging.info("Trying with min_count=2...")
        recurrent = aggregate_triples(discovery, min_count=2)
        if not recurrent:
            logging.warning("Still no recurrent triples. Coverage-gap hypothesis may be wrong for this mechanism × attribute × modality granularity.")
            # Save discovery set stats for analysis
            obs_out_dir = os.path.join(root, "results", "observations", args.dataset)
            stats_path = os.path.join(obs_out_dir, "discovery_stats.json")
            with open(stats_path, "w") as f:
                json.dump({
                    "n_observations": len(observations),
                    "n_hateful": n_hateful,
                    "n_discovery": len(discovery),
                    "mechanism_distribution": dict(mech_counter),
                    "attribute_distribution": dict(attr_counter),
                }, f, indent=2)
            # Write merged as copy of original so downstream doesn't crash
            const_dir = os.path.join(root, "constitution")
            original_path = os.path.join(const_dir, f"preconditions_{platform}.json")
            with open(original_path) as f:
                original_rules = json.load(f)
            merged_path = os.path.join(const_dir, f"preconditions_{platform}_merged.json")
            with open(merged_path, "w") as f:
                json.dump(original_rules, f, indent=2, ensure_ascii=False)
            logging.info(f"No recurrent triples — merged file is copy of original: {merged_path}")
            return

    # Step C part 2: Induce rules (requires LLM)
    logging.info(f"\nLoading LLM for rule induction...")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=0.92,
        max_model_len=4096,
    )
    sampling_params = SamplingParams(temperature=0.3, max_tokens=512)

    new_rules = induce_rules_from_triples(recurrent, llm, sampling_params)
    logging.info(f"\nInduced {len(new_rules)} new rules")

    # Always extract preconditions — downstream scoring requires them
    if new_rules:
        logging.info("\nExtracting preconditions (implicit-mechanism-preserving)...")
        pc_sampling = SamplingParams(temperature=0, max_tokens=512)
        new_rules = extract_preconditions_implicit(new_rules, llm, pc_sampling)

    # Save discovered rules
    out_dir = os.path.join(root, "constitution")
    out_path = os.path.join(out_dir, f"discovered_{platform}.json")
    with open(out_path, "w") as f:
        json.dump(new_rules, f, indent=2, ensure_ascii=False)
    logging.info(f"\nSaved {len(new_rules)} discovered rules to {out_path}")

    # Also save merged constitution (original + discovered)
    original_path = os.path.join(root, "constitution", f"preconditions_{platform}.json")
    with open(original_path) as f:
        original_rules = json.load(f)

    merged = original_rules + new_rules
    merged_path = os.path.join(out_dir, f"preconditions_{platform}_merged.json")
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved merged constitution ({len(original_rules)} original + {len(new_rules)} discovered) to {merged_path}")


if __name__ == "__main__":
    main()
