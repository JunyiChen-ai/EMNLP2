#!/usr/bin/env python3
"""MLLM-based direct classification with confidence scoring.

Uses the previously generated analysis as context for a focused classification
prompt. This is Stage 1b: convert MLLM understanding into a classification signal.
Runs on text only (no GPU needed for vllm), uses the cached analysis.
"""

import argparse
import json
import os
import re
import sys


def parse_classification_from_analysis(analysis: str) -> dict:
    """Parse hate assessment from MLLM analysis text."""
    analysis_lower = analysis.lower()

    # Look for section 4 (HATE ASSESSMENT)
    sections = re.split(r'\n\s*\d+\.', analysis)
    hate_section = ""
    for i, section in enumerate(sections):
        if "hate assessment" in section.lower() or "classification" in section.lower():
            hate_section = section
            break
    if not hate_section and len(sections) >= 4:
        hate_section = sections[3] if len(sections) > 3 else sections[-1]

    # Also check section 5 (CONFIDENCE)
    confidence_section = ""
    if len(sections) >= 5:
        confidence_section = sections[-1]

    # Parse classification
    # Priority: look for explicit statements in hate assessment section
    hate_text = hate_section.lower() if hate_section else analysis_lower[-1000:]

    # Strong hateful indicators
    hateful_patterns = [
        r'this (video|content) is (clearly |definitely |undoubtedly )?(hateful|offensive|promoting hate)',
        r'(hateful|offensive|hate speech)',
        r'contains? (hateful|offensive|hate)',
        r'classif\w* as (hateful|offensive)',
        r'(is|appears?) (hateful|offensive)',
    ]

    # Strong normal indicators
    normal_patterns = [
        r'this (video|content) is (not |neither )?(hateful|offensive)',
        r'(not |no |neither )(hateful|offensive|hate speech)',
        r'(normal|benign|harmless|non[-\s]?hat)',
        r'classif\w* as (normal|non[-\s]?hat)',
        r'does not (contain|promote|constitute)',
        r'no (evidence|indication|sign) of (hate|hatred|offensive)',
    ]

    # Score
    hateful_score = 0
    normal_score = 0

    for pattern in hateful_patterns:
        if re.search(pattern, hate_text):
            hateful_score += 1
    for pattern in normal_patterns:
        if re.search(pattern, hate_text):
            normal_score += 1

    # Handle negation: "not hateful" should count as normal
    if re.search(r'not\s+(hateful|offensive|promoting\s+hate)', hate_text):
        normal_score += 2
        hateful_score = max(0, hateful_score - 1)

    # Parse confidence
    confidence = 0.5  # default
    conf_text = (confidence_section + " " + hate_section).lower()
    conf_match = re.search(r'(\d+)\s*/\s*10|(\d+)\s*out of\s*10|confidence[:\s]+(\d+)', conf_text)
    if conf_match:
        val = int(next(g for g in conf_match.groups() if g is not None))
        confidence = min(val / 10.0, 1.0)
    elif "high" in conf_text:
        confidence = 0.8
    elif "moderate" in conf_text or "medium" in conf_text:
        confidence = 0.6
    elif "low" in conf_text:
        confidence = 0.4

    if hateful_score > normal_score:
        pred = "Hateful"
        pred_conf = confidence
    elif normal_score > hateful_score:
        pred = "Normal"
        pred_conf = confidence
    else:
        pred = "Unknown"
        pred_conf = 0.3

    return {
        "prediction": pred,
        "confidence": pred_conf,
        "hateful_score": hateful_score,
        "normal_score": normal_score,
    }


def main(args):
    for dataset in args.datasets:
        result_path = os.path.join(args.input_dir, dataset, "mllm_results.json")
        if not os.path.exists(result_path):
            print(f"Skipping {dataset}: no results at {result_path}")
            continue

        with open(result_path) as f:
            results = json.load(f)

        print(f"\n=== {dataset}: {len(results)} samples ===")

        # Parse classifications
        classifications = {}
        label_map_binary = {
            "Hate": 1, "Non Hate": 0, "Normal": 0,
            "Offensive": 1, "Hateful": 1,
        }

        correct = 0
        total = 0
        for vid, res in results.items():
            analysis = res.get("analysis", "")
            cls_result = parse_classification_from_analysis(analysis)
            classifications[vid] = cls_result

            # Check accuracy
            true_label = res.get("label", "")
            if true_label in label_map_binary:
                total += 1
                pred_binary = 1 if cls_result["prediction"] == "Hateful" else 0
                true_binary = label_map_binary[true_label]
                if pred_binary == true_binary:
                    correct += 1

        if total > 0:
            print(f"  MLLM classification accuracy: {correct}/{total} = {correct/total*100:.1f}%")

        # Save
        output_dir = os.path.join(args.output_dir, dataset)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "mllm_classifications.json")
        with open(output_path, "w") as f:
            json.dump(classifications, f, indent=2)
        print(f"  Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["HateMM", "MultiHateClip_CN", "MultiHateClip_EN"])
    parser.add_argument("--input_dir", default="/data/jehc223/EMNLP2/results/mllm")
    parser.add_argument("--output_dir", default="/data/jehc223/EMNLP2/results/mllm")
    args = parser.parse_args()
    main(args)
