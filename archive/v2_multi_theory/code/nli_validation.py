"""
NLI-based external validation of SCM fields.
Uses DeBERTa-v3-large-mnli to check if MLLM-generated SCM fields
are internally consistent with SCM theory predictions.

This is EXTERNAL validation because the NLI model was not involved
in generating the SCM fields — it independently verifies them.
"""

import json, os, sys
import torch
import numpy as np
from transformers import pipeline

device = 0 if torch.cuda.is_available() else -1

DATASETS = [
    ("HateMM", "datasets/HateMM/scm_data.json"),
    ("MHClip-EN", "datasets/Multihateclip/English/scm_data.json"),
    ("MHClip-ZH", "datasets/Multihateclip/Chinese/scm_data.json"),
    ("ImpliHateVid", "datasets/ImpliHateVid/scm_data.json"),
]

# NLI hypotheses for each SCM dimension
WARMTH_HYPOTHESES = {
    "low": "The group is portrayed as cold, hostile, or threatening.",
    "high": "The group is portrayed as warm, friendly, or trustworthy.",
}

COMPETENCE_HYPOTHESES = {
    "low": "The group is portrayed as incompetent, foolish, or incapable.",
    "high": "The group is portrayed as competent, skilled, or capable.",
}

QUADRANT_HYPOTHESES = {
    "contempt": "The social perception expressed is contempt or disgust toward the group.",
    "envy": "The social perception expressed is envy or resentment toward the group.",
    "pity": "The social perception expressed is pity or condescension toward the group.",
    "admiration": "The social perception expressed is admiration or respect toward the group.",
}

HARM_HYPOTHESES = {
    "active_harm": "The video endorses or implies active harm, violence, or hostility toward the group.",
    "passive_exclusion": "The video endorses or implies passive exclusion or avoidance of the group.",
    "patronizing": "The video endorses or implies patronizing or condescending treatment of the group.",
    "respect": "The video endorses or implies respectful or positive treatment of the group.",
    "none": "The video does not endorse any particular treatment of the group.",
}


def classify_nli(nli_model, premise, hypotheses, batch_size=32):
    """Run NLI classification: which hypothesis is most entailed by the premise?"""
    pairs = [(premise, h) for h in hypotheses.values()]
    results = nli_model(pairs, batch_size=batch_size)

    scores = {}
    for label, result in zip(hypotheses.keys(), results):
        # Get entailment score
        entail_score = 0
        for item in result if isinstance(result, list) else [result]:
            if item["label"].lower() in ["entailment", "entail"]:
                entail_score = item["score"]
        scores[label] = entail_score

    best = max(scores, key=scores.get)
    return best, scores


def extract_mllm_quadrant(resp):
    """Extract the MLLM's stated quadrant from social_perception text."""
    sp = resp.get("social_perception", "").lower()
    for q in ["contempt", "envy", "pity", "admiration"]:
        if q in sp:
            return q
    return "other"


def extract_mllm_warmth(resp):
    """Extract warmth polarity from warmth_evidence."""
    text = resp.get("warmth_evidence", "").lower()
    # Check for explicit polarity statements
    cold_indicators = ["cold", "hostile", "threatening", "unfriendly", "low warmth",
                       "not warm", "antagonistic", "aggressive", "malicious"]
    warm_indicators = ["warm", "friendly", "trustworthy", "well-intentioned",
                       "high warmth", "positive", "supportive", "kind"]

    cold = sum(1 for w in cold_indicators if w in text)
    warm = sum(1 for w in warm_indicators if w in text)

    if cold > warm: return "low"
    if warm > cold: return "high"
    return "ambiguous"


def main():
    print("Loading NLI model (DeBERTa-v3-large-mnli)...")
    try:
        nli = pipeline("zero-shot-classification",
                       model="cross-encoder/nli-deberta-v3-large",
                       device=device)
    except Exception:
        # Fallback to base deberta with NLI head
        print("Trying alternative model...")
        nli = pipeline("zero-shot-classification",
                       model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                       device=device)

    print("Model loaded.\n")

    all_results = {}

    for ds_name, ds_path in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        if not os.path.exists(ds_path):
            print(f"  Skipping: {ds_path} not found")
            continue

        data = json.load(open(ds_path))

        # Sample for efficiency — use all data if < 500, else sample 300
        if len(data) > 500:
            np.random.seed(42)
            indices = np.random.choice(len(data), 300, replace=False)
            sample = [data[i] for i in indices]
        else:
            sample = data

        warmth_agreement = 0
        warmth_total = 0
        quadrant_agreement = 0
        quadrant_total = 0
        warmth_nli_labels = []
        warmth_mllm_labels = []

        for i, item in enumerate(sample):
            resp = item.get("scm_response", {})
            warmth_text = resp.get("warmth_evidence", "")
            sp_text = resp.get("social_perception", "")

            if not warmth_text or not sp_text:
                continue

            # 1. Validate warmth dimension via NLI
            try:
                result = nli(warmth_text,
                           candidate_labels=["cold, hostile, or threatening",
                                           "warm, friendly, or trustworthy"],
                           hypothesis_template="The group described is {}.")
                nli_warmth = "low" if result["labels"][0].startswith("cold") else "high"
                mllm_warmth = extract_mllm_warmth(resp)

                if mllm_warmth != "ambiguous":
                    warmth_total += 1
                    warmth_nli_labels.append(nli_warmth)
                    warmth_mllm_labels.append(mllm_warmth)
                    if nli_warmth == mllm_warmth:
                        warmth_agreement += 1
            except Exception as e:
                pass

            # 2. Validate quadrant via NLI on social_perception
            mllm_quadrant = extract_mllm_quadrant(resp)
            if mllm_quadrant != "other":
                try:
                    result = nli(sp_text,
                               candidate_labels=["contempt or disgust",
                                                "envy or resentment",
                                                "pity or condescension",
                                                "admiration or respect"],
                               hypothesis_template="The social perception expressed is {}.")
                    label_map = {
                        "contempt or disgust": "contempt",
                        "envy or resentment": "envy",
                        "pity or condescension": "pity",
                        "admiration or respect": "admiration",
                    }
                    nli_quadrant = label_map[result["labels"][0]]
                    quadrant_total += 1
                    if nli_quadrant == mllm_quadrant:
                        quadrant_agreement += 1
                except Exception:
                    pass

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(sample)}...")

        warmth_acc = warmth_agreement / warmth_total * 100 if warmth_total > 0 else 0
        quadrant_acc = quadrant_agreement / quadrant_total * 100 if quadrant_total > 0 else 0

        # Compute Cohen's kappa for warmth
        if warmth_total >= 10:
            from sklearn.metrics import cohen_kappa_score
            kappa = cohen_kappa_score(warmth_nli_labels, warmth_mllm_labels)
        else:
            kappa = 0

        print(f"\n  Warmth validation: {warmth_agreement}/{warmth_total} = {warmth_acc:.1f}% agreement (κ={kappa:.3f})")
        print(f"  Quadrant validation: {quadrant_agreement}/{quadrant_total} = {quadrant_acc:.1f}% agreement")

        all_results[ds_name] = {
            "warmth_agreement": warmth_acc,
            "warmth_kappa": kappa,
            "warmth_n": warmth_total,
            "quadrant_agreement": quadrant_acc,
            "quadrant_n": quadrant_total,
        }

    # Summary
    print(f"\n{'='*60}")
    print("NLI VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<15} {'Warmth Agree':>15} {'Warmth κ':>10} {'Quadrant Agree':>15}")
    print("-" * 60)
    for ds, r in all_results.items():
        print(f"{ds:<15} {r['warmth_agreement']:>13.1f}% {r['warmth_kappa']:>10.3f} {r['quadrant_agreement']:>13.1f}%")

    with open("nli_validation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to nli_validation_results.json")


if __name__ == "__main__":
    main()
