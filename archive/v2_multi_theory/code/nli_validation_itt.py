"""
NLI-based external validation of ITT (Intergroup Threat Theory) fields.
Uses DeBERTa-v3-large to check if MLLM-generated ITT fields are meaningful.
"""

import json, os, torch, numpy as np
from transformers import pipeline

device = 0 if torch.cuda.is_available() else -1

DATASETS = [
    ("HateMM", "datasets/HateMM/itt_data.json"),
    ("MHClip-EN", "datasets/Multihateclip/English/itt_data.json"),
    ("MHClip-ZH", "datasets/Multihateclip/Chinese/itt_data.json"),
]

ITT_FIELDS = ["target_salience", "realistic_threat", "symbolic_threat",
              "anxiety_discomfort", "stereotype_support", "hostility_prescription"]

# For each ITT field, define NLI validation hypotheses
FIELD_HYPOTHESES = {
    "realistic_threat": {
        "present": "threats to safety, resources, or economic well-being",
        "absent": "no threats to safety, resources, or economic well-being",
    },
    "symbolic_threat": {
        "present": "threats to cultural values, beliefs, or identity",
        "absent": "no threats to cultural values, beliefs, or identity",
    },
    "anxiety_discomfort": {
        "high": "anxiety, discomfort, or unease",
        "low": "calm, comfort, or ease",
    },
    "hostility_prescription": {
        "hostile": "hostility, aggression, or punitive action",
        "neutral": "tolerance, acceptance, or peaceful coexistence",
    },
}


def main():
    print("Loading NLI model...")
    nli = pipeline("zero-shot-classification",
                   model="cross-encoder/nli-deberta-v3-large",
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
        labels_path = ds_path.replace("itt_data.json", "annotation(new).json")
        labels_data = json.load(open(labels_path))
        vid_to_label = {}
        for item in labels_data:
            vid_to_label[item["Video_ID"]] = item.get("Label", "")

        if len(data) > 500:
            np.random.seed(42)
            indices = np.random.choice(len(data), 300, replace=False)
            sample = [data[i] for i in indices]
        else:
            sample = data

        field_results = {}

        for field, hyps in FIELD_HYPOTHESES.items():
            labels_a = list(hyps.values())
            pos_label = labels_a[0]  # first is the "present/high/hostile" direction

            agree_with_hate = 0  # field indicates threat AND video is hateful
            disagree_with_hate = 0
            total_hateful = 0
            total_nonhateful = 0
            field_positive = 0
            field_total = 0

            for item in sample:
                resp = item.get("itt_response", {})
                text = resp.get(field, "")
                if not text or len(text) < 10:
                    continue

                vid = item["Video_ID"]
                label = vid_to_label.get(vid, "")
                is_hateful = label in ["Hate", "Hateful", "Offensive"]

                try:
                    result = nli(text, candidate_labels=labels_a,
                               hypothesis_template="This text describes {}.")
                    nli_positive = result["labels"][0] == pos_label
                    field_total += 1

                    if nli_positive:
                        field_positive += 1

                    if is_hateful:
                        total_hateful += 1
                        if nli_positive:
                            agree_with_hate += 1
                    else:
                        total_nonhateful += 1
                        if nli_positive:
                            disagree_with_hate += 1
                except Exception:
                    continue

            # Compute field-label correlation
            if total_hateful > 0 and total_nonhateful > 0 and field_total > 0:
                hate_positive_rate = agree_with_hate / total_hateful * 100
                nonhate_positive_rate = disagree_with_hate / total_nonhateful * 100
                overall_positive_rate = field_positive / field_total * 100

                field_results[field] = {
                    "overall_positive_rate": overall_positive_rate,
                    "hateful_positive_rate": hate_positive_rate,
                    "nonhateful_positive_rate": nonhate_positive_rate,
                    "discrimination": hate_positive_rate - nonhate_positive_rate,
                    "n": field_total,
                }
                print(f"  {field:25s}: hateful→positive={hate_positive_rate:.1f}%, non-hate→positive={nonhate_positive_rate:.1f}%, Δ={hate_positive_rate - nonhate_positive_rate:+.1f}pp")

        all_results[ds_name] = field_results

    # Summary
    print(f"\n{'='*60}")
    print("ITT NLI VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Field':<25} {'HateMM Δ':>10} {'MHC-EN Δ':>10} {'MHC-ZH Δ':>10}")
    print("-" * 60)
    for field in FIELD_HYPOTHESES:
        row = f"{field:<25}"
        for ds in ["HateMM", "MHClip-EN", "MHClip-ZH"]:
            if ds in all_results and field in all_results[ds]:
                d = all_results[ds][field]["discrimination"]
                row += f" {d:>+9.1f}pp"
            else:
                row += f" {'N/A':>10}"
        print(row)

    print("\n(Δ = hateful_positive_rate − nonhateful_positive_rate)")
    print("Positive Δ means field correctly discriminates hateful from non-hateful content")

    with open("nli_validation_itt_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to nli_validation_itt_results.json")


if __name__ == "__main__":
    main()
