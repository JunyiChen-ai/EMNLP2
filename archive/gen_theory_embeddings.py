"""
Generate BERT [CLS] embeddings for theory rationale fields.

Usage:
  python gen_theory_embeddings.py --theory itt --dataset_name HateMM
  python gen_theory_embeddings.py --theory iet --dataset_name Multihateclip --language English
"""

import argparse, json, os
import torch
from transformers import AutoTokenizer, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

THEORY_FIELDS = {
    "generic": ["content_summary", "target_analysis", "sentiment_tone", "harm_assessment", "overall_judgment"],
    "itt": ["target_salience", "realistic_threat", "symbolic_threat", "anxiety_discomfort", "stereotype_support", "hostility_prescription"],
    "iet": ["group_framing", "appraisal_evidence", "emotion_inference", "action_tendency", "endorsement_stance"],
    "att": ["negative_outcome", "causal_attribution", "controllability", "responsibility_blame", "punitive_tendency"],
    "scm": ["target_group", "warmth_evidence", "competence_evidence", "social_perception", "behavioral_tendency"],
    "scm_v2": ["target_group", "warmth_evidence", "competence_evidence", "social_perception", "endorsement_context", "behavioral_tendency"],
}


def encode_texts(texts, tokenizer, model, max_length=256, pool="cls"):
    features = {}
    model.eval()
    with torch.no_grad():
        for vid, text in texts.items():
            if not text or text.strip() == "":
                text = "No content available."
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=max_length, padding=True).to(device)
            outputs = model(**inputs)
            if pool == "mean":
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                features[vid] = ((outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)).squeeze(0).cpu()
            else:
                features[vid] = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--theory", required=True, choices=["generic", "itt", "iet", "att", "scm", "scm_v2"])
    parser.add_argument("--dataset_name", default="HateMM", choices=["HateMM", "Multihateclip", "ImpliHateVid"])
    parser.add_argument("--language", default="English")
    parser.add_argument("--pool", default="mean", choices=["cls", "mean"])
    args = parser.parse_args()

    fields = THEORY_FIELDS[args.theory]

    if args.dataset_name == "HateMM":
        data_path = f"./datasets/HateMM/{args.theory}_data.json"
        out_dir = "./embeddings/HateMM"
    elif args.dataset_name == "ImpliHateVid":
        data_path = f"./datasets/ImpliHateVid/{args.theory}_data.json"
        out_dir = "./embeddings/ImpliHateVid"
    else:
        data_path = f"./datasets/Multihateclip/{args.language}/{args.theory}_data.json"
        out_dir = f"./embeddings/Multihateclip/{args.language}"

    os.makedirs(out_dir, exist_ok=True)

    with open(data_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} videos from {data_path}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()

    # Extract field texts
    field_texts = {field: {} for field in fields}
    all_rationale_texts = {}  # concatenation of all fields for pooled embedding

    resp_key = f"{args.theory}_response"
    for item in data:
        vid = item["Video_ID"]
        resp = item.get(resp_key, {})

        parts = []
        for field in fields:
            text = resp.get(field, "")
            field_texts[field][vid] = text
            if text:
                parts.append(text)

        all_rationale_texts[vid] = " ".join(parts) if parts else ""

    # Encode each field independently
    prefix = f"{args.theory}_mean" if args.pool == "mean" else args.theory
    field_feats = {}
    for field in fields:
        print(f"Encoding {prefix}_{field} (pool={args.pool})...")
        field_feats[field] = encode_texts(field_texts[field], tokenizer, model, max_length=256, pool=args.pool)

    # Encode pooled rationale (all fields concatenated)
    print(f"Encoding {args.theory}_rationale (pooled)...")
    rationale_feats = encode_texts(all_rationale_texts, tokenizer, model, max_length=512)

    # Save per-field embeddings
    for field in fields:
        save_path = os.path.join(out_dir, f"{prefix}_{field}_features.pth")
        torch.save(field_feats[field], save_path)
        print(f"  Saved {save_path} ({len(field_feats[field])} videos)")

    # Save pooled rationale embedding
    save_path = os.path.join(out_dir, f"{args.theory}_rationale_features.pth")
    torch.save(rationale_feats, save_path)
    print(f"  Saved {save_path} ({len(rationale_feats)} videos)")

    print(f"\nDone: {args.theory} embeddings for {args.dataset_name}")


if __name__ == "__main__":
    main()
