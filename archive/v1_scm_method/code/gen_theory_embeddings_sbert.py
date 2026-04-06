"""
Generate sentence-transformer embeddings for SCM theory fields.

Round 014: Re-encode LLM text outputs using sentence-transformers
(trained on NLI/similarity tasks) instead of bert-base-uncased.

This produces more discriminative embeddings for the downstream
classification task without changing the LLM or re-querying.

Usage:
  python gen_theory_embeddings_sbert.py --dataset_name HateMM
  python gen_theory_embeddings_sbert.py --dataset_name Multihateclip --language English
  python gen_theory_embeddings_sbert.py --dataset_name Multihateclip --language Chinese
"""

import argparse, json, os
import torch
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

SCM_FIELDS = ["target_group", "warmth_evidence", "competence_evidence",
              "social_perception", "behavioral_tendency"]


def encode_field(model, texts_dict, batch_size=64):
    """Encode a dictionary of {vid: text} using sentence-transformer."""
    vids = list(texts_dict.keys())
    texts = [texts_dict[v] if texts_dict[v] and texts_dict[v].strip() else "No content available."
             for v in vids]

    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                              convert_to_tensor=True, device=device)
    features = {vid: emb.cpu() for vid, emb in zip(vids, embeddings)}
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="HateMM", choices=["HateMM", "Multihateclip"])
    parser.add_argument("--language", default="English")
    parser.add_argument("--model_name", default="all-mpnet-base-v2",
                        help="Sentence-transformer model name")
    args = parser.parse_args()

    if args.dataset_name == "HateMM":
        data_path = "./datasets/HateMM/scm_data.json"
        out_dir = "./embeddings/HateMM"
    else:
        data_path = f"./datasets/Multihateclip/{args.language}/scm_data.json"
        out_dir = f"./embeddings/Multihateclip/{args.language}"

    os.makedirs(out_dir, exist_ok=True)

    with open(data_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} videos from {data_path}")

    # Load sentence-transformer model
    model = SentenceTransformer(args.model_name, device=device)
    embed_dim = model.get_sentence_embedding_dimension()
    print(f"Model: {args.model_name}, dim={embed_dim}")

    # Extract field texts
    field_texts = {field: {} for field in SCM_FIELDS}
    resp_key = "scm_response"
    for item in data:
        vid = item["Video_ID"]
        resp = item.get(resp_key, {})
        for field in SCM_FIELDS:
            field_texts[field][vid] = resp.get(field, "")

    # Determine prefix for saving (short model name)
    model_short = args.model_name.replace("all-", "").replace("-v2", "2").replace("-", "")
    prefix = f"scm_sbert_{model_short}"

    # Encode each field
    for field in SCM_FIELDS:
        print(f"Encoding {prefix}_{field}...")
        feats = encode_field(model, field_texts[field])
        save_path = os.path.join(out_dir, f"{prefix}_{field}_features.pth")
        torch.save(feats, save_path)
        print(f"  Saved {save_path} ({len(feats)} videos, dim={embed_dim})")

    print(f"\nDone: SBERT embeddings for {args.dataset_name}")


if __name__ == "__main__":
    main()
