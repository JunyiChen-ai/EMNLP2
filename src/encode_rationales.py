#!/usr/bin/env python3
"""Encode MLLM rationales into embeddings using a text encoder.

Uses sentence-transformers (BGE or similar) to encode the structured
MLLM analysis into dense feature vectors for the downstream classifier.
"""

import argparse
import json
import os
import torch
from transformers import AutoTokenizer, AutoModel


def mean_pooling(model_output, attention_mask):
    """Mean pooling of token embeddings."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def encode_texts(texts, tokenizer, model, device, batch_size=32, max_length=512):
    """Encode a list of texts into embeddings."""
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt").to(device)
            outputs = model(**encoded)
            embeddings = mean_pooling(outputs, encoded["attention_mask"])
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use BERT-base to match HVGuard's embedding dimension (768)
    model_name = args.encoder
    print(f"Loading encoder: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    for dataset in args.datasets:
        result_path = os.path.join(args.input_dir, dataset, "mllm_results.json")
        if not os.path.exists(result_path):
            print(f"Skipping {dataset}: no results found at {result_path}")
            continue

        with open(result_path) as f:
            results = json.load(f)

        print(f"\nEncoding {dataset}: {len(results)} samples")

        # Prepare texts
        video_ids = list(results.keys())
        texts = []
        for vid in video_ids:
            analysis = results[vid].get("analysis", "")
            if analysis.startswith("ERROR"):
                # Use transcript as fallback
                analysis = results[vid].get("transcript", "")
            texts.append(analysis)

        # Encode
        embeddings = encode_texts(texts, tokenizer, model, device,
                                  batch_size=args.batch_size, max_length=args.max_length)

        # Save as dict {video_id: embedding_tensor}
        embedding_dict = {}
        for vid, emb in zip(video_ids, embeddings):
            embedding_dict[vid] = emb

        output_dir = os.path.join(args.output_dir, dataset)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "mllm_rationale_features.pth")
        torch.save(embedding_dict, output_path)
        print(f"  Saved {len(embedding_dict)} embeddings to {output_path}")
        print(f"  Embedding dim: {embeddings.shape[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["HateMM", "MultiHateClip_CN", "MultiHateClip_EN"])
    parser.add_argument("--input_dir", default="/data/jehc223/EMNLP2/results/mllm")
    parser.add_argument("--output_dir", default="/data/jehc223/EMNLP2/embeddings")
    parser.add_argument("--encoder", default="bert-base-uncased")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()
    main(args)
