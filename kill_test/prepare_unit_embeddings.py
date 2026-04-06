"""
Step 0: Prepare per-unit BERT embeddings from MLLM rationale text.

Each rationale has 5 structured fields (content_summary, target_analysis,
sentiment_tone, harm_assessment, overall_judgment). We treat each field as
one "evidence unit" and encode it with BERT [CLS].

Output: {video_id: tensor of shape [5, 768]} saved to unit_features.pth
"""
import json
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from pathlib import Path

UNIT_FIELDS = [
    "content_summary",
    "target_analysis",
    "sentiment_tone",
    "harm_assessment",
    "overall_judgment",
]
K = len(UNIT_FIELDS)  # 5 units
MAX_LEN = 128
BERT_MODEL = "google-bert/bert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def encode_units(data_path: str, output_path: str):
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    model = BertModel.from_pretrained(BERT_MODEL).to(DEVICE).eval()

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    unit_features = {}
    skipped = 0
    for i, item in enumerate(data):
        vid = item["Video_ID"]
        if "generic_response" not in item:
            skipped += 1
            continue
        resp = item["generic_response"]

        units = []
        for field in UNIT_FIELDS:
            text = resp.get(field, "") or ""
            text = text.strip() or " "
            enc = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            with torch.no_grad():
                out = model(**{k: v.to(DEVICE) for k, v in enc.items()})
                cls = out.last_hidden_state[:, 0, :].squeeze(0)  # [768]
            units.append(cls.cpu())

        unit_features[vid] = torch.stack(units)  # [5, 768]

        if (i + 1) % 100 == 0:
            print(f"  Encoded {i + 1}/{len(data)} videos")

    torch.save(unit_features, output_path)
    print(f"Saved unit features for {len(unit_features)} videos to {output_path}")
    print(f"Shape per video: [{K}, 768]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="/home/junyi/EMNLP2/datasets/HateMM/generic_data.json",
    )
    parser.add_argument(
        "--output_path",
        default="/home/junyi/EMNLP2/embeddings/HateMM/unit_features.pth",
    )
    args = parser.parse_args()
    encode_units(args.data_path, args.output_path)
