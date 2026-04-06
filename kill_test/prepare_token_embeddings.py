"""
Extract token-level BERT hidden states for GTT (Grounded Token Trust).

For each video's 5 structured text fields, saves:
- unit_token_features.pth: {video_id: [5, T, 768]} token embeddings
- unit_token_masks.pth:    {video_id: [5, T]}      attention masks

Uses float16 on disk for memory efficiency.
"""
import json
import torch
from transformers import BertTokenizer, BertModel
from pathlib import Path

UNIT_FIELDS = [
    "content_summary",
    "target_analysis",
    "sentiment_tone",
    "harm_assessment",
    "overall_judgment",
]
K = len(UNIT_FIELDS)
MAX_LEN = 128
BERT_MODEL = "google-bert/bert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_token_embeddings(data_path: str, output_dir: str):
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    model = BertModel.from_pretrained(BERT_MODEL).to(DEVICE).eval()

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    token_features = {}
    token_masks = {}
    skipped = 0

    for i, item in enumerate(data):
        vid = item["Video_ID"]
        if "generic_response" not in item:
            skipped += 1
            continue
        resp = item["generic_response"]

        all_tokens = []
        all_masks = []

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
                # last_hidden_state: [1, T, 768]
                hidden = out.last_hidden_state.squeeze(0).cpu().half()  # [T, 768] fp16
                mask = enc["attention_mask"].squeeze(0).cpu()  # [T]

            all_tokens.append(hidden)
            all_masks.append(mask)

        token_features[vid] = torch.stack(all_tokens)  # [5, T, 768]
        token_masks[vid] = torch.stack(all_masks)  # [5, T]

        if (i + 1) % 100 == 0:
            print(f"  Extracted {i + 1}/{len(data)} videos")

    out_dir = Path(output_dir)
    feat_path = out_dir / "unit_token_features.pth"
    mask_path = out_dir / "unit_token_masks.pth"

    torch.save(token_features, feat_path)
    torch.save(token_masks, mask_path)

    print(f"Saved token features for {len(token_features)} videos to {feat_path}")
    print(f"Saved token masks for {len(token_masks)} videos to {mask_path}")
    print(f"Shape per video: tokens=[{K}, {MAX_LEN}, 768], masks=[{K}, {MAX_LEN}]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="/home/junyi/EMNLP2/datasets/HateMM/generic_data.json",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/junyi/EMNLP2/embeddings/HateMM",
    )
    args = parser.parse_args()
    extract_token_embeddings(args.data_path, args.output_dir)
