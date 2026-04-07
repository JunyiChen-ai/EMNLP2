"""Merge shard files into a single rationales.jsonl, deduplicating by video_id."""
import json
import os
import sys

dataset = sys.argv[1]
prompt_family = sys.argv[2] if len(sys.argv) > 2 else "diagnostic"
num_shards = int(sys.argv[3]) if len(sys.argv) > 3 else 2

outdir = f"rationales/{dataset}/{prompt_family}"
merged_path = os.path.join(outdir, "rationales.jsonl")

seen = set()
records = []

# Read existing merged file
if os.path.exists(merged_path):
    with open(merged_path) as f:
        for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                    if obj["video_id"] not in seen:
                        seen.add(obj["video_id"])
                        records.append(line.strip())
                except (json.JSONDecodeError, KeyError):
                    pass

# Read shard files
for shard_idx in range(num_shards):
    shard_path = os.path.join(outdir, f"rationales_shard{shard_idx}.jsonl")
    if not os.path.exists(shard_path):
        print(f"Shard {shard_idx} not found: {shard_path}")
        continue
    with open(shard_path) as f:
        for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                    if obj["video_id"] not in seen:
                        seen.add(obj["video_id"])
                        records.append(line.strip())
                except (json.JSONDecodeError, KeyError):
                    pass

with open(merged_path, "w") as f:
    for r in records:
        f.write(r + "\n")

print(f"Merged: {len(records)} unique records → {merged_path}")
