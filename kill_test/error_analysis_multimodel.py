"""
Multi-model error pattern analysis.

Models:
  1. text_only_mlp: generic rationale [CLS] → MLP (text only)
  2. text_av_mlp: generic rationale [CLS] + audio + frame → MLP (full modality, simple)
  3. scm_moe: SCM-QMoE-QELS (archive model, full modality + theory features)

For each model, run 10 seeds and track per-sample correctness.
Then compare: which errors are shared, which are model-specific?
"""
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Simple Models ───────────────────────────────────────────

class TextOnlyMLP(nn.Module):
    def __init__(self, dim=768, hidden=256, nc=2, drop=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(drop), nn.Linear(hidden, nc))
    def forward(self, batch):
        return self.mlp(batch["text"])

class TextAVMLP(nn.Module):
    def __init__(self, dim=768, hidden=256, nc=2, drop=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 3, hidden), nn.ReLU(), nn.Dropout(drop), nn.Linear(hidden, nc))
    def forward(self, batch):
        x = torch.cat([batch["text"], batch["audio"], batch["frame"]], dim=-1)
        return self.mlp(x)


# ─── Dataset ─────────────────────────────────────────────────

LABEL_MAP = {"Non Hate": 0, "Hate": 1}
SCM_FIELDS = ["target_group", "warmth_evidence", "competence_evidence",
              "social_perception", "behavioral_tendency"]


class SimpleDS(Dataset):
    def __init__(self, vids, feats, labels):
        valid = set(vids) & set(labels.keys())
        for v in feats.values():
            valid &= set(v.keys())
        self.vids = [v for v in vids if v in valid]
        self.feats = feats
        self.labels = labels

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, idx):
        vid = self.vids[idx]
        out = {k: self.feats[k][vid] for k in self.feats}
        out["label"] = self.labels[vid]
        out["video_id"] = vid
        return out


def simple_collate(batch):
    result = {}
    for k in batch[0]:
        if k == "video_id":
            result[k] = [b[k] for b in batch]
        elif k == "label":
            result[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
        else:
            result[k] = torch.stack([b[k] for b in batch])
    return result


# ─── SCM-MoE Dataset (needs labels dict format) ─────────────

class SCMDS(Dataset):
    def __init__(self, vids, feats, lm, mk):
        self.vids = vids
        self.f = feats
        self.lm = lm
        self.mk = mk

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, i):
        v = self.vids[i]
        out = {k: self.f[k][v] for k in self.mk}
        out["label"] = torch.tensor(self.lm[self.f["labels"][v]["Label"]], dtype=torch.long)
        out["video_id"] = v
        return out


def scm_collate(b):
    result = {}
    for k in b[0]:
        if k == "video_id":
            result[k] = [x[k] for x in b]
        else:
            result[k] = torch.stack([x[k] for x in b])
    return result


# ─── Training ────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_simple(model, train_dl, valid_dl, epochs=50, lr=2e-4):
    model = model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    cw = torch.tensor([1.0, 1.5], dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)
    total = epochs * len(train_dl)
    warmup = 5 * len(train_dl)
    sch = optim.lr_scheduler.LambdaLR(opt, lambda s: max(1e-2, s / max(1, warmup))
                                       if s < warmup else max(0, 0.5 * (1 + np.cos(np.pi * (s - warmup) / max(1, total - warmup)))))
    best_va, best_st, no_imp = -1, None, 0
    for ep in range(epochs):
        model.train()
        for batch in train_dl:
            batch_g = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            opt.zero_grad()
            loss = criterion(model(batch_g), batch_g["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
        model.eval()
        ps, ls = [], []
        with torch.no_grad():
            for batch in valid_dl:
                batch_g = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                ps.extend(model(batch_g).argmax(1).cpu().numpy())
                ls.extend(batch_g["label"].cpu().numpy())
        va = accuracy_score(ls, ps)
        if va > best_va:
            best_va = va
            best_st = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= 10:
                break
    model.load_state_dict(best_st)
    return model


def train_scm(model, train_dl, valid_dl, epochs=45):
    """Train SCM-MoE with EMA and QELS loss, matching archive code."""
    import copy
    model = model.to(DEVICE)
    ema = copy.deepcopy(model)
    opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
    cw = torch.tensor([1.0, 1.5], dtype=torch.float).to(DEVICE)
    total = epochs * len(train_dl)
    warmup = 5 * len(train_dl)
    sch = optim.lr_scheduler.LambdaLR(opt, lambda s: s / max(1, warmup)
                                       if s < warmup else max(0, 0.5 * (1 + np.cos(np.pi * (s - warmup) / max(1, total - warmup)))))
    best_va, best_st = -1, None
    for ep in range(epochs):
        model.train()
        for batch in train_dl:
            batch_g = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            opt.zero_grad()
            logits, q_ent = model(batch_g, training=True, return_qels=True)
            loss = qels_cross_entropy(logits, batch_g["label"], q_ent, nc=2,
                                       eps_min=0.01, eps_lambda=0.15, class_weight=cw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            with torch.no_grad():
                for p, ep2 in zip(model.parameters(), ema.parameters()):
                    ep2.data.mul_(0.999).add_(p.data, alpha=0.001)
        ema.eval()
        ps, ls = [], []
        with torch.no_grad():
            for batch in valid_dl:
                batch_g = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                ps.extend(ema(batch_g).argmax(1).cpu().numpy())
                ls.extend(batch_g["label"].cpu().numpy())
        va = accuracy_score(ls, ps)
        if va > best_va:
            best_va = va
            best_st = {k: v.clone() for k, v in ema.state_dict().items()}
    ema.load_state_dict(best_st)
    return ema


def predict(model, dl):
    model.eval()
    results = {}
    with torch.no_grad():
        for batch in dl:
            batch_g = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = model(batch_g)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(1).cpu().numpy()
            labels = batch["label"].numpy()
            for i, vid in enumerate(batch["video_id"]):
                results[vid] = {
                    "pred": int(preds[i]),
                    "prob_hate": float(probs[i, 1].cpu()),
                    "label": int(labels[i]),
                    "correct": int(preds[i]) == int(labels[i]),
                }
    return results


# ─── Main ────────────────────────────────────────────────────

def main():
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dir", default="/home/junyi/EMNLP2/embeddings/HateMM")
    parser.add_argument("--ann_path", default="/home/junyi/EMNLP2/datasets/HateMM/annotation(new).json")
    parser.add_argument("--split_dir", default="/home/junyi/EMNLP2/datasets/HateMM/splits")
    parser.add_argument("--data_path", default="/home/junyi/EMNLP2/datasets/HateMM/generic_data.json")
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--output_dir", default="/home/junyi/EMNLP2/kill_test/results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load splits
    splits = {}
    for name in ["train", "valid", "test"]:
        df = pd.read_csv(os.path.join(args.split_dir, f"{name}.csv"), header=None)
        splits[name] = df.iloc[:, 0].tolist()

    # Load labels
    with open(args.ann_path) as f:
        ann_data = json.load(f)
    labels_int = {d["Video_ID"]: LABEL_MAP[d["Label"]] for d in ann_data}
    labels_raw = {d["Video_ID"]: d for d in ann_data}

    # Load raw rationale text
    with open(args.data_path) as f:
        raw_data = {d["Video_ID"]: d for d in json.load(f)}

    emb = args.emb_dir

    # ─── Model 1 & 2: Simple features ───
    simple_feats = {
        "text": torch.load(f"{emb}/generic_rationale_features.pth", map_location="cpu"),
        "audio": torch.load(f"{emb}/wavlm_audio_features.pth", map_location="cpu"),
        "frame": torch.load(f"{emb}/frame_features.pth", map_location="cpu"),
    }
    simple_train = SimpleDS(splits["train"], simple_feats, labels_int)
    simple_valid = SimpleDS(splits["valid"], simple_feats, labels_int)
    simple_test = SimpleDS(splits["test"], simple_feats, labels_int)
    s_train_dl = DataLoader(simple_train, 32, True, collate_fn=simple_collate)
    s_valid_dl = DataLoader(simple_valid, 64, False, collate_fn=simple_collate)
    s_test_dl = DataLoader(simple_test, 64, False, collate_fn=simple_collate)

    # ─── Model 3: SCM-MoE features ───
    sys.path.insert(0, "/home/junyi/EMNLP2/archive/v1_scm_method/code")
    from main_scm_qmoe_qels import SCMQMoEQELS, qels_cross_entropy  # noqa

    # Make qels_cross_entropy available in train_scm scope
    globals()["qels_cross_entropy"] = qels_cross_entropy

    base_mk = ["text", "audio", "frame"]
    scm_mk = base_mk + [f"scm_{f}" for f in SCM_FIELDS]
    scm_feats = {
        "text": torch.load(f"{emb}/text_features.pth", map_location="cpu"),
        "audio": torch.load(f"{emb}/wavlm_audio_features.pth", map_location="cpu"),
        "frame": torch.load(f"{emb}/frame_features.pth", map_location="cpu"),
        "labels": labels_raw,
    }
    for f in SCM_FIELDS:
        scm_feats[f"scm_{f}"] = torch.load(f"{emb}/scm_mean_{f}_features.pth", map_location="cpu")

    lm = {"Non Hate": 0, "Hate": 1}
    common = set.intersection(*[set(scm_feats[k].keys()) for k in scm_mk]) & set(labels_raw.keys())
    scm_splits = {s: [v for v in splits[s] if v in common] for s in splits}
    scm_train_dl = DataLoader(SCMDS(scm_splits["train"], scm_feats, lm, scm_mk), 32, True, collate_fn=scm_collate)
    scm_valid_dl = DataLoader(SCMDS(scm_splits["valid"], scm_feats, lm, scm_mk), 64, False, collate_fn=scm_collate)
    scm_test_dl = DataLoader(SCMDS(scm_splits["test"], scm_feats, lm, scm_mk), 64, False, collate_fn=scm_collate)

    print(f"Simple test: {len(simple_test)}, SCM test: {len(scm_splits['test'])}")

    seeds = list(range(42, 42 + args.num_seeds))
    models_config = [
        ("text_only_mlp", "simple"),
        ("text_av_mlp", "simple"),
        ("scm_moe", "scm"),
    ]

    all_results = {}  # model_name → seed → {vid: result}

    for model_name, model_type in models_config:
        all_results[model_name] = {}
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        for seed in seeds:
            set_seed(seed)
            print(f"  Seed {seed}...", end="", flush=True)

            if model_name == "text_only_mlp":
                model = TextOnlyMLP()
                model = train_simple(model, s_train_dl, s_valid_dl)
                results = predict(model, s_test_dl)
            elif model_name == "text_av_mlp":
                model = TextAVMLP()
                model = train_simple(model, s_train_dl, s_valid_dl)
                results = predict(model, s_test_dl)
            elif model_name == "scm_moe":
                model = SCMQMoEQELS(base_mk, nc=2)
                model = train_scm(model, scm_train_dl, scm_valid_dl)
                results = predict(model, scm_test_dl)

            acc = sum(r["correct"] for r in results.values()) / len(results)
            print(f" ACC={acc*100:.2f}")
            all_results[model_name][seed] = results
            del model
            torch.cuda.empty_cache()

    # ─── Compare error patterns ─────────────────────────────
    # Use test vids common to all models
    test_vids_sets = [set(all_results[m][seeds[0]].keys()) for m in all_results]
    common_test = sorted(set.intersection(*test_vids_sets))
    print(f"\nCommon test samples: {len(common_test)}")

    # Per-model summary
    print(f"\n{'='*60}")
    print(f"PER-MODEL ACCURACY SUMMARY ({args.num_seeds} seeds)")
    print(f"{'='*60}")
    for model_name in all_results:
        accs = []
        for seed in seeds:
            res = all_results[model_name][seed]
            acc = sum(res[v]["correct"] for v in common_test) / len(common_test)
            accs.append(acc * 100)
        print(f"  {model_name:20s}: mean={np.mean(accs):.2f}, std={np.std(accs):.2f}, "
              f"worst={np.min(accs):.2f}, best={np.max(accs):.2f}")

    # Per-sample correctness count
    def get_correct_count(model_name):
        counts = {}
        for vid in common_test:
            counts[vid] = sum(all_results[model_name][s][vid]["correct"] for s in seeds)
        return counts

    cc = {m: get_correct_count(m) for m in all_results}

    # Categorize per model
    def categorize(counts, n_seeds):
        return {
            "always_wrong": [v for v, c in counts.items() if c == 0],
            "mostly_wrong": [v for v, c in counts.items() if 0 < c <= 2],
            "unstable": [v for v, c in counts.items() if 3 <= c <= n_seeds - 3],
            "mostly_correct": [v for v, c in counts.items() if n_seeds - 2 <= c < n_seeds],
            "always_correct": [v for v, c in counts.items() if c == n_seeds],
        }

    cats = {m: categorize(cc[m], len(seeds)) for m in all_results}

    print(f"\n{'='*60}")
    print(f"ERROR CATEGORY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Category':20s} {'text_only':>12s} {'text_av':>12s} {'scm_moe':>12s}")
    for cat in ["always_correct", "mostly_correct", "unstable", "mostly_wrong", "always_wrong"]:
        vals = [len(cats[m][cat]) for m in all_results]
        print(f"  {cat:20s} {vals[0]:>10d}   {vals[1]:>10d}   {vals[2]:>10d}")

    # Shared always-wrong
    aw_sets = {m: set(cats[m]["always_wrong"]) for m in all_results}
    shared_aw = aw_sets["text_only_mlp"] & aw_sets["text_av_mlp"] & aw_sets["scm_moe"]
    text_only_aw = aw_sets["text_only_mlp"] - aw_sets["text_av_mlp"] - aw_sets["scm_moe"]
    av_fixed = aw_sets["text_only_mlp"] - aw_sets["text_av_mlp"]
    scm_fixed = aw_sets["text_only_mlp"] - aw_sets["scm_moe"]

    print(f"\n  Always-wrong shared across ALL 3 models: {len(shared_aw)}")
    print(f"  Always-wrong in text_only but NOT text_av (AV fixes): {len(av_fixed)}")
    print(f"  Always-wrong in text_only but NOT scm_moe (SCM fixes): {len(scm_fixed)}")
    print(f"  Always-wrong ONLY in text_only: {len(text_only_aw)}")

    # ─── Detailed error analysis for shared always-wrong ─────
    print(f"\n{'='*60}")
    print(f"SHARED ALWAYS-WRONG CASES ({len(shared_aw)} samples)")
    print(f"{'='*60}")

    for vid in sorted(shared_aw):
        if vid not in raw_data:
            continue
        entry = raw_data[vid]
        resp = entry.get("generic_response", {})
        label = "HATE" if labels_int[vid] == 1 else "NON-HATE"

        avg_probs = {m: np.mean([all_results[m][s][vid]["prob_hate"] for s in seeds]) for m in all_results}

        print(f"\n  [{vid}] Label={label}")
        print(f"    Avg P(hate): text_only={avg_probs['text_only_mlp']:.3f}, "
              f"text_av={avg_probs['text_av_mlp']:.3f}, scm_moe={avg_probs['scm_moe']:.3f}")
        print(f"    Transcript: {(entry.get('Transcript', '') or '')[:200]}...")
        print(f"    Content: {(resp.get('content_summary', '') or '')[:200]}...")
        print(f"    Judgment: {(resp.get('overall_judgment', '') or '')[:200]}...")

    # ─── Cases where AV helps ───────────────────────────────
    print(f"\n{'='*60}")
    print(f"CASES WHERE AV FIXES TEXT-ONLY ERRORS ({len(av_fixed)} samples)")
    print(f"{'='*60}")

    for vid in sorted(av_fixed):
        if vid not in raw_data:
            continue
        entry = raw_data[vid]
        resp = entry.get("generic_response", {})
        label = "HATE" if labels_int[vid] == 1 else "NON-HATE"
        text_cc = cc["text_only_mlp"][vid]
        av_cc = cc["text_av_mlp"][vid]
        scm_cc = cc["scm_moe"][vid]

        print(f"\n  [{vid}] Label={label}, Correct: text={text_cc}/10, av={av_cc}/10, scm={scm_cc}/10")
        print(f"    Transcript: {(entry.get('Transcript', '') or '')[:200]}...")
        print(f"    Judgment: {(resp.get('overall_judgment', '') or '')[:150]}...")

    # ─── Cases where SCM-MoE helps but simple AV doesn't ────
    scm_unique_fix = (aw_sets["text_only_mlp"] & aw_sets["text_av_mlp"]) - aw_sets["scm_moe"]
    print(f"\n{'='*60}")
    print(f"CASES WHERE SCM-MOE FIXES BUT SIMPLE AV DOESN'T ({len(scm_unique_fix)} samples)")
    print(f"{'='*60}")

    for vid in sorted(scm_unique_fix):
        if vid not in raw_data:
            continue
        entry = raw_data[vid]
        resp = entry.get("generic_response", {})
        label = "HATE" if labels_int[vid] == 1 else "NON-HATE"
        text_cc = cc["text_only_mlp"][vid]
        av_cc = cc["text_av_mlp"][vid]
        scm_cc = cc["scm_moe"][vid]

        print(f"\n  [{vid}] Label={label}, Correct: text={text_cc}/10, av={av_cc}/10, scm={scm_cc}/10")
        print(f"    Transcript: {(entry.get('Transcript', '') or '')[:200]}...")
        print(f"    Judgment: {(resp.get('overall_judgment', '') or '')[:150]}...")

    # ─── FP vs FN breakdown per model ───────────────────────
    print(f"\n{'='*60}")
    print(f"FP vs FN IN ALWAYS-WRONG (per model)")
    print(f"{'='*60}")
    for model_name in all_results:
        aw = cats[model_name]["always_wrong"]
        if not aw:
            continue
        fp = sum(1 for v in aw if labels_int[v] == 0)
        fn = sum(1 for v in aw if labels_int[v] == 1)
        print(f"  {model_name:20s}: {len(aw)} always-wrong = {fp} FP + {fn} FN")

    # ─── MLLM judgment alignment for always-wrong ────────────
    print(f"\n{'='*60}")
    print(f"MLLM JUDGMENT × ERROR PATTERN (per model)")
    print(f"{'='*60}")
    for model_name in all_results:
        aw = cats[model_name]["always_wrong"]
        if not aw:
            continue
        mllm_wrong = 0
        for vid in aw:
            if vid not in raw_data:
                continue
            resp = raw_data[vid].get("generic_response", {})
            oj = (resp.get("overall_judgment", "") or "").lower()
            label = labels_int[vid]
            mllm_says_hate = "yes" in oj[:20] or ("hateful" in oj[:50] and "not hateful" not in oj[:50])
            mllm_says_not_hate = "no" in oj[:5] or "not hateful" in oj[:50]

            if mllm_says_hate and label == 0:
                mllm_wrong += 1
            elif mllm_says_not_hate and label == 1:
                mllm_wrong += 1
        print(f"  {model_name:20s}: {mllm_wrong}/{len(aw)} always-wrong have MLLM judgment ≠ ground truth")

    # ─── Save results ────────────────────────────────────────
    output = {
        "per_model_acc": {},
        "per_sample": {},
    }
    for model_name in all_results:
        accs = []
        for seed in seeds:
            res = all_results[model_name][seed]
            acc = sum(res[v]["correct"] for v in common_test) / len(common_test)
            accs.append(acc * 100)
        output["per_model_acc"][model_name] = {
            "mean": round(np.mean(accs), 2),
            "std": round(np.std(accs), 2),
            "worst": round(np.min(accs), 2),
            "best": round(np.max(accs), 2),
        }

    for vid in common_test:
        output["per_sample"][vid] = {
            "label": labels_int[vid],
        }
        for model_name in all_results:
            output["per_sample"][vid][f"{model_name}_correct"] = cc[model_name][vid]
            output["per_sample"][vid][f"{model_name}_avg_prob"] = float(
                np.mean([all_results[model_name][s][vid]["prob_hate"] for s in seeds])
            )

    out_path = os.path.join(args.output_dir, "error_analysis_multimodel.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
