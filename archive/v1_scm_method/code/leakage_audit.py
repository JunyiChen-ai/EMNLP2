"""
Leakage audit: test if individual SCM fields can predict the hate label alone.
Uses logistic regression on BERT embeddings of each field independently.
Also tests direct MLLM classification from generic prompt's overall_judgment.

Usage:
  python leakage_audit.py
"""

import json, os, re
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

SCM_FIELDS = ["target_group", "warmth_evidence", "competence_evidence",
              "social_perception", "behavioral_tendency"]

DATASETS = [
    ("HateMM", "./embeddings/HateMM", "./datasets/HateMM/annotation(new).json",
     "./datasets/HateMM/splits", {"Non Hate": 0, "Hate": 1},
     "./datasets/HateMM/generic_data.json"),
    ("MHClip-EN", "./embeddings/Multihateclip/English", "./datasets/Multihateclip/English/annotation(new).json",
     "./datasets/Multihateclip/English/splits", {"Normal": 0, "Offensive": 1, "Hateful": 1},
     "./datasets/Multihateclip/English/generic_data.json"),
    ("MHClip-ZH", "./embeddings/Multihateclip/Chinese", "./datasets/Multihateclip/Chinese/annotation(new).json",
     "./datasets/Multihateclip/Chinese/splits", {"Normal": 0, "Offensive": 1, "Hateful": 1},
     "./datasets/Multihateclip/Chinese/generic_data.json"),
    ("ImpliHateVid", "./embeddings/ImpliHateVid", "./datasets/ImpliHateVid/annotation(new).json",
     "./datasets/ImpliHateVid/splits", {"Normal": 0, "Non Hate": 0, "Non-Hate": 0, "Hate": 1, "Hateful": 1},
     "./datasets/ImpliHateVid/generic_data.json"),
]


def load_split(split_dir, split_name):
    for ext in [".csv", ".txt"]:
        path = os.path.join(split_dir, f"{split_name}{ext}")
        if os.path.exists(path):
            with open(path) as f:
                return [line.strip() for line in f if line.strip()]
    return []


def load_labels(anno_path, label_map):
    with open(anno_path) as f:
        data = json.load(f)
    labels = {}
    for item in data:
        vid = item["Video_ID"]
        raw_label = item.get("Label", item.get("label", ""))
        if raw_label in label_map:
            labels[vid] = label_map[raw_label]
    return labels


def parse_mllm_prediction(judgment_text):
    """Parse Yes/No from overall_judgment field."""
    text = judgment_text.lower().strip()
    if text.startswith("yes") or "this is hateful" in text or "hateful content" in text:
        return 1
    elif text.startswith("no") or "not hateful" in text or "non-hateful" in text or "is not hate" in text:
        return 0
    # Ambiguous — check for keywords
    hate_keywords = ["hate", "hateful", "harmful", "offensive", "derogatory"]
    non_hate_keywords = ["not hateful", "non-hateful", "benign", "neutral", "harmless"]
    for kw in non_hate_keywords:
        if kw in text:
            return 0
    for kw in hate_keywords:
        if kw in text:
            return 1
    return -1  # cannot determine


def main():
    results = {}

    for ds_name, emb_dir, anno_path, split_dir, label_map, generic_path in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        labels = load_labels(anno_path, label_map)
        train_vids = load_split(split_dir, "train")
        test_vids = load_split(split_dir, "test")

        if not train_vids or not test_vids:
            print(f"  Skipping {ds_name}: missing splits")
            continue

        results[ds_name] = {}

        # ---- Part 1: Field-only logistic regression ----
        print(f"\n--- Field-Only Logistic Regression (Leakage Audit) ---")

        for field in SCM_FIELDS:
            feat_path = os.path.join(emb_dir, f"scm_mean_{field}_features.pth")
            if not os.path.exists(feat_path):
                print(f"  {field}: embedding not found at {feat_path}")
                continue

            feats = torch.load(feat_path, weights_only=False)

            X_train, y_train = [], []
            X_test, y_test = [], []

            for vid in train_vids:
                if vid in feats and vid in labels:
                    X_train.append(feats[vid].numpy())
                    y_train.append(labels[vid])
            for vid in test_vids:
                if vid in feats and vid in labels:
                    X_test.append(feats[vid].numpy())
                    y_test.append(labels[vid])

            if not X_train or not X_test:
                print(f"  {field}: insufficient data")
                continue

            X_train = np.array(X_train)
            y_train = np.array(y_train)
            X_test = np.array(X_test)
            y_test = np.array(y_test)

            clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)

            acc = accuracy_score(y_test, preds) * 100
            f1 = f1_score(y_test, preds, average="macro") * 100

            results[ds_name][f"field_only_{field}"] = {"acc": acc, "f1": f1}
            print(f"  {field:25s}: ACC={acc:.1f}%  M-F1={f1:.1f}%")

        # Also test all fields concatenated
        all_X_train, all_X_test = [], []
        for vid in train_vids:
            parts = []
            ok = True
            for field in SCM_FIELDS:
                feat_path = os.path.join(emb_dir, f"scm_mean_{field}_features.pth")
                if os.path.exists(feat_path):
                    feats = torch.load(feat_path, weights_only=False)
                    if vid in feats:
                        parts.append(feats[vid].numpy())
                    else:
                        ok = False; break
                else:
                    ok = False; break
            if ok and vid in labels:
                all_X_train.append(np.concatenate(parts))

        y_train_all = []
        for vid in train_vids:
            if vid in labels:
                y_train_all.append(labels[vid])

        for vid in test_vids:
            parts = []
            ok = True
            for field in SCM_FIELDS:
                feat_path = os.path.join(emb_dir, f"scm_mean_{field}_features.pth")
                if os.path.exists(feat_path):
                    feats = torch.load(feat_path, weights_only=False)
                    if vid in feats:
                        parts.append(feats[vid].numpy())
                    else:
                        ok = False; break
                else:
                    ok = False; break
            if ok and vid in labels:
                all_X_test.append(np.concatenate(parts))

        y_test_all = []
        for vid in test_vids:
            if vid in labels:
                y_test_all.append(labels[vid])

        if all_X_train and all_X_test:
            X_tr = np.array(all_X_train[:len(y_train_all)])
            y_tr = np.array(y_train_all[:len(all_X_train)])
            X_te = np.array(all_X_test[:len(y_test_all)])
            y_te = np.array(y_test_all[:len(all_X_test)])

            clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)
            acc = accuracy_score(y_te, preds) * 100
            f1 = f1_score(y_te, preds, average="macro") * 100
            results[ds_name]["field_all_concat_logreg"] = {"acc": acc, "f1": f1}
            print(f"  {'all_fields_concat (LR)':25s}: ACC={acc:.1f}%  M-F1={f1:.1f}%")

        # ---- Part 2: Direct MLLM Classification ----
        print(f"\n--- Direct MLLM Classification (overall_judgment) ---")

        if os.path.exists(generic_path):
            with open(generic_path) as f:
                generic_data = json.load(f)

            vid_to_judgment = {}
            for item in generic_data:
                vid = item["Video_ID"]
                resp = item.get("generic_response", {})
                judgment = resp.get("overall_judgment", "")
                vid_to_judgment[vid] = judgment

            correct, total = 0, 0
            y_true, y_pred = [], []
            undetermined = 0

            for vid in test_vids:
                if vid in vid_to_judgment and vid in labels:
                    pred = parse_mllm_prediction(vid_to_judgment[vid])
                    if pred == -1:
                        undetermined += 1
                        continue
                    y_true.append(labels[vid])
                    y_pred.append(pred)

            if y_true:
                acc = accuracy_score(y_true, y_pred) * 100
                f1 = f1_score(y_true, y_pred, average="macro") * 100
                results[ds_name]["mllm_direct"] = {"acc": acc, "f1": f1, "undetermined": undetermined}
                print(f"  MLLM direct (GPT-5.4-nano): ACC={acc:.1f}%  M-F1={f1:.1f}%  (undetermined: {undetermined})")
        else:
            print(f"  Generic data not found: {generic_path}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"\n{'Method':<30} {'HateMM':>10} {'MHClip-EN':>10} {'MHClip-ZH':>10} {'ImpliHateVid':>12}")
    print("-" * 75)

    for method_key in ["field_only_target_group", "field_only_warmth_evidence",
                       "field_only_competence_evidence", "field_only_social_perception",
                       "field_only_behavioral_tendency", "field_all_concat_logreg", "mllm_direct"]:
        label = method_key.replace("field_only_", "").replace("field_all_concat_logreg", "all_fields (LR)")
        label = label.replace("mllm_direct", "MLLM direct")
        row = f"{label:<30}"
        for ds in ["HateMM", "MHClip-EN", "MHClip-ZH", "ImpliHateVid"]:
            if ds in results and method_key in results[ds]:
                row += f" {results[ds][method_key]['f1']:>9.1f}%"
            else:
                row += f" {'N/A':>10}"
        print(row)

    # Save results
    out_path = "leakage_audit_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
