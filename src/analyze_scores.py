import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

CONFIGS = {
    "HateMM": {"label_file": "baseline/HVGuard/datasets/HateMM/data (base).json",
                "splits_dir": "/data/jehc223/HateMM/splits",
                "label_map": {"Hate": 1, "Non Hate": 0}},
    "MultiHateClip_CN": {"label_file": "baseline/HVGuard/datasets/Multihateclip/Chinese/data.json",
                          "splits_dir": "/data/jehc223/Multihateclip/Chinese/splits",
                          "label_map": {"Normal": 0, "Offensive": 1, "Hateful": 1}},
    "MultiHateClip_EN": {"label_file": "baseline/HVGuard/datasets/Multihateclip/English/data.json",
                          "splits_dir": "/data/jehc223/Multihateclip/English/splits",
                          "label_map": {"Normal": 0, "Offensive": 1, "Hateful": 1}},
}
SCORE_KEYS = ["hate_speech_score", "visual_hate_score", "cross_modal_score",
              "implicit_hate_score", "overall_hate_score", "confidence"]

import os
os.chdir("/data/jehc223/EMNLP2")

for ds in ["HateMM", "MultiHateClip_CN", "MultiHateClip_EN"]:
    cfg = CONFIGS[ds]
    with open("results/mllm/" + ds + "/mllm_scores.json") as f:
        scores = json.load(f)
    with open(cfg["label_file"]) as f:
        labels = {d["Video_ID"]: cfg["label_map"].get(d["Label"], -1) for d in json.load(f)}
    splits = {}
    for split in ["train", "valid", "test"]:
        sdir = cfg["splits_dir"]
        with open(sdir + "/" + split + ".csv") as f:
            splits[split] = [l.strip() for l in f if l.strip() in labels and labels[l.strip()] >= 0]

    correct = total = 0
    for vid in splits["test"]:
        if vid in scores:
            sc = scores[vid]["scores"]
            pred = 1 if sc.get("classification", "").lower() in ["hateful", "offensive"] else 0
            if pred == labels[vid]:
                correct += 1
            total += 1
    print(f"\n{ds}: MLLM direct = {correct}/{total} = {correct/total*100:.1f}%")

    X_train, y_train, X_test, y_test = [], [], [], []
    for split, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        for vid in splits[split]:
            if vid in scores:
                sc = scores[vid]["scores"]
                feats = [sc.get(k, 5) / 10.0 for k in SCORE_KEYS]
                feats.append(1 if sc.get("classification", "").lower() in ["hateful", "offensive"] else 0)
                X.append(feats)
                y.append(labels[vid])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)
    print(f"  Score LogReg: {accuracy_score(y_test, clf.predict(X_test))*100:.1f}%")

    hs = [scores[vid]["scores"].get("overall_hate_score", 5) for vid in splits["test"]
          if vid in scores and labels[vid] == 1]
    ns = [scores[vid]["scores"].get("overall_hate_score", 5) for vid in splits["test"]
          if vid in scores and labels[vid] == 0]
    if hs and ns:
        print(f"  Overall score: Hate={np.mean(hs):.1f} Normal={np.mean(ns):.1f}")

    # Also try combining scores with raw MLLM analysis text features
    # Check parse success rate
    parse_ok = sum(1 for v in scores.values() if v["scores"].get("key_evidence", "") != "parse_failed")
    print(f"  Parse success: {parse_ok}/{len(scores)}")
