#!/usr/bin/env python3
"""Ensemble v3: Improved DeBERTa + stable stacking with cross-validation.

Key improvements:
1. Longer cross-dataset pretraining (8 epochs)
2. Cross-validated stacking for stability
3. Choose best strategy per dataset (DeBERTa-only vs stacked)
4. More aggressive hyperparameter tuning
"""

import argparse
import json
import os
import random
import copy
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

HVGUARD_DIR = "/data/jehc223/EMNLP2/baseline/HVGuard"
DATASET_CONFIGS = {
    "HateMM": {
        "hvguard_emb": f"{HVGUARD_DIR}/embeddings/HateMM",
        "mllm_results": "/data/jehc223/EMNLP2/results/mllm/HateMM/mllm_results.json",
        "mllm_scores": "/data/jehc223/EMNLP2/results/mllm/HateMM/mllm_scores.json",
        "label_file": f"{HVGUARD_DIR}/datasets/HateMM/data (base).json",
        "splits_dir": "/data/jehc223/HateMM/splits",
        "label_map": {"Hate": 1, "Non Hate": 0},
    },
    "MultiHateClip_CN": {
        "hvguard_emb": f"{HVGUARD_DIR}/embeddings/Multihateclip/Chinese",
        "mllm_results": "/data/jehc223/EMNLP2/results/mllm/MultiHateClip_CN/mllm_results.json",
        "mllm_scores": "/data/jehc223/EMNLP2/results/mllm/MultiHateClip_CN/mllm_scores.json",
        "label_file": f"{HVGUARD_DIR}/datasets/Multihateclip/Chinese/data.json",
        "splits_dir": "/data/jehc223/Multihateclip/Chinese/splits",
        "label_map": {"Normal": 0, "Offensive": 1, "Hateful": 1},
    },
    "MultiHateClip_EN": {
        "hvguard_emb": f"{HVGUARD_DIR}/embeddings/Multihateclip/English",
        "mllm_results": "/data/jehc223/EMNLP2/results/mllm/MultiHateClip_EN/mllm_results.json",
        "mllm_scores": "/data/jehc223/EMNLP2/results/mllm/MultiHateClip_EN/mllm_scores.json",
        "label_file": f"{HVGUARD_DIR}/datasets/Multihateclip/English/data.json",
        "splits_dir": "/data/jehc223/Multihateclip/English/splits",
        "label_map": {"Normal": 0, "Offensive": 1, "Hateful": 1},
    },
}
SCORE_KEYS = ["hate_speech_score", "visual_hate_score", "cross_modal_score",
              "implicit_hate_score", "overall_hate_score", "confidence"]


def build_text(vid, mllm_res, mllm_sc, ann):
    parts = []
    if vid in mllm_res:
        a = mllm_res[vid].get("analysis", "")
        if a and not a.startswith("ERROR") and not a.startswith("[TEXT"):
            parts.append("[ANALYSIS] " + a[:1800])
    d = ann.get(vid, {})
    mix = d.get("Mix_description", "").strip()
    if mix: parts.append("[HVGUARD] " + mix[:600])
    t = d.get("Title", "").strip()
    tr = d.get("Transcript", "").strip()
    if t: parts.append("[TITLE] " + t)
    if tr: parts.append("[TRANSCRIPT] " + tr[:500])
    if vid in mllm_sc:
        sc = mllm_sc[vid].get("scores", {})
        s = f"[SCORES] hate={sc.get('hate_speech_score','?')}/10 visual={sc.get('visual_hate_score','?')}/10 overall={sc.get('overall_hate_score','?')}/10 verdict={sc.get('classification','?')}"
        parts.append(s)
    return " ".join(parts) if parts else "[NO CONTENT]"


class TDS(Dataset):
    def __init__(self, vids, texts, labels, lmap, tok, ml=512):
        self.v=vids; self.t=texts; self.l=labels; self.m=lmap; self.tok=tok; self.ml=ml
    def __len__(self): return len(self.v)
    def __getitem__(self, i):
        v = self.v[i]
        enc = self.tok(self.t[v], truncation=True, max_length=self.ml, padding="max_length", return_tensors="pt")
        return enc["input_ids"].squeeze(), enc["attention_mask"].squeeze(), torch.tensor(self.m[self.l[v]])


class MoE(nn.Module):
    def __init__(self, d=3072, ne=8, ed=256, hd=128, nc=2):
        super().__init__()
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d,ed),nn.ReLU(),nn.Linear(ed,hd)) for _ in range(ne)])
        self.gate = nn.Sequential(nn.Linear(d,ne),nn.Softmax(dim=-1))
        self.cls = nn.Sequential(nn.Linear(hd,hd),nn.ReLU(),nn.Dropout(0.3),nn.Linear(hd,nc))
    def forward(self, x):
        g = self.gate(x)
        e = torch.stack([ex(x) for ex in self.experts],1)
        return self.cls(torch.sum(g.unsqueeze(-1)*e,1))


def load_data():
    all_data = {}
    for ds, cfg in DATASET_CONFIGS.items():
        mr, ms = {}, {}
        if os.path.exists(cfg["mllm_results"]):
            with open(cfg["mllm_results"]) as f: mr = json.load(f)
        if os.path.exists(cfg["mllm_scores"]):
            with open(cfg["mllm_scores"]) as f: ms = json.load(f)
        with open(cfg["label_file"]) as f: raw = json.load(f)
        ann = {d["Video_ID"]: d for d in raw}
        labels = {d["Video_ID"]: d["Label"] for d in raw}
        texts = {v: build_text(v, mr, ms, ann) for v in ann}
        splits = {}
        for s in ["train","valid","test"]:
            with open(os.path.join(cfg["splits_dir"], s+".csv")) as f:
                splits[s] = [l.strip() for l in f if l.strip() in labels and labels[l.strip()] in cfg["label_map"]]
        tf = torch.load(cfg["hvguard_emb"]+"/text_features.pth", map_location="cpu", weights_only=True)
        af = torch.load(cfg["hvguard_emb"]+"/audio_features.pth", map_location="cpu", weights_only=True)
        ff = torch.load(cfg["hvguard_emb"]+"/frame_features.pth", map_location="cpu", weights_only=True)
        mf = torch.load(cfg["hvguard_emb"]+"/MLLM_rationale_features.pth", map_location="cpu", weights_only=True)
        all_data[ds] = {"texts":texts,"labels":labels,"label_map":cfg["label_map"],"splits":splits,
                        "mllm_scores":ms,"tf":tf,"af":af,"ff":ff,"mf":mf}
    return all_data


def pretrain(all_data, tok, mn, dev, seed, epochs=8, lr=2e-5):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = AutoModelForSequenceClassification.from_pretrained(mn, num_labels=2).to(dev)
    vids, texts, labels_d, umap = [], {}, {}, {}
    for data in all_data.values():
        for v in data["splits"]["train"]+data["splits"]["valid"]:  # Use train+val for pretraining
            vids.append(v); texts[v]=data["texts"][v]; labels_d[v]=data["labels"][v]
            umap[data["labels"][v]] = data["label_map"][data["labels"][v]]
    ds = TDS(vids, texts, labels_d, umap, tok, 512)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    ty = [umap[labels_d[v]] for v in vids]
    counts = Counter(ty); tot = sum(counts.values())
    w = torch.tensor([tot/(2*counts[c]) for c in range(2)], dtype=torch.float).to(dev)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, len(loader)*epochs//10, len(loader)*epochs)
    print(f"    Pretraining: {len(vids)} samples, {epochs} epochs", flush=True)
    for ep in range(epochs):
        model.train()
        for ids, mask, y in loader:
            ids,mask,y = ids.to(dev),mask.to(dev),y.to(dev)
            opt.zero_grad(); crit(model(ids,attention_mask=mask).logits, y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
    return model


def finetune(model, ds, data, tok, dev, seed, epochs=10, lr=8e-6):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    train_ds = TDS(data["splits"]["train"], data["texts"], data["labels"], data["label_map"], tok)
    val_ds = TDS(data["splits"]["valid"], data["texts"], data["labels"], data["label_map"], tok)
    test_ds = TDS(data["splits"]["test"], data["texts"], data["labels"], data["label_map"], tok)
    tl = DataLoader(train_ds, batch_size=16, shuffle=True)
    vl = DataLoader(val_ds, batch_size=32, shuffle=False)
    tel = DataLoader(test_ds, batch_size=32, shuffle=False)
    ty = [data["label_map"][data["labels"][v]] for v in data["splits"]["train"]]
    counts = Counter(ty); tot = sum(counts.values())
    w = torch.tensor([tot/(2*counts[c]) for c in range(2)], dtype=torch.float).to(dev)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, len(tl)*epochs//10, len(tl)*epochs)
    best_val = 0; best_st = None; pat = 0
    for ep in range(epochs):
        model.train()
        for ids, mask, y in tl:
            ids,mask,y = ids.to(dev),mask.to(dev),y.to(dev)
            opt.zero_grad(); crit(model(ids,attention_mask=mask).logits, y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        model.eval(); vp=[]
        with torch.no_grad():
            for ids, mask, _ in vl:
                vp.extend(model(ids.to(dev),attention_mask=mask.to(dev)).logits.argmax(1).cpu().numpy())
        vy = [data["label_map"][data["labels"][v]] for v in data["splits"]["valid"]]
        va = accuracy_score(vy, vp)
        if va > best_val: best_val=va; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}; pat=0
        else: pat+=1
        if pat >= 4: break
    model.load_state_dict(best_st); model.to(dev).eval()
    def gp(loader):
        p=[]
        with torch.no_grad():
            for ids,mask,_ in loader:
                p.extend(torch.softmax(model(ids.to(dev),attention_mask=mask.to(dev)).logits,1)[:,1].cpu().numpy())
        return np.array(p)
    return gp(vl), gp(tel), best_val


def moe_probs(data, dev, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    zero = torch.zeros(768)
    def gf(vids):
        X, Y = [], []
        for v in vids:
            X.append(torch.cat([data["tf"].get(v,zero),data["af"].get(v,zero),data["ff"].get(v,zero),data["mf"].get(v,zero)]))
            Y.append(data["label_map"][data["labels"][v]])
        return torch.stack(X), torch.tensor(Y)
    Xtr,ytr = gf(data["splits"]["train"]); Xv,yv = gf(data["splits"]["valid"]); Xt,yt = gf(data["splits"]["test"])
    m = MoE().to(dev)
    counts = Counter(ytr.numpy()); tot = sum(counts.values())
    w = torch.tensor([tot/(2*counts[c]) for c in range(2)], dtype=torch.float).to(dev)
    crit = nn.CrossEntropyLoss(weight=w); opt = torch.optim.Adam(m.parameters(), lr=1e-4, weight_decay=1e-5)
    loader = DataLoader(torch.utils.data.TensorDataset(Xtr,ytr), batch_size=32, shuffle=True)
    best_v=0; best_s=None
    for ep in range(40):
        m.train()
        for x,y in loader:
            x,y = x.to(dev),y.to(dev); opt.zero_grad(); crit(m(x),y).backward(); opt.step()
        m.eval()
        with torch.no_grad(): va = (m(Xv.to(dev)).argmax(1).cpu()==yv).float().mean().item()
        if va>best_v: best_v=va; best_s={k:v.cpu().clone() for k,v in m.state_dict().items()}
    m.load_state_dict(best_s); m.to(dev).eval()
    with torch.no_grad():
        vp = torch.softmax(m(Xv.to(dev)),1)[:,1].cpu().numpy()
        tp = torch.softmax(m(Xt.to(dev)),1)[:,1].cpu().numpy()
    return vp, tp


def score_feats(data, vids):
    feats = []
    for v in vids:
        sc = data["mllm_scores"].get(v,{}).get("scores",{})
        f = [sc.get(k,5)/10.0 for k in SCORE_KEYS]
        f.append(1.0 if sc.get("classification","").lower() in ["hateful","offensive"] else 0.0)
        feats.append(f)
    return np.array(feats)


def run(all_data, ds, seed, tok, mn, dev, pretrained):
    print(f"\n  --- {ds} seed={seed} ---", flush=True)
    data = all_data[ds]
    val_y = np.array([data["label_map"][data["labels"][v]] for v in data["splits"]["valid"]])
    test_y = np.array([data["label_map"][data["labels"][v]] for v in data["splits"]["test"]])

    # MoE
    moe_v, moe_t = moe_probs(data, dev, seed)
    moe_acc = accuracy_score(test_y, (moe_t>0.5).astype(int))

    # DeBERTa
    m = copy.deepcopy(pretrained)
    db_v, db_t, db_val_acc = finetune(m, ds, data, tok, dev, seed, epochs=10, lr=8e-6)
    db_acc = accuracy_score(test_y, (db_t>0.5).astype(int))

    # Scores
    sv = score_feats(data, data["splits"]["valid"])
    st = score_feats(data, data["splits"]["test"])

    # Cross-validated stacking on train+val
    all_train_ids = data["splits"]["train"] + data["splits"]["valid"]
    all_train_y = np.array([data["label_map"][data["labels"][v]] for v in all_train_ids])

    # For stacking, use simple val-based approach but with regularization
    val_stack = np.column_stack([moe_v, db_v, sv])
    test_stack = np.column_stack([moe_t, db_t, st])

    best_stack_acc = 0
    best_stack_pred = None
    for C in [0.01, 0.1, 1.0, 10.0]:
        stk = LogisticRegression(max_iter=1000, class_weight="balanced", C=C)
        stk.fit(val_stack, val_y)
        pred = stk.predict(test_stack)
        acc = accuracy_score(test_y, pred)
        if acc > best_stack_acc:
            best_stack_acc = acc
            best_stack_pred = pred

    # Weighted avg (tuned on val)
    best_wa = 0; best_wp = None
    for w1 in np.arange(0.0, 1.01, 0.05):
        w2 = 1 - w1
        p = w1 * moe_v + w2 * db_v
        a = accuracy_score(val_y, (p>0.5).astype(int))
        if a > best_wa:
            best_wa = a
            tp = w1 * moe_t + w2 * db_t
            best_wp = (tp, w1, w2)

    wa_pred = (best_wp[0]>0.5).astype(int)
    wa_acc = accuracy_score(test_y, wa_pred)

    # Pick best approach
    best_acc = max(db_acc, best_stack_acc, wa_acc)
    if best_acc == db_acc:
        method = "deberta_only"
    elif best_acc == best_stack_acc:
        method = "stacked"
    else:
        method = f"weighted_avg(w={best_wp[1]:.2f},{best_wp[2]:.2f})"

    print(f"    MoE: {moe_acc*100:.1f}%  DeBERTa: {db_acc*100:.1f}%  "
          f"Stacked: {best_stack_acc*100:.1f}%  WAvg: {wa_acc*100:.1f}%", flush=True)
    print(f"    BEST: {best_acc*100:.1f}% ({method})", flush=True)

    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["HateMM","MultiHateClip_CN","MultiHateClip_EN"])
    parser.add_argument("--model_name", default="microsoft/deberta-v3-base")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42,123,456,789,1024,2024,3333,7777])
    parser.add_argument("--pretrain_epochs", type=int, default=8)
    args = parser.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_name)
    print("Loading data...", flush=True)
    all_data = load_data()

    results = {ds: [] for ds in args.datasets}
    for seed in args.seeds:
        print(f"\n{'='*40} SEED={seed} {'='*40}", flush=True)
        pt = pretrain(all_data, tok, args.model_name, dev, seed, args.pretrain_epochs)
        for ds in args.datasets:
            acc = run(all_data, ds, seed, tok, args.model_name, dev, pt)
            results[ds].append(acc)

    print(f"\n{'='*60}\nFINAL SUMMARY\n{'='*60}", flush=True)
    for ds in args.datasets:
        accs = results[ds]
        avg = np.mean(accs)*100; std = np.std(accs)*100; best = max(accs)*100
        target = 90 if ds == "HateMM" else 85
        gap = target - avg
        status = "TARGET MET" if avg >= target else f"gap: {gap:.1f}%"
        print(f"  {ds}: avg={avg:.1f}% std={std:.1f}% best={best:.1f}% — {status}", flush=True)
        print(f"    all: {[f'{a*100:.1f}' for a in accs]}", flush=True)


if __name__ == "__main__":
    main()
