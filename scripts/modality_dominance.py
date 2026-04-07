"""
Modality dominance analysis:
1. Title-only vs Transcript-only vs Title+Transcript vs MLLM Rationale
2. What does MLLM rationale add over raw text?
"""
import json, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer
import warnings; warnings.filterwarnings("ignore")

def load_data(dataset_name):
    paths = {"HateMM": "datasets/HateMM", "MHClip_EN": "datasets/Multihateclip/English", "MHClip_ZH": "datasets/Multihateclip/Chinese"}
    base = paths[dataset_name]
    with open(f"{base}/generic_data.json") as f: data = json.load(f)
    with open(f"{base}/splits/train.csv") as f: train_ids = [l.strip() for l in f if l.strip()]
    with open(f"{base}/splits/test.csv") as f: test_ids = [l.strip() for l in f if l.strip()]
    label_map = {"Hate": 1, "Non Hate": 0, "Hateful": 1, "Offensive": 1, "Normal": 0}
    id2sample = {s["Video_ID"]: s for s in data}
    return id2sample, train_ids, test_ids, label_map

def train_mlp(train_X, train_y, test_X, test_y, dim, n_seeds=20):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    all_preds = []
    for seed in range(n_seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        model = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2)).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        tX = torch.tensor(train_X, dtype=torch.float32).to(dev)
        ty = torch.tensor(train_y, dtype=torch.long).to(dev)
        ds = torch.utils.data.TensorDataset(tX, ty)
        loader = DataLoader(ds, batch_size=32, shuffle=True)
        model.train()
        for _ in range(50):
            for x, y in loader:
                opt.zero_grad(); crit(model(x.to(dev)), y.to(dev)).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(test_X, dtype=torch.float32).to(dev)).argmax(1).cpu().numpy()
        all_preds.append(preds)
    return np.round(np.array(all_preds).mean(0)).astype(int)

def get_rationale_text(sample):
    resp = sample.get("generic_response", {})
    if isinstance(resp, dict):
        parts = [str(resp.get(k, "")) for k in ["content_summary", "target_analysis", "sentiment_tone", "harm_assessment", "overall_judgment"] if resp.get(k)]
        return " ".join(parts) if parts else "[empty]"
    return str(resp) if resp else "[empty]"

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
print(f"Using device: {device}")

for ds_name in ["HateMM", "MHClip_EN", "MHClip_ZH"]:
    print(f"\n{'#'*50}")
    print(f"# {ds_name}")
    print(f"{'#'*50}")
    id2sample, train_ids, test_ids, label_map = load_data(ds_name)
    
    train_vids, train_labels, test_vids, test_labels = [], [], [], []
    for v in train_ids:
        if v in id2sample:
            gt = label_map.get(id2sample[v]["Label"], -1)
            if gt != -1: train_vids.append(v); train_labels.append(gt)
    for v in test_ids:
        if v in id2sample:
            gt = label_map.get(id2sample[v]["Label"], -1)
            if gt != -1: test_vids.append(v); test_labels.append(gt)
    train_labels = np.array(train_labels); test_labels = np.array(test_labels)
    
    def enc(texts): return encoder.encode(texts, show_progress_bar=False)
    
    get = lambda v, k: (id2sample[v].get(k, "") or "")
    train_title = [get(v,"Title") or "[empty]" for v in train_vids]
    test_title = [get(v,"Title") or "[empty]" for v in test_vids]
    train_trans = [get(v,"Transcript") or "[empty]" for v in train_vids]
    test_trans = [get(v,"Transcript") or "[empty]" for v in test_vids]
    train_both = [f"{get(v,'Title')} {get(v,'Transcript')}".strip() or "[empty]" for v in train_vids]
    test_both = [f"{get(v,'Title')} {get(v,'Transcript')}".strip() or "[empty]" for v in test_vids]
    train_rat = [get_rationale_text(id2sample[v]) for v in train_vids]
    test_rat = [get_rationale_text(id2sample[v]) for v in test_vids]
    
    n_empty_title = sum(1 for t in test_title if t == "[empty]")
    n_empty_trans = sum(1 for t in test_trans if t == "[empty]")
    print(f"\nTest: {len(test_labels)} | Empty title: {n_empty_title} ({n_empty_title/len(test_labels)*100:.0f}%) | Empty transcript: {n_empty_trans} ({n_empty_trans/len(test_labels)*100:.0f}%)")
    
    print("\nSingle-modality results (20 seeds):")
    results = {}
    for name, tr, te in [("Title only", enc(train_title), enc(test_title)),
                          ("Transcript only", enc(train_trans), enc(test_trans)),
                          ("Title+Transcript", enc(train_both), enc(test_both)),
                          ("MLLM Rationale", enc(train_rat), enc(test_rat))]:
        preds = train_mlp(tr, train_labels, te, test_labels, tr.shape[1])
        acc = accuracy_score(test_labels, preds)
        f1 = f1_score(test_labels, preds, average="macro")
        results[name] = preds
        print(f"  {name:25s}: Acc={acc:.3f}, F1={f1:.3f}")
    
    # Complementarity
    text_c = (results["Title+Transcript"] == test_labels)
    rat_c = (results["MLLM Rationale"] == test_labels)
    print(f"\n--- Rationale vs Raw Text ---")
    print(f"Both correct:      {(text_c & rat_c).sum()}")
    print(f"Text only correct:  {(text_c & ~rat_c).sum()}")
    print(f"Rat only correct:   {(~text_c & rat_c).sum()} ← MLLM adds value here")
    print(f"Both wrong:         {(~text_c & ~rat_c).sum()}")
    
    # What does rationale uniquely solve?
    rat_only = (~text_c & rat_c)
    if rat_only.sum() > 0:
        print(f"\nSamples MLLM rationale solves but raw text can't ({rat_only.sum()}):")
        count = 0
        for i, v in enumerate(test_vids):
            if rat_only[i]:
                s = id2sample[v]
                trans = (s.get("Transcript","") or "")[:60]
                title = (s.get("Title","") or "")[:40]
                print(f"  [{v}] GT={s['Label']} | title='{title}' | trans='{trans}'")
                count += 1
                if count >= 10: 
                    if rat_only.sum() > 10: print(f"  ... ({rat_only.sum()-10} more)")
                    break
    
    # What does raw text solve that rationale can't?
    text_only = (text_c & ~rat_c)
    if text_only.sum() > 0:
        print(f"\nSamples raw text solves but MLLM rationale can't ({text_only.sum()}):")
        count = 0
        for i, v in enumerate(test_vids):
            if text_only[i]:
                s = id2sample[v]
                trans = (s.get("Transcript","") or "")[:60]
                title = (s.get("Title","") or "")[:40]
                print(f"  [{v}] GT={s['Label']} | title='{title}' | trans='{trans}'")
                count += 1
                if count >= 10:
                    if text_only.sum() > 10: print(f"  ... ({text_only.sum()-10} more)")
                    break
