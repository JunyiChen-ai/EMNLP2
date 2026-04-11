"""Outlier analysis - top sparse samples vs labels."""
import json, os, sys, numpy as np, torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from collections import Counter

ROOT = "/data/jehc223/EMNLP2"
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().cuda()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def encode_frames(fps):
    if not fps: return np.zeros(768)
    imgs = [Image.open(p).convert("RGB") for p in fps]
    inp = processor(images=imgs, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad(): e = model.get_image_features(**inp)
    return e.cpu().numpy().mean(axis=0)

def encode_text(text):
    inp = processor(text=[text[:300]], return_tensors="pt", padding=True, truncation=True).to("cuda")
    with torch.no_grad(): e = model.get_text_features(**inp)
    return e[0].cpu().numpy()

def get_fps(d, vid, mx=32):
    p = os.path.join(d, vid)
    if not os.path.isdir(p): return []
    fs = sorted((f for f in os.listdir(p) if f.endswith((".jpg",".png",".jpeg"))),
                key=lambda f: int(''.join(c for c in os.path.splitext(f)[0] if c.isdigit()) or '0'))
    if len(fs)>mx:
        idx=np.linspace(0,len(fs)-1,mx).round().astype(int)
        s=set(); fs=[fs[i] for i in idx if i not in s and not s.add(i)]
    return [os.path.join(p,f) for f in fs]

for dataset in ["MHClip_EN", "MHClip_ZH"]:
    print(f"\n{'='*60}\n{dataset}\n{'='*60}", flush=True)
    base = f"datasets/{dataset}"
    with open(os.path.join(ROOT,base,"annotation(new).json")) as f: ann=json.load(f)
    id2s={s["Video_ID"]:s for s in ann}
    id2raw={s["Video_ID"]:s["Label"] for s in ann}
    with open(os.path.join(ROOT,base,"splits/test.csv")) as f:
        tids=[l.strip() for l in f if l.strip()]
    tids=[v for v in tids if v in id2s and id2raw.get(v,"?") in ("Hateful","Offensive","Normal")]
    fdir=os.path.join(ROOT,base,"frames")

    embs,vids,raws=[],[],[]
    for i,vid in enumerate(tids):
        s=id2s[vid]
        fe=encode_frames(get_fps(fdir,vid))
        te=encode_text(((s.get("Title","") or "")+" "+(s.get("Transcript","") or ""))[:300])
        embs.append(np.concatenate([fe,te])); vids.append(vid); raws.append(id2raw[vid])
        if (i+1)%50==0: print(f"  {i+1}/{len(tids)}", flush=True)

    X=np.array(embs); X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-8)
    nn=NearestNeighbors(n_neighbors=11,metric='cosine').fit(X)
    dist,_=nn.kneighbors(X)
    avg_d=dist[:,1:].mean(axis=1)
    order=np.argsort(-avg_d)

    print(f"\n  Top-15 MOST SPARSE (outliers):", flush=True)
    for i in order[:15]:
        title=(id2s[vids[i]].get("Title","") or "")[:50]
        print(f"    dist={avg_d[i]:.4f} | {raws[i]:10s} | {vids[i][:20]:20s} | {title}", flush=True)

    print(f"\n  Top-15 MOST DENSE:", flush=True)
    for i in order[-15:]:
        title=(id2s[vids[i]].get("Title","") or "")[:50]
        print(f"    dist={avg_d[i]:.4f} | {raws[i]:10s} | {vids[i][:20]:20s} | {title}", flush=True)

    for pct in [5, 10, 20]:
        n=max(1,len(order)*pct//100)
        top=Counter(raws[i] for i in order[:n])
        bot=Counter(raws[i] for i in order[-n:])
        print(f"\n  Top-{pct}% outliers ({n}): {dict(top)}", flush=True)
        print(f"  Bottom-{pct}% densest ({n}): {dict(bot)}", flush=True)

print("\nDone.", flush=True)
