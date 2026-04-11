"""Analyze embedding density vs hateful labels."""
import json, os, numpy as np
from collections import Counter
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from sklearn.neighbors import NearestNeighbors

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

def get_fps(frames_dir, vid_id, mx=32):
    d = os.path.join(frames_dir, vid_id)
    if not os.path.isdir(d): return []
    fs = sorted((f for f in os.listdir(d) if f.endswith((".jpg",".png",".jpeg"))),
                key=lambda f: int(''.join(c for c in os.path.splitext(f)[0] if c.isdigit()) or '0'))
    if len(fs) > mx:
        idx = np.linspace(0,len(fs)-1,mx).round().astype(int)
        seen=set(); fs=[fs[i] for i in idx if i not in seen and not seen.add(i)]
    return [os.path.join(d,f) for f in fs]

for dataset in ["HateMM", "MHClip_EN", "MHClip_ZH"]:
    print(f"\n{'='*60}\n{dataset}\n{'='*60}")
    cfgs = {
        "HateMM": ("datasets/HateMM", {"Hate":1,"Non Hate":0}),
        "MHClip_EN": ("datasets/MHClip_EN", {"Hateful":1,"Offensive":1,"Normal":0}),
        "MHClip_ZH": ("datasets/MHClip_ZH", {"Hateful":1,"Offensive":1,"Normal":0}),
    }
    base, lmap = cfgs[dataset]
    with open(os.path.join(ROOT,base,"annotation(new).json")) as f: ann=json.load(f)
    id2s = {s["Video_ID"]:s for s in ann}
    id2raw = {s["Video_ID"]:s["Label"] for s in ann}
    with open(os.path.join(ROOT,base,"splits/test.csv")) as f:
        tids=[l.strip() for l in f if l.strip()]
    tids=[v for v in tids if v in id2s]
    fdir=os.path.join(ROOT,base,"frames")

    embs,labs,raws,vids=[],[],[],[]
    for i,vid in enumerate(tids):
        s=id2s[vid]; raw=id2raw.get(vid,"?"); gt=lmap.get(raw,-1)
        if gt==-1: continue
        title=s.get("Title","") or ""; trans=s.get("Transcript","") or ""
        fe=encode_frames(get_fps(fdir,vid))
        te=encode_text((title+" "+trans)[:300])
        embs.append(np.concatenate([fe,te])); labs.append(gt); raws.append(raw); vids.append(vid)
        if (i+1)%50==0: print(f"  {i+1}/{len(tids)}")

    X=np.array(embs); X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-8)
    k=10
    nn=NearestNeighbors(n_neighbors=k+1,metric='cosine').fit(X)
    dist,_=nn.kneighbors(X)
    avg_d=dist[:,1:].mean(axis=1)

    print(f"\n  N={len(X)}, k={k}")
    print(f"  Avg dist: mean={avg_d.mean():.4f}, std={avg_d.std():.4f}")
    qs=np.percentile(avg_d,[25,50,75])
    print(f"  Quartiles: {qs}")
    for qn,lo,hi in [("Q1 densest",0,qs[0]),("Q2",qs[0],qs[1]),("Q3",qs[1],qs[2]),("Q4 sparsest",qs[2],999)]:
        m=(avg_d>=lo)&(avg_d<hi)
        ql=[labs[i] for i in range(len(labs)) if m[i]]
        qr=[raws[i] for i in range(len(raws)) if m[i]]
        nh=sum(1 for l in ql if l==1); nt=len(ql)
        print(f"  {qn}: n={nt}, hate={nh} ({100*nh/max(nt,1):.0f}%), raw={dict(Counter(qr))}")

    # Also: Hateful-only for MHClip
    if dataset != "HateMM":
        print(f"\n  Hateful-only density analysis:")
        for qn,lo,hi in [("Q1 densest",0,qs[0]),("Q2",qs[0],qs[1]),("Q3",qs[1],qs[2]),("Q4 sparsest",qs[2],999)]:
            m=(avg_d>=lo)&(avg_d<hi)
            qr=[raws[i] for i in range(len(raws)) if m[i]]
            n_h=sum(1 for r in qr if r=="Hateful")
            n_o=sum(1 for r in qr if r=="Offensive")
            n_n=sum(1 for r in qr if r=="Normal")
            print(f"  {qn}: Hateful={n_h}, Offensive={n_o}, Normal={n_n}")

print("\nDone.")
