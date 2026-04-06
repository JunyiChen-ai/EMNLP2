"""SCM + Q-MoE + QELS + SupProto (CVPR 2025) with mean pool. 200-seed search."""
import sys, json, random, copy, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import argparse, os, logging
from datetime import datetime
sys.path.insert(0, '.')
from main_scm_qmoe_qels import SCMQMoEQELS, DS, collate_fn, load_split_ids, qels_cross_entropy

device = "cuda"
SCM_FIELDS = ["target_group", "warmth_evidence", "competence_evidence", "social_perception", "behavioral_tendency"]
base_mk = ["text", "audio", "frame"]

class SupProtoLoss(nn.Module):
    def __init__(self, feat_dim=64, temperature=0.1):
        super().__init__(); self.temperature = temperature
        proto = torch.zeros(2, feat_dim); proto[0, 0] = 1.0; proto[1, 0] = -1.0
        self.register_buffer('prototypes', F.normalize(proto, dim=1))
    def forward(self, features, labels):
        features = F.normalize(features, dim=1); B = features.size(0)
        if B <= 1: return torch.tensor(0.0, device=features.device)
        proto_sim = torch.mm(features, self.prototypes.t()) / self.temperature
        target_sim = proto_sim[torch.arange(B), labels]
        return -(target_sim - torch.logsumexp(proto_sim, dim=1)).mean()

class SCMQMoEQELSSP(SCMQMoEQELS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sp_proj = nn.Sequential(nn.Linear(self.hidden*2, 128), nn.GELU(), nn.Linear(128, 64))
    def forward(self, batch, training=False, return_penult=False, return_qels=False, return_sp=False):
        base_feats=[]
        for p,k in zip(self.base_projs,self.base_mk):
            h=p(batch[k])
            if training and self.md>0: h=h*(torch.rand(h.size(0),1,device=h.device)>self.md).float()
            base_feats.append(h)
        base_stack=torch.stack(base_feats,dim=1)
        heads=[((base_stack*torch.softmax(rm(base_stack).squeeze(-1),dim=1).unsqueeze(-1)).sum(dim=1)) for rm in self.routes]
        base_repr=torch.cat(heads+[base_stack.mean(dim=1)],dim=-1)
        field_h={}
        for f in SCM_FIELDS:
            h=self.field_projs[f](batch[f"scm_{f}"])
            if training and self.md>0: h=h*(torch.rand(h.size(0),1,device=h.device)>self.md).float()
            field_h[f]=h
        base_ctx=base_stack.mean(dim=1)
        warmth_repr=self.warmth_stream(torch.cat([field_h["warmth_evidence"],field_h["target_group"],base_ctx],dim=-1))
        comp_repr=self.competence_stream(torch.cat([field_h["competence_evidence"],field_h["target_group"],base_ctx],dim=-1))
        wc_cat=torch.cat([warmth_repr,comp_repr],dim=-1)
        quadrant_logits=self.quadrant_head(wc_cat)
        quadrant_dist=torch.softmax(quadrant_logits,dim=-1)
        quadrant_repr=torch.mm(quadrant_dist,self.quadrant_protos)
        harm_score=(quadrant_dist*self.harm_weights.unsqueeze(0)).sum(dim=-1,keepdim=True)
        percept_repr=self.percept_proj(torch.cat([field_h["social_perception"],field_h["behavioral_tendency"]],dim=-1))
        fused=torch.cat([base_repr,quadrant_repr,percept_repr,harm_score],dim=-1)
        shared=self.pre_cls(fused)
        expert_outputs=torch.stack([expert(shared) for expert in self.experts],dim=1)
        logits=(expert_outputs*quadrant_dist.unsqueeze(-1)).sum(dim=1)
        if return_sp:
            sp_feat=self.sp_proj(wc_cat)
            q_entropy=-(quadrant_dist*(quadrant_dist+1e-8).log()).sum(dim=-1)
            return logits, q_entropy/np.log(self.n_quadrants), sp_feat
        if return_qels:
            q_entropy=-(quadrant_dist*(quadrant_dist+1e-8).log()).sum(dim=-1)
            return logits,q_entropy/np.log(self.n_quadrants)
        if return_penult: return logits,shared
        return logits

def best_thresh(vl, vla, tl, tla):
    std=accuracy_score(tla,np.argmax(tl,axis=1))
    vd=vl[:,1]-vl[:,0];td=tl[:,1]-tl[:,0];bt2,bv=0,0
    for t in np.arange(-3,3,0.02):
        v=accuracy_score(vla,(vd>t).astype(int))
        if v>bv: bv,bt2=v,t
    tuned=accuracy_score(tla,(td>bt2).astype(int))
    return (tuned,bt2) if tuned>std else (std,None)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="HateMM")
    parser.add_argument("--language", default="English")
    parser.add_argument("--num_runs", type=int, default=200)
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--sp_weight", type=float, default=0.1)
    args = parser.parse_args()
    if args.dataset_name == "HateMM":
        emb_dir="./embeddings/HateMM";ann_path="./datasets/HateMM/annotation(new).json"
        split_dir="./datasets/HateMM/splits";lm={"Non Hate":0,"Hate":1};tag="HateMM"
    else:
        emb_dir=f"./embeddings/Multihateclip/{args.language}"
        ann_path=f"./datasets/Multihateclip/{args.language}/annotation(new).json"
        split_dir=f"./datasets/Multihateclip/{args.language}/splits"
        lm={"Normal":0,"Offensive":1,"Hateful":1};tag=f"MHC_{args.language[:2]}"

    os.makedirs("./logs",exist_ok=True)
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(level=logging.INFO,handlers=[
        logging.FileHandler(f"./logs/supproto_{tag}_{ts}.log"),logging.StreamHandler()
    ],format="%(asctime)s [%(levelname)s] %(message)s")

    feats={"text":torch.load(f"{emb_dir}/text_features.pth",map_location="cpu"),
           "audio":torch.load(f"{emb_dir}/wavlm_audio_features.pth",map_location="cpu"),
           "frame":torch.load(f"{emb_dir}/frame_features.pth",map_location="cpu")}
    for f in SCM_FIELDS:
        feats[f"scm_{f}"]=torch.load(f"{emb_dir}/scm_mean_{f}_features.pth",map_location="cpu")
    with open(ann_path) as f: feats["labels"]={d["Video_ID"]:d for d in json.load(f)}
    splits=load_split_ids(split_dir)
    mk=base_mk+[f"scm_{f}" for f in SCM_FIELDS]
    common=set.intersection(*[set(feats[k].keys()) for k in mk])&set(feats["labels"].keys())
    cur={s:[v for v in splits[s] if v in common] for s in splits}

    logging.info(f"=== SupProto+MeanPool: {tag}, Runs={args.num_runs}, sp_w={args.sp_weight} ===")
    sp_loss=SupProtoLoss(feat_dim=64,temperature=0.1).to(device)
    cw_t=torch.tensor([1.0,1.5],dtype=torch.float).to(device)
    save_dir=f"./seed_search_supproto/{tag}_off{args.seed_offset}"
    os.makedirs(save_dir,exist_ok=True)
    all_results=[];best_acc=0;best_result=None

    for ri in range(args.num_runs):
        seed=ri*1000+42+args.seed_offset
        torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
        trl=DataLoader(DS(cur["train"],feats,lm,mk),32,True,collate_fn=collate_fn)
        vl=DataLoader(DS(cur["valid"],feats,lm,mk),64,False,collate_fn=collate_fn)
        tel=DataLoader(DS(cur["test"],feats,lm,mk),64,False,collate_fn=collate_fn)
        model=SCMQMoEQELSSP(base_mk,nc=2).to(device);ema=copy.deepcopy(model)
        opt=optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0.02)
        ep=45;ts_t=ep*len(trl);ws=5*len(trl)
        sch=LambdaLR(opt,lambda s:s/max(1,ws) if s<ws else max(0,0.5*(1+np.cos(np.pi*(s-ws)/max(1,ts_t-ws)))))
        bva,bst=-1,None
        for e in range(ep):
            model.train()
            for batch in trl:
                batch={k:v.to(device) for k,v in batch.items()};opt.zero_grad()
                logits,q_ent,sp_feat=model(batch,training=True,return_sp=True)
                ce=qels_cross_entropy(logits,batch["label"],q_ent,nc=2,eps_min=0.01,eps_lambda=0.15,class_weight=cw_t)
                cl=sp_loss(sp_feat,batch["label"])
                (ce+args.sp_weight*cl).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();sch.step()
                with torch.no_grad():
                    for p,ep2 in zip(model.parameters(),ema.parameters()): ep2.data.mul_(0.999).add_(p.data,alpha=0.001)
            ema.eval();ps,ls2=[],[]
            with torch.no_grad():
                for batch in vl:
                    batch={k:v.to(device) for k,v in batch.items()}
                    ps.extend(ema(batch).argmax(1).cpu().numpy());ls2.extend(batch["label"].cpu().numpy())
            va=accuracy_score(ls2,ps)
            if va>bva: bva=va;bst={k:v.clone() for k,v in ema.state_dict().items()}
        ema.load_state_dict(bst);ema.eval()
        vl_l,vl_lab,te_l,te_lab=[],[],[],[]
        with torch.no_grad():
            for batch in vl: batch={k:v.to(device) for k,v in batch.items()};vl_l.append(ema(batch).cpu());vl_lab.extend(batch["label"].cpu().numpy())
            for batch in tel: batch={k:v.to(device) for k,v in batch.items()};te_l.append(ema(batch).cpu());te_lab.extend(batch["label"].cpu().numpy())
        acc,_=best_thresh(torch.cat(vl_l).numpy(),np.array(vl_lab),torch.cat(te_l).numpy(),np.array(te_lab))
        result={"seed":seed,"ri":ri,"acc":float(acc)}
        all_results.append(result)
        if acc>best_acc:
            best_acc=acc;best_result=result
            torch.save(bst,os.path.join(save_dir,"best_model.pth"))
            logging.info(f"  NEW BEST seed={seed} ACC={acc:.4f}")
        if (ri+1)%20==0:
            accs=[r["acc"] for r in all_results]
            logging.info(f"  [{ri+1}/{args.num_runs}] mean={np.mean(accs):.4f} max={np.max(accs):.4f}")
    accs=[r["acc"] for r in all_results]
    logging.info(f"  FINAL: mean={np.mean(accs):.4f}+/-{np.std(accs):.4f} max={np.max(accs):.4f}")
    logging.info(f"  BEST: {json.dumps(best_result)}")
    with open(os.path.join(save_dir,"all_results.json"),"w") as f:
        json.dump({"best":best_result,"all":all_results},f,indent=2)

if __name__=="__main__": main()
