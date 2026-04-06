"""Theory Consistency Analysis + SCM Extraction Quality."""
import csv, json, os, random, copy, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
import warnings; warnings.filterwarnings('ignore')

device = 'cuda'
SCM_FIELDS = ['target_group', 'warmth_evidence', 'competence_evidence', 'social_perception', 'behavioral_tendency']

class DS(Dataset):
    def __init__(s, v, f, l, m): s.vids=v;s.f=f;s.lm=l;s.mk=m
    def __len__(s): return len(s.vids)
    def __getitem__(s, i):
        v=s.vids[i]; o={k:s.f[k][v] for k in s.mk}
        o['label']=torch.tensor(s.lm[s.f['labels'][v]['Label']],dtype=torch.long); return o

def collate_fn(b): return {k: torch.stack([x[k] for x in b]) for k in b[0]}

from main_scm_qmoe_qels_mr2 import SCMQMoEQELSMR2, qels_cross_entropy, mr2_loss, load_balance_loss

def cw_sched(opt, ws, ts):
    from torch.optim.lr_scheduler import LambdaLR
    def f(s):
        if s < ws: return s / max(1, ws)
        return max(0, 0.5*(1+np.cos(np.pi*(s-ws)/max(1,ts-ws))))
    return LambdaLR(opt, f)

def load_split_ids(d):
    s={}
    for n in ['train','valid','test']:
        with open(os.path.join(d,f'{n}.csv')) as f:
            s[n]=[r[0] for r in csv.reader(f) if r]
    return s

configs = [
    ('HateMM', './embeddings/HateMM', './datasets/HateMM/annotation(new).json', './datasets/HateMM/splits', {'Non Hate':0,'Hate':1}, 0.3, 0.0, 159042),
    ('MHClip-Y', './embeddings/Multihateclip/English', './datasets/Multihateclip/English/annotation(new).json', './datasets/Multihateclip/English/splits', {'Normal':0,'Offensive':1,'Hateful':1}, 0.1, 0.1, 9042),
    ('MHClip-B', './embeddings/Multihateclip/Chinese', './datasets/Multihateclip/Chinese/annotation(new).json', './datasets/Multihateclip/Chinese/splits', {'Normal':0,'Offensive':1,'Hateful':1}, 0.1, 0.0, 15042),
    ('ImpliHateVid', './embeddings/ImpliHateVid', './datasets/ImpliHateVid/annotation(new).json', './datasets/ImpliHateVid/splits', {'Normal':0,'Hateful':1}, 0.3, 0.0, 2042),
]

results = {}

for name, emb_dir, ann_path, split_dir, lm, alpha, beta, best_seed in configs:
    feats={'text':torch.load(f'{emb_dir}/text_features.pth',map_location='cpu'),
           'audio':torch.load(f'{emb_dir}/wavlm_audio_features.pth',map_location='cpu'),
           'frame':torch.load(f'{emb_dir}/frame_features.pth',map_location='cpu')}
    for field in SCM_FIELDS:
        feats[f'scm_{field}']=torch.load(f'{emb_dir}/scm_mean_{field}_features.pth',map_location='cpu')
    with open(ann_path) as f:
        feats['labels']={d['Video_ID']:d for d in json.load(f)}
    splits=load_split_ids(split_dir)
    base_mk=['text','audio','frame']
    all_mk=list(base_mk)+[f'scm_{f}' for f in SCM_FIELDS]
    common=set.intersection(*[set(feats[k].keys()) for k in all_mk])&set(feats['labels'].keys())
    cur={s:[v for v in splits[s] if v in common] for s in splits}
    cw_tensor=torch.tensor([1.0,1.5],dtype=torch.float).to(device)

    seed = best_seed
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    trd=DS(cur['train'],feats,lm,all_mk); ted=DS(cur['test'],feats,lm,all_mk); vd=DS(cur['valid'],feats,lm,all_mk)
    trl=DataLoader(trd,32,True,collate_fn=collate_fn); tel=DataLoader(ted,64,False,collate_fn=collate_fn)
    vl=DataLoader(vd,64,False,collate_fn=collate_fn)

    model=SCMQMoEQELSMR2(base_mk,nc=2,expert_hidden=64,load_balance_weight=0.01).to(device)
    ema=copy.deepcopy(model)
    opt_=optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0.02)
    ep=45;ts_t=ep*len(trl);ws=5*len(trl);sch=cw_sched(opt_,ws,ts_t)
    bva,bst=-1,None
    for e in range(ep):
        model.train()
        for batch in trl:
            b={k:v.to(device) for k,v in batch.items()}; opt_.zero_grad()
            logits,qe,shared,qdist=model(b,training=True,return_all=True)
            ce=qels_cross_entropy(logits,b['label'],qe,nc=2,class_weight=cw_tensor)
            compact,sep=mr2_loss(shared,b['label'],nc=2)
            lb=load_balance_loss(qdist)
            loss=ce+alpha*compact+beta*sep+0.01*lb; loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt_.step(); sch.step()
            with torch.no_grad():
                for p,ep2 in zip(model.parameters(),ema.parameters()):
                    ep2.data.mul_(0.999).add_(p.data,alpha=0.001)
        ema.eval(); ps,ls2=[],[]
        with torch.no_grad():
            for batch in vl:
                b={k:v.to(device) for k,v in batch.items()}
                ps.extend(ema(b).argmax(1).cpu().numpy()); ls2.extend(b['label'].cpu().numpy())
        va=accuracy_score(ls2,ps)
        if va>bva: bva=va; bst={k:v.clone() for k,v in ema.state_dict().items()}

    ema.load_state_dict(bst); ema.eval()

    all_qdist=[]; all_labels=[]; all_preds=[]; all_entropy=[]
    with torch.no_grad():
        for batch in tel:
            b={k:v.to(device) for k,v in batch.items()}
            logits,qe,shared,qdist=ema(b,return_all=True)
            all_qdist.append(qdist.cpu()); all_labels.extend(b['label'].cpu().numpy())
            all_preds.extend(logits.argmax(1).cpu().numpy()); all_entropy.append(qe.cpu())
    qdist=torch.cat(all_qdist).numpy(); labels=np.array(all_labels); preds=np.array(all_preds)
    entropy=torch.cat(all_entropy).numpy()
    dom_quad=qdist.argmax(axis=1)
    quad_names=['Contempt','Envy','Pity','Admiration']

    print(f'\n--- {name} ---')
    print(f'Test ACC: {accuracy_score(labels,preds)*100:.1f}%')

    print(f'\n(a) Quadrant Distribution:')
    for q in range(4):
        hat_count=((dom_quad==q)&(labels==1)).sum()
        non_count=((dom_quad==q)&(labels==0)).sum()
        hat_pct=hat_count/max(1,(labels==1).sum())*100
        non_pct=non_count/max(1,(labels==0).sum())*100
        print(f'  {quad_names[q]:12s}: Hateful={hat_pct:5.1f}% Non-hate={non_pct:5.1f}%')

    print(f'\n(b) Per-Quadrant Accuracy:')
    for q in range(4):
        mask=(dom_quad==q)
        if mask.sum()>0:
            qacc=accuracy_score(labels[mask],preds[mask])*100
            print(f'  {quad_names[q]:12s}: n={mask.sum():4d} ACC={qacc:5.1f}% avg_weight={qdist[mask,q].mean():.3f}')

    print(f'\n(c) Entropy vs Accuracy:')
    for lo,hi,label in [(0,0.33,'Low'),(0.33,0.66,'Med'),(0.66,1.01,'High')]:
        mask=(entropy>=lo)&(entropy<hi)
        if mask.sum()>0:
            eacc=accuracy_score(labels[mask],preds[mask])*100
            print(f'  {label} entropy: n={mask.sum():4d} ACC={eacc:5.1f}%')

    # Clean up GPU
    del model, ema, qdist, labels, preds, entropy
    torch.cuda.empty_cache()

print('\n' + '='*80)
print('SCM EXTRACTION QUALITY')
print('='*80)

for ds_name, data_path in [('HateMM','./datasets/HateMM/scm_data.json'),
                            ('MHClip-Y','./datasets/Multihateclip/English/scm_data.json'),
                            ('MHClip-B','./datasets/Multihateclip/Chinese/scm_data.json'),
                            ('ImpliHateVid','./datasets/ImpliHateVid/scm_data.json')]:
    data=json.load(open(data_path))
    consistent=0; total=0
    for item in data:
        resp=item.get('scm_response',{})
        sp=resp.get('social_perception','').lower()
        w=resp.get('warmth_evidence','').lower()
        c=resp.get('competence_evidence','').lower()
        if not sp: continue
        total+=1
        is_cold = any(x in w for x in ['cold','hostile','threatening','dehumaniz','slur','contempt','negative'])
        is_warm = any(x in w for x in ['warm','friendly','positive','respect','admir','trust'])
        is_incomp = any(x in c for x in ['incompetent','foolish','primitive','backward','incapable','stupid'])
        is_comp = any(x in c for x in ['competent','capable','skilled','intelligent','smart'])
        predicted_quad = None
        if is_cold and is_incomp: predicted_quad = 'contempt'
        elif is_cold and is_comp: predicted_quad = 'envy'
        elif is_warm and is_incomp: predicted_quad = 'pity'
        elif is_warm and is_comp: predicted_quad = 'admiration'
        if predicted_quad and predicted_quad in sp:
            consistent+=1
        elif predicted_quad is None:
            total-=1
    if total>0:
        print(f'{ds_name}: {consistent}/{total} consistent ({consistent/total*100:.1f}%)')
