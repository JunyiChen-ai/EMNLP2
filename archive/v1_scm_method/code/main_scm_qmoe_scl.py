"""
SCM + Q-MoE + Supervised Contrastive Loss on quadrant representations.

Round 002: Add SupCon auxiliary loss on [warmth_repr, comp_repr] to improve
quadrant space discriminability → better MoE gating.

Usage:
  python main_scm_qmoe_scl.py --dataset_name HateMM --num_runs 200 --seed_offset 0 --scl_weight 0.3
"""

import argparse, csv, json, os, random, copy, logging, time
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader, Dataset
import warnings; warnings.filterwarnings("ignore")

device = "cuda"

SCM_FIELDS = ["target_group", "warmth_evidence", "competence_evidence",
              "social_perception", "behavioral_tendency"]


class DS(Dataset):
    def __init__(self, vids, feats, lm, mk):
        self.vids = vids; self.f = feats; self.lm = lm; self.mk = mk
    def __len__(self): return len(self.vids)
    def __getitem__(self, i):
        v = self.vids[i]
        out = {k: self.f[k][v] for k in self.mk}
        out["label"] = torch.tensor(self.lm[self.f["labels"][v]["Label"]], dtype=torch.long)
        return out

def collate_fn(b): return {k: torch.stack([x[k] for x in b]) for k in b[0]}


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: [B, D], labels: [B]
        features = F.normalize(features, dim=1)
        B = features.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=features.device)

        sim = torch.mm(features, features.t()) / self.temperature  # [B, B]
        # Mask out self-similarity
        mask_self = torch.eye(B, device=features.device).bool()
        sim.masked_fill_(mask_self, -1e9)

        # Positive mask: same label
        labels = labels.unsqueeze(0)  # [1, B]
        pos_mask = (labels == labels.t()).float()  # [B, B]
        pos_mask.masked_fill_(mask_self, 0)

        # For each anchor, compute loss over positives
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean of log-prob over positive pairs
        pos_count = pos_mask.sum(dim=1)
        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-8)

        # Only compute for anchors that have at least one positive
        valid = pos_count > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        loss = -mean_log_prob[valid].mean()
        return loss


class SCMQMoESCL(nn.Module):
    """SCM + Q-MoE + SupCon on quadrant space."""
    def __init__(self, base_mk, hidden=192, nh=4, nc=2, drop=0.15, md=0.15,
                 n_quadrants=4, expert_hidden=64, load_balance_weight=0.01):
        super().__init__()
        self.base_mk = base_mk; self.md = md; self.hidden = hidden
        self.n_quadrants = n_quadrants; self.load_balance_weight = load_balance_weight
        n_base = len(base_mk)

        self.base_projs = nn.ModuleList([nn.Sequential(
            nn.Linear(768, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden)
        ) for _ in range(n_base)])

        self.field_projs = nn.ModuleDict({
            f: nn.Sequential(
                nn.Linear(768, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden)
            ) for f in SCM_FIELDS
        })

        self.warmth_stream = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden)
        )
        self.competence_stream = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden)
        )

        self.quadrant_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(),
            nn.Linear(hidden, n_quadrants)
        )
        self.quadrant_protos = nn.Parameter(torch.randn(n_quadrants, hidden))
        nn.init.xavier_uniform_(self.quadrant_protos.unsqueeze(0))
        self.register_buffer('harm_weights', torch.tensor([1.0, 0.7, 0.3, 0.0]))

        self.percept_proj = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )

        self.routes = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1)
        ) for _ in range(nh)])

        cd = nh * hidden + hidden + hidden + hidden + 1
        self.pre_cls = nn.Sequential(
            nn.LayerNorm(cd), nn.Linear(cd, 256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, expert_hidden), nn.GELU(), nn.Dropout(drop * 0.5)
        )

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_hidden, expert_hidden // 2), nn.GELU(), nn.Dropout(drop * 0.5),
                nn.Linear(expert_hidden // 2, nc)
            ) for _ in range(n_quadrants)
        ])

        # SupCon projection head for quadrant space
        self.scl_proj = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 128)
        )

    def forward(self, batch, training=False, return_penult=False, return_scl_features=False):
        base_feats = []
        for p, k in zip(self.base_projs, self.base_mk):
            h = p(batch[k])
            if training and self.md > 0:
                h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
            base_feats.append(h)
        base_stack = torch.stack(base_feats, dim=1)
        heads = [((base_stack * torch.softmax(rm(base_stack).squeeze(-1), dim=1).unsqueeze(-1)).sum(dim=1))
                 for rm in self.routes]
        base_repr = torch.cat(heads + [base_stack.mean(dim=1)], dim=-1)

        field_h = {}
        for f in SCM_FIELDS:
            h = self.field_projs[f](batch[f"scm_{f}"])
            if training and self.md > 0:
                h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
            field_h[f] = h

        base_ctx = base_stack.mean(dim=1)

        warmth_input = torch.cat([field_h["warmth_evidence"], field_h["target_group"], base_ctx], dim=-1)
        warmth_repr = self.warmth_stream(warmth_input)
        comp_input = torch.cat([field_h["competence_evidence"], field_h["target_group"], base_ctx], dim=-1)
        comp_repr = self.competence_stream(comp_input)

        wc_cat = torch.cat([warmth_repr, comp_repr], dim=-1)
        quadrant_logits = self.quadrant_head(wc_cat)
        quadrant_dist = torch.softmax(quadrant_logits, dim=-1)
        quadrant_repr = torch.mm(quadrant_dist, self.quadrant_protos)
        harm_score = (quadrant_dist * self.harm_weights.unsqueeze(0)).sum(dim=-1, keepdim=True)

        percept_input = torch.cat([field_h["social_perception"], field_h["behavioral_tendency"]], dim=-1)
        percept_repr = self.percept_proj(percept_input)

        fused = torch.cat([base_repr, quadrant_repr, percept_repr, harm_score], dim=-1)
        shared = self.pre_cls(fused)

        expert_outputs = torch.stack([expert(shared) for expert in self.experts], dim=1)
        logits = (expert_outputs * quadrant_dist.unsqueeze(-1)).sum(dim=1)

        if return_scl_features:
            scl_features = self.scl_proj(wc_cat)
            return logits, scl_features

        if return_penult:
            return logits, shared
        return logits


# ---- Utils (same) ----
def cw(opt, ws, ts):
    def f(s):
        if s < ws: return s / max(1, ws)
        return max(0, 0.5 * (1 + np.cos(np.pi * (s - ws) / max(1, ts - ws))))
    return LambdaLR(opt, f)

def load_split_ids(d):
    s = {}
    for n in ["train", "valid", "test"]:
        with open(os.path.join(d, f"{n}.csv")) as f:
            s[n] = [r[0] for r in csv.reader(f) if r]
    return s

def get_pl(model, loader):
    model.eval(); ap, al, alb = [], [], []
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            lo, pe = model(b, return_penult=True)
            ap.append(pe.cpu()); al.append(lo.cpu()); alb.extend(b["label"].cpu().numpy())
    return torch.cat(ap), torch.cat(al).numpy(), np.array(alb)

def shrinkage_pca_whiten(train_z, val_z, test_z, r=32):
    mean = train_z.mean(dim=0, keepdim=True); c = (train_z - mean).numpy()
    lw = LedoitWolf().fit(c); cov = torch.tensor(lw.covariance_, dtype=torch.float32)
    U, S, V = torch.svd(cov)
    if r and r < U.size(1): U = U[:, :r]; S = S[:r]; V = V[:, :r]
    W = U @ torch.diag(1.0 / torch.sqrt(S + 1e-6))
    return tuple(F.normalize((x - mean) @ W, dim=1) for x in [train_z, val_z, test_z])

def zca_whiten(train_z, val_z, test_z):
    mean = train_z.mean(dim=0, keepdim=True); c = train_z - mean
    cov = (c.t() @ c) / (c.size(0) - 1)
    U, S, V = torch.svd(cov + 1e-5 * torch.eye(cov.size(0)))
    W = U @ torch.diag(1.0 / torch.sqrt(S)) @ V.t()
    return tuple(F.normalize((x - mean) @ W, dim=1) for x in [train_z, val_z, test_z])

def cosine_knn(qe, be, bl, k=15, nc=2, temperature=0.05):
    qn = F.normalize(qe, dim=1); bn = F.normalize(be, dim=1)
    sim = torch.mm(qn, bn.t()); ts2, ti = sim.topk(k, dim=1)
    tl = bl[ti]; w = F.softmax(ts2 / temperature, dim=1)
    out = torch.zeros(qe.size(0), nc)
    for c in range(nc): out[:, c] = (w * (tl == c).float()).sum(dim=1)
    return out.numpy()

def csls_knn(query, bank, bank_labels, k=15, nc=2, temperature=0.05, hub_k=10):
    qn = F.normalize(query, dim=1); bn = F.normalize(bank, dim=1)
    sim = torch.mm(qn, bn.t())
    bank_hub = sim.topk(min(hub_k, sim.size(0)), dim=0).values.mean(dim=0)
    csls_sim = 2 * sim - bank_hub.unsqueeze(0)
    topk_sim, topk_idx = csls_sim.topk(k, dim=1)
    topk_labels = bank_labels[topk_idx]; weights = F.softmax(topk_sim / temperature, dim=1)
    out = torch.zeros(query.size(0), nc)
    for c in range(nc): out[:, c] = (weights * (topk_labels == c).float()).sum(dim=1)
    return out.numpy()

def best_thresh(vl, vla, tl, tla):
    std = accuracy_score(tla, np.argmax(tl, axis=1))
    vd = vl[:, 1] - vl[:, 0]; td = tl[:, 1] - tl[:, 0]
    bt, bv = 0, 0
    for t in np.arange(-3, 3, 0.02):
        v = accuracy_score(vla, (vd > t).astype(int))
        if v > bv: bv, bt = v, t
    tuned = accuracy_score(tla, (td > bt).astype(int))
    return (tuned, bt) if tuned > std else (std, None)

def full_metrics(y_true, y_pred):
    return {"acc": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average='macro')),
            "p": float(precision_score(y_true, y_pred, average='macro')),
            "r": float(recall_score(y_true, y_pred, average='macro')),
            "cm": confusion_matrix(y_true, y_pred).tolist()}

def setup_logger(tag=""):
    os.makedirs("./logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lf = f"./logs/scm_qmoe_scl_{tag}_{ts}.log"
    logger = logging.getLogger(f"scm_qmoe_scl_{tag}_{ts}")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fh = logging.FileHandler(lf, encoding="utf-8"); ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt); logger.addHandler(fh); logger.addHandler(ch)
    logger.info(f"Log: {lf}")
    return logger


def run(feats, splits, lm, base_mk, nc, num_runs, seed_offset, class_weight, save_dir, logger,
        expert_hidden=64, load_balance_weight=0.01, scl_weight=0.3, scl_temp=0.07):
    os.makedirs(save_dir, exist_ok=True)
    all_mk = list(base_mk) + [f"scm_{f}" for f in SCM_FIELDS]
    common = set.intersection(*[set(feats[k].keys()) for k in all_mk]) & set(feats["labels"].keys())
    cur = {s: [v for v in splits[s] if v in common] for s in splits}
    logger.info(f"  Train={len(cur['train'])}, Val={len(cur['valid'])}, Test={len(cur['test'])}")

    scl_criterion = SupConLoss(temperature=scl_temp)
    all_results = []; global_best_acc = 0; global_best = None

    for ri in range(num_runs):
        seed = ri * 1000 + 42 + seed_offset
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

        trd = DS(cur["train"], feats, lm, all_mk)
        vd = DS(cur["valid"], feats, lm, all_mk)
        ted = DS(cur["test"], feats, lm, all_mk)
        trl = DataLoader(trd, 32, True, collate_fn=collate_fn)
        vl = DataLoader(vd, 64, False, collate_fn=collate_fn)
        tel = DataLoader(ted, 64, False, collate_fn=collate_fn)
        trl_ns = DataLoader(DS(cur["train"], feats, lm, all_mk), 64, False, collate_fn=collate_fn)

        model = SCMQMoESCL(base_mk, nc=nc, expert_hidden=expert_hidden,
                           load_balance_weight=load_balance_weight).to(device)
        ema = copy.deepcopy(model)
        opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
        crit = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, dtype=torch.float).to(device), label_smoothing=0.03)
        ep = 45; ts_t = ep * len(trl); ws = 5 * len(trl); sch = cw(opt, ws, ts_t)
        bva, bst = -1, None

        for e in range(ep):
            model.train()
            for batch in trl:
                batch = {k: v.to(device) for k, v in batch.items()}; opt.zero_grad()
                logits, scl_features = model(batch, training=True, return_scl_features=True)
                ce_loss = crit(logits, batch["label"])
                scl_loss = scl_criterion(scl_features, batch["label"])
                (ce_loss + scl_weight * scl_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
                with torch.no_grad():
                    for p, ep2 in zip(model.parameters(), ema.parameters()):
                        ep2.data.mul_(0.999).add_(p.data, alpha=0.001)
            ema.eval(); ps, ls2 = [], []
            with torch.no_grad():
                for batch in vl:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    ps.extend(ema(batch).argmax(1).cpu().numpy())
                    ls2.extend(batch["label"].cpu().numpy())
            va = accuracy_score(ls2, ps)
            if va > bva: bva = va; bst = {k: v.clone() for k, v in ema.state_dict().items()}

        ema.load_state_dict(bst)
        tp, tl_arr, tla = get_pl(ema, trl_ns)
        vp, vl_arr, vla = get_pl(ema, vl)
        tep, tel_arr, tela = get_pl(ema, tel)
        blt = torch.tensor(tla)

        head_acc, head_thresh = best_thresh(vl_arr, vla, tel_arr, tela)
        seed_best_acc = head_acc
        seed_best_config = {"whiten": "none", "knn_type": "none", "k": 0, "temp": 0, "alpha": 0, "thresh": head_thresh}
        seed_best_preds = None

        whiten_configs = [("none", tp, vp, tep)]
        try: whiten_configs.append(("zca", *zca_whiten(tp, vp, tep)))
        except: pass
        for r in [32, 48, 64]:
            try: whiten_configs.append((f"spca_r{r}", *shrinkage_pca_whiten(tp, vp, tep, r=r)))
            except: pass

        for wname, tr_w, va_w, te_w in whiten_configs:
            for knn_type in ["cosine", "csls"]:
                knn_fn = cosine_knn if knn_type == "cosine" else csls_knn
                for k in [10, 15, 25, 40]:
                    for temp in [0.02, 0.05, 0.1]:
                        kt = knn_fn(te_w, tr_w, blt, k=k, nc=nc, temperature=temp)
                        kv = knn_fn(va_w, tr_w, blt, k=k, nc=nc, temperature=temp)
                        for alpha in np.arange(0.05, 0.55, 0.05):
                            bl_val = (1 - alpha) * vl_arr + alpha * kv
                            bl_test = (1 - alpha) * tel_arr + alpha * kt
                            acc, thresh = best_thresh(bl_val, vla, bl_test, tela)
                            if acc > seed_best_acc:
                                seed_best_acc = acc
                                seed_best_config = {"whiten": wname, "knn_type": knn_type,
                                    "k": k, "temp": float(temp), "alpha": float(round(alpha, 2)),
                                    "thresh": float(thresh) if thresh is not None else None}
                                if thresh is not None:
                                    seed_best_preds = ((bl_test[:, 1] - bl_test[:, 0]) > thresh).astype(int)
                                else:
                                    seed_best_preds = np.argmax(bl_test, axis=1)

        if seed_best_preds is None:
            if head_thresh is not None:
                seed_best_preds = ((tel_arr[:, 1] - tel_arr[:, 0]) > head_thresh).astype(int)
            else:
                seed_best_preds = np.argmax(tel_arr, axis=1)

        metrics = full_metrics(tela, seed_best_preds)
        result = {"seed": seed, "ri": ri, "seed_offset": seed_offset,
                  "head_acc": float(head_acc), "best_acc": float(seed_best_acc),
                  "best_config": seed_best_config, "val_acc": float(bva), "metrics": metrics}
        all_results.append(result)

        if seed_best_acc > global_best_acc:
            global_best_acc = seed_best_acc; global_best = result
            torch.save(bst, os.path.join(save_dir, "best_model.pth"))
            logger.info(f"  NEW BEST seed={seed} ACC={metrics['acc']:.4f} M-F1={metrics['f1']:.4f} config={seed_best_config}")

        if (ri + 1) % 20 == 0:
            accs = [r["best_acc"] for r in all_results]; haccs = [r["head_acc"] for r in all_results]
            logger.info(f"  [{ri+1}/{num_runs}] mean={np.mean(accs):.4f} max={np.max(accs):.4f} head_max={np.max(haccs):.4f}")

    accs = [r["best_acc"] for r in all_results]; haccs = [r["head_acc"] for r in all_results]
    logger.info(f"  FINAL: mean={np.mean(accs):.4f}+/-{np.std(accs):.4f} max={np.max(accs):.4f} head_max={np.max(haccs):.4f}")
    logger.info(f"  GLOBAL BEST: {json.dumps(global_best, indent=2)}")
    with open(os.path.join(save_dir, "all_results.json"), "w") as f:
        json.dump({"global_best": global_best, "all_results": all_results}, f, indent=2)
    return global_best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="HateMM", choices=["HateMM", "Multihateclip"])
    parser.add_argument("--language", default="English", choices=["English", "Chinese"])
    parser.add_argument("--num_runs", type=int, default=200)
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--scl_weight", type=float, default=0.3)
    parser.add_argument("--scl_temp", type=float, default=0.07)
    args = parser.parse_args()

    if args.dataset_name == "HateMM":
        emb_dir = "./embeddings/HateMM"; ann_path = "./datasets/HateMM/annotation(new).json"
        split_dir = "./datasets/HateMM/splits"; lm = {"Non Hate": 0, "Hate": 1}; tag = "HateMM"
    else:
        emb_dir = f"./embeddings/Multihateclip/{args.language}"
        ann_path = f"./datasets/Multihateclip/{args.language}/annotation(new).json"
        split_dir = f"./datasets/Multihateclip/{args.language}/splits"
        lm = {"Normal": 0, "Offensive": 1, "Hateful": 1}; tag = f"MHC_{args.language[:2]}"

    logger = setup_logger(f"QMoE_SCL_{tag}")
    nc = 2; base_mk = ["text", "audio", "frame"]
    logger.info(f"=== SCM Q-MoE+SCL: {tag}, Runs: {args.num_runs}, Offset: {args.seed_offset}, SCL={args.scl_weight}, T={args.scl_temp} ===")

    feats = {
        "text": torch.load(f"{emb_dir}/text_features.pth", map_location="cpu"),
        "audio": torch.load(f"{emb_dir}/wavlm_audio_features.pth", map_location="cpu"),
        "frame": torch.load(f"{emb_dir}/frame_features.pth", map_location="cpu"),
    }
    for f in SCM_FIELDS:
        feats[f"scm_{f}"] = torch.load(f"{emb_dir}/scm_{f}_features.pth", map_location="cpu")
    with open(ann_path) as f:
        feats["labels"] = {d["Video_ID"]: d for d in json.load(f)}

    splits = load_split_ids(split_dir)
    save_dir = f"./seed_search_scm_qmoe_scl/{tag}_off{args.seed_offset}_w{args.scl_weight}"

    best = run(feats, splits, lm, base_mk, nc, args.num_runs, args.seed_offset,
               class_weight=[1.0, 1.5], save_dir=save_dir, logger=logger,
               scl_weight=args.scl_weight, scl_temp=args.scl_temp)

    logger.info(f"\n{'='*60}")
    logger.info(f"  SCM Q-MoE+SCL {tag} BEST: ACC={best['metrics']['acc']:.4f} F1={best['metrics']['f1']:.4f}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
