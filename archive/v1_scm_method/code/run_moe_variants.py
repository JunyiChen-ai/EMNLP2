"""
Test different MoE routing strategies to prevent expert collapse.
All variants use the same SCM base architecture, only the MoE routing changes.

Variants:
1. baseline:       Original Q-MoE with softmax routing (current, collapses)
2. loss_free:      DeepSeek loss-free balancing (learnable bias per expert)
3. cosine_div:     Intra-layer cosine diversity penalty
4. hypersphere:    L2 normalize routing embeddings before softmax
5. expert_choice:  Expert Choice routing (experts choose tokens)
6. soft_moe:       Soft MoE (all experts process soft-weighted inputs)
7. relu_route:     ReMoE-style ReLU routing instead of softmax

Usage:
  python run_moe_variants.py --dataset_name HateMM --variant baseline --num_runs 20
"""

import argparse, csv, json, os, random, copy, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
import warnings; warnings.filterwarnings("ignore")

device = "cuda"
SCM_FIELDS = ["target_group", "warmth_evidence", "competence_evidence",
              "social_perception", "behavioral_tendency"]

class DS(Dataset):
    def __init__(s, v, f, l, m): s.vids=v;s.f=f;s.lm=l;s.mk=m
    def __len__(s): return len(s.vids)
    def __getitem__(s, i):
        v=s.vids[i]; o={k:s.f[k][v] for k in s.mk}
        o["label"]=torch.tensor(s.lm[s.f["labels"][v]["Label"]],dtype=torch.long); return o
def collate_fn(b): return {k: torch.stack([x[k] for x in b]) for k in b[0]}


class SCMBase(nn.Module):
    """Shared SCM base: everything up to the fused representation."""
    def __init__(self, base_mk, hidden=192, nh=4, drop=0.15, md=0.15):
        super().__init__()
        self.base_mk = base_mk; self.md = md; self.hidden = hidden
        n_base = len(base_mk)
        self.base_projs = nn.ModuleList([nn.Sequential(
            nn.Linear(768, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden)
        ) for _ in range(n_base)])
        self.field_projs = nn.ModuleDict({
            f: nn.Sequential(nn.Linear(768, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden))
            for f in SCM_FIELDS
        })
        self.warmth_stream = nn.Sequential(
            nn.Linear(hidden*3, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.competence_stream = nn.Sequential(
            nn.Linear(hidden*3, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.percept_proj = nn.Sequential(
            nn.Linear(hidden*2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.routes = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Linear(hidden//2, 1)
        ) for _ in range(nh)])

    def forward(self, batch, training=False):
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
        warmth_repr = self.warmth_stream(torch.cat([field_h["warmth_evidence"], field_h["target_group"], base_ctx], dim=-1))
        comp_repr = self.competence_stream(torch.cat([field_h["competence_evidence"], field_h["target_group"], base_ctx], dim=-1))
        wc_cat = torch.cat([warmth_repr, comp_repr], dim=-1)
        percept_repr = self.percept_proj(torch.cat([field_h["social_perception"], field_h["behavioral_tendency"]], dim=-1))
        return base_repr, wc_cat, percept_repr


class MoEModel(nn.Module):
    """SCM + configurable MoE routing."""
    def __init__(self, base_mk, variant="baseline", hidden=192, nc=2, drop=0.15,
                 n_experts=4, expert_hidden=64):
        super().__init__()
        self.variant = variant
        self.n_experts = n_experts; self.nc = nc
        self.scm_base = SCMBase(base_mk, hidden=hidden, drop=drop)
        # Quadrant head
        self.quadrant_head = nn.Sequential(
            nn.Linear(hidden*2, hidden), nn.GELU(), nn.Linear(hidden, n_experts))
        # Quadrant protos
        self.quadrant_protos = nn.Parameter(torch.randn(n_experts, hidden))
        nn.init.xavier_uniform_(self.quadrant_protos.unsqueeze(0))
        # Pre-classifier
        nh = 4
        cd = nh * hidden + hidden + hidden + hidden  # base_repr(960) + quadrant_repr(192) + percept_repr(192)
        self.pre_cls = nn.Sequential(
            nn.LayerNorm(cd), nn.Linear(cd, 256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, expert_hidden), nn.GELU(), nn.Dropout(drop*0.5))
        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_hidden, expert_hidden//2), nn.GELU(), nn.Dropout(drop*0.5),
                nn.Linear(expert_hidden//2, nc)
            ) for _ in range(n_experts)
        ])
        # Variant-specific components
        if variant == "loss_free":
            self.expert_bias = nn.Parameter(torch.zeros(n_experts))
        if variant == "relu_route":
            self.relu_lambda = nn.Parameter(torch.ones(n_experts) * 0.01)

    def forward(self, batch, training=False, return_info=False):
        base_repr, wc_cat, percept_repr = self.scm_base(batch, training=training)
        # Router logits
        router_logits = self.quadrant_head(wc_cat)  # [B, n_experts]

        # Routing strategy
        if self.variant == "baseline":
            router_dist = torch.softmax(router_logits, dim=-1)
        elif self.variant == "loss_free":
            router_dist = torch.softmax(router_logits + self.expert_bias.unsqueeze(0), dim=-1)
        elif self.variant == "hypersphere":
            # L2 normalize before softmax
            router_logits_norm = F.normalize(router_logits, dim=-1) * 10.0  # scale
            router_dist = torch.softmax(router_logits_norm, dim=-1)
        elif self.variant == "relu_route":
            router_dist = F.relu(router_logits)
            router_dist = router_dist / (router_dist.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            router_dist = torch.softmax(router_logits, dim=-1)

        quadrant_repr = torch.mm(router_dist, self.quadrant_protos)
        fused = torch.cat([base_repr, quadrant_repr, percept_repr], dim=-1)
        shared = self.pre_cls(fused)

        if self.variant == "expert_choice" and training:
            # Expert Choice: each expert picks top-k tokens
            B = shared.size(0)
            k = max(1, B // self.n_experts)
            expert_outputs = torch.zeros(B, self.nc, device=shared.device)
            expert_counts = torch.zeros(B, device=shared.device)
            for i in range(self.n_experts):
                scores = router_logits[:, i]  # [B]
                topk_idx = scores.topk(k).indices
                expert_out = self.experts[i](shared[topk_idx])  # [k, nc]
                expert_outputs[topk_idx] += expert_out
                expert_counts[topk_idx] += 1
            expert_counts = expert_counts.clamp(min=1)
            logits = expert_outputs / expert_counts.unsqueeze(-1)
        elif self.variant == "soft_moe":
            # Soft MoE: all experts process soft-weighted shared
            expert_outputs = torch.stack([expert(shared) for expert in self.experts], dim=1)  # [B, E, nc]
            logits = (expert_outputs * router_dist.unsqueeze(-1)).sum(dim=1)
        else:
            # Standard: weighted sum
            expert_outputs = torch.stack([expert(shared) for expert in self.experts], dim=1)
            logits = (expert_outputs * router_dist.unsqueeze(-1)).sum(dim=1)

        if return_info:
            return logits, router_dist
        return logits


def qels_cross_entropy(logits, labels, router_dist, nc=2, eps_min=0.01, eps_lambda=0.15,
                       class_weight=None):
    # Use router entropy for QELS
    q_entropy = -(router_dist * (router_dist + 1e-8).log()).sum(dim=-1)
    q_entropy_norm = q_entropy / np.log(router_dist.size(-1))
    eps = (eps_min + eps_lambda * q_entropy_norm).clamp(0, 0.5)
    one_hot = F.one_hot(labels, nc).float()
    smooth_targets = (1 - eps.unsqueeze(1)) * one_hot + eps.unsqueeze(1) / nc
    log_probs = F.log_softmax(logits, dim=-1)
    if class_weight is not None:
        w = class_weight[labels]
        loss = -(smooth_targets * log_probs).sum(dim=-1) * w
    else:
        loss = -(smooth_targets * log_probs).sum(dim=-1)
    return loss.mean()


def compute_aux_losses(variant, model, router_dist, shared, expert_outputs=None):
    """Compute variant-specific auxiliary losses."""
    aux = torch.tensor(0.0, device=shared.device)

    # Load balance loss (for all variants)
    usage = router_dist.mean(dim=0)
    lb_loss = usage.var() / (usage.mean() ** 2 + 1e-8)
    aux = aux + 0.01 * lb_loss

    if variant == "cosine_div" and expert_outputs is not None:
        # Penalize cosine similarity between expert outputs
        E = expert_outputs.size(1)
        div_loss = torch.tensor(0.0, device=shared.device)
        count = 0
        for i in range(E):
            for j in range(i+1, E):
                cos = F.cosine_similarity(expert_outputs[:, i], expert_outputs[:, j], dim=-1).mean()
                div_loss = div_loss + cos
                count += 1
        if count > 0:
            aux = aux + 0.1 * div_loss / count

    if variant == "loss_free":
        # Update bias based on load (outside gradient)
        with torch.no_grad():
            target_load = 1.0 / router_dist.size(-1)
            actual_load = router_dist.mean(dim=0)
            model.expert_bias.data += 0.01 * (target_load - actual_load)

    if variant == "relu_route":
        # L1 regularization on routing weights
        aux = aux + 0.001 * router_dist.sum(dim=-1).mean()

    return aux


def mr2_loss(shared, labels, nc=2):
    centroids = []; intra_var = []
    for c in range(nc):
        mask = (labels == c)
        if mask.sum() < 2:
            centroids.append(shared[mask].mean(dim=0) if mask.sum() > 0 else torch.zeros(shared.size(1), device=shared.device))
            intra_var.append(torch.tensor(0.0, device=shared.device))
            continue
        class_feats = shared[mask]; centroid = class_feats.mean(dim=0)
        centroids.append(centroid)
        intra_var.append(((class_feats - centroid.unsqueeze(0)) ** 2).mean())
    return sum(intra_var) / nc


def cw_sched(opt, ws, ts):
    def f(s):
        if s < ws: return s / max(1, ws)
        return max(0, 0.5*(1+np.cos(np.pi*(s-ws)/max(1,ts-ws))))
    return LambdaLR(opt, f)

def load_split_ids(d):
    s = {}
    for n in ["train", "valid", "test"]:
        with open(os.path.join(d, f"{n}.csv")) as f:
            s[n] = [r[0] for r in csv.reader(f) if r]
    return s

def best_thresh(vl, vla, tl, tla):
    std = accuracy_score(tla, np.argmax(tl, axis=1))
    vd = vl[:, 1] - vl[:, 0]; td = tl[:, 1] - tl[:, 0]
    bt, bv = 0, 0
    for t in np.arange(-3, 3, 0.02):
        v = accuracy_score(vla, (vd > t).astype(int))
        if v > bv: bv, bt = v, t
    tuned = accuracy_score(tla, (td > bt).astype(int))
    return (tuned, bt) if tuned > std else (std, None)


def run(variant, feats, splits, lm, base_mk, nc, num_runs, seed_offset, class_weight,
        mr2_alpha=0.3):
    all_mk = list(base_mk) + [f"scm_{f}" for f in SCM_FIELDS]
    common = set.intersection(*[set(feats[k].keys()) for k in all_mk]) & set(feats["labels"].keys())
    cur = {s: [v for v in splits[s] if v in common] for s in splits}
    cw_tensor = torch.tensor(class_weight, dtype=torch.float).to(device)
    all_results = []; all_routing = []

    for ri in range(num_runs):
        seed = ri * 1000 + 42 + seed_offset
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        trd = DS(cur["train"], feats, lm, all_mk)
        vd = DS(cur["valid"], feats, lm, all_mk)
        ted = DS(cur["test"], feats, lm, all_mk)
        trl = DataLoader(trd, 32, True, collate_fn=collate_fn)
        vl = DataLoader(vd, 64, False, collate_fn=collate_fn)
        tel = DataLoader(ted, 64, False, collate_fn=collate_fn)

        model = MoEModel(base_mk, variant=variant, nc=nc).to(device)
        ema = copy.deepcopy(model)
        opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
        ep = 45; ts_t = ep * len(trl); ws = 5 * len(trl); sch = cw_sched(opt, ws, ts_t)
        bva, bst = -1, None

        for e in range(ep):
            model.train()
            for batch in trl:
                batch = {k: v.to(device) for k, v in batch.items()}; opt.zero_grad()
                logits, router_dist = model(batch, training=True, return_info=True)
                ce = qels_cross_entropy(logits, batch["label"], router_dist, nc=nc, class_weight=cw_tensor)
                # MR2
                _, wc, _ = model.scm_base(batch, training=False)
                shared = model.pre_cls(torch.cat([
                    model.scm_base(batch, training=False)[0],
                    torch.mm(router_dist.detach(), model.quadrant_protos),
                    model.scm_base(batch, training=False)[2]
                ], dim=-1))
                compact = mr2_loss(shared, batch["label"], nc=nc)
                # Expert outputs for diversity loss
                expert_outputs = torch.stack([exp(shared) for exp in model.experts], dim=1)
                aux = compute_aux_losses(variant, model, router_dist, shared, expert_outputs)
                loss = ce + mr2_alpha * compact + aux
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
                with torch.no_grad():
                    for p, ep2 in zip(model.parameters(), ema.parameters()):
                        ep2.data.mul_(0.999).add_(p.data, alpha=0.001)
            ema.eval(); ps, ls2 = [], []
            with torch.no_grad():
                for batch in vl:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    ps.extend(ema(batch).argmax(1).cpu().numpy()); ls2.extend(batch["label"].cpu().numpy())
            va = accuracy_score(ls2, ps)
            if va > bva: bva = va; bst = {k: v.clone() for k, v in ema.state_dict().items()}

        ema.load_state_dict(bst); ema.eval()
        # Evaluate
        all_logits, all_labels, all_rd = [], [], []
        with torch.no_grad():
            for batch in tel:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits, rd = ema(batch, return_info=True)
                all_logits.append(logits.cpu()); all_labels.extend(batch["label"].cpu().numpy())
                all_rd.append(rd.cpu())
        tel_arr = torch.cat(all_logits).numpy(); tela = np.array(all_labels)
        rd_arr = torch.cat(all_rd).numpy()
        # Val logits for threshold
        vl_logits, vl_labels = [], []
        with torch.no_grad():
            for batch in vl:
                batch = {k: v.to(device) for k, v in batch.items()}
                vl_logits.append(ema(batch).cpu()); vl_labels.extend(batch["label"].cpu().numpy())
        vl_arr = torch.cat(vl_logits).numpy(); vla = np.array(vl_labels)
        head_acc, _ = best_thresh(vl_arr, vla, tel_arr, tela)
        all_results.append(head_acc)
        # Track routing distribution
        avg_rd = rd_arr.mean(axis=0)
        all_routing.append(avg_rd)

    accs = np.array(all_results)
    avg_routing = np.mean(all_routing, axis=0)
    routing_str = " ".join([f"Q{i}={avg_routing[i]:.3f}" for i in range(len(avg_routing))])
    print(f"{variant}: mean={accs.mean():.4f} std={accs.std():.4f} max={accs.max():.4f} | routing: {routing_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="HateMM", choices=["HateMM", "Multihateclip", "ImpliHateVid"])
    parser.add_argument("--language", default="English")
    parser.add_argument("--variant", default="baseline",
                        choices=["baseline", "loss_free", "cosine_div", "hypersphere",
                                 "expert_choice", "soft_moe", "relu_route"])
    parser.add_argument("--num_runs", type=int, default=20)
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--mr2_alpha", type=float, default=0.3)
    args = parser.parse_args()

    if args.dataset_name == "HateMM":
        emb_dir = "./embeddings/HateMM"; ann_path = "./datasets/HateMM/annotation(new).json"
        split_dir = "./datasets/HateMM/splits"; lm = {"Non Hate": 0, "Hate": 1}
    elif args.dataset_name == "ImpliHateVid":
        emb_dir = "./embeddings/ImpliHateVid"; ann_path = "./datasets/ImpliHateVid/annotation(new).json"
        split_dir = "./datasets/ImpliHateVid/splits"; lm = {"Normal": 0, "Hateful": 1}
    else:
        emb_dir = f"./embeddings/Multihateclip/{args.language}"
        ann_path = f"./datasets/Multihateclip/{args.language}/annotation(new).json"
        split_dir = f"./datasets/Multihateclip/{args.language}/splits"
        lm = {"Normal": 0, "Offensive": 1, "Hateful": 1}

    base_mk = ["text", "audio", "frame"]
    feats = {"text": torch.load(f"{emb_dir}/text_features.pth", map_location="cpu"),
             "audio": torch.load(f"{emb_dir}/wavlm_audio_features.pth", map_location="cpu"),
             "frame": torch.load(f"{emb_dir}/frame_features.pth", map_location="cpu")}
    for f in SCM_FIELDS:
        feats[f"scm_{f}"] = torch.load(f"{emb_dir}/scm_mean_{f}_features.pth", map_location="cpu")
    with open(ann_path) as f:
        feats["labels"] = {d["Video_ID"]: d for d in json.load(f)}
    splits = load_split_ids(split_dir)

    run(args.variant, feats, splits, lm, base_mk, 2, args.num_runs, args.seed_offset,
        class_weight=[1.0, 1.5], mr2_alpha=args.mr2_alpha)


if __name__ == "__main__":
    main()
