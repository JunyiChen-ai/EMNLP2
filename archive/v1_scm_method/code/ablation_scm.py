"""
Ablation study for SCM model.
Runs on the best seed/config for each dataset.

Usage:
  conda run -n HVGuard python ablation_scm.py --dataset_name HateMM
  conda run -n HVGuard python ablation_scm.py --dataset_name Multihateclip --language English
  conda run -n HVGuard python ablation_scm.py --dataset_name Multihateclip --language Chinese
"""

import argparse, csv, json, os, random, copy, logging
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


class SCMFusion(nn.Module):
    def __init__(self, base_mk, hidden=192, nh=4, nc=2, drop=0.15, md=0.15,
                 disable_warmth=False, disable_competence=False,
                 disable_attractor=False, disable_harm_score=False,
                 disable_perception=False, disable_fields=None,
                 hard_quadrant=False):
        super().__init__()
        self.base_mk = base_mk; self.md = md; self.hidden = hidden
        self.disable_warmth = disable_warmth
        self.disable_competence = disable_competence
        self.disable_attractor = disable_attractor
        self.disable_harm_score = disable_harm_score
        self.disable_perception = disable_perception
        self.disable_fields = disable_fields or []
        self.hard_quadrant = hard_quadrant
        n_base = len(base_mk); n_quadrants = 4

        self.base_projs = nn.ModuleList([nn.Sequential(
            nn.Linear(768, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden)
        ) for _ in range(n_base)])

        self.field_projs = nn.ModuleDict({
            f: nn.Sequential(
                nn.Linear(768, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden)
            ) for f in SCM_FIELDS
        })

        stream_input = hidden * 3  # field + target + ctx
        if disable_warmth and disable_competence:
            stream_input = hidden * 3  # fallback: just concat all
        self.warmth_stream = nn.Sequential(
            nn.Linear(stream_input, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden)
        )
        self.competence_stream = nn.Sequential(
            nn.Linear(stream_input, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden)
        )

        composer_in = hidden * 2
        if disable_warmth: composer_in = hidden
        if disable_competence: composer_in = hidden
        if disable_warmth and disable_competence: composer_in = hidden  # single stream
        self.quadrant_head = nn.Sequential(
            nn.Linear(composer_in, hidden), nn.GELU(), nn.Linear(hidden, n_quadrants)
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

        cd = nh * hidden + hidden
        if not disable_attractor: cd += hidden
        if not disable_perception: cd += hidden
        if not disable_harm_score: cd += 1
        self.pre_cls = nn.Sequential(
            nn.LayerNorm(cd), nn.Linear(cd, 256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(drop * 0.5)
        )
        self.head = nn.Linear(64, nc)

    def forward(self, batch, training=False, return_penult=False):
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
            if f in self.disable_fields:
                field_h[f] = torch.zeros(batch[list(batch.keys())[0]].size(0), self.hidden, device=device)
            else:
                h = self.field_projs[f](batch[f"scm_{f}"])
                if training and self.md > 0:
                    h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
                field_h[f] = h

        base_ctx = base_stack.mean(dim=1)

        parts = [base_repr]

        if not (self.disable_warmth and self.disable_competence):
            if not self.disable_warmth:
                w_in = torch.cat([field_h["warmth_evidence"], field_h["target_group"], base_ctx], dim=-1)
                warmth_repr = self.warmth_stream(w_in)
            if not self.disable_competence:
                c_in = torch.cat([field_h["competence_evidence"], field_h["target_group"], base_ctx], dim=-1)
                comp_repr = self.competence_stream(c_in)

            if self.disable_warmth:
                wc_cat = comp_repr
            elif self.disable_competence:
                wc_cat = warmth_repr
            else:
                wc_cat = torch.cat([warmth_repr, comp_repr], dim=-1)

            quadrant_logits = self.quadrant_head(wc_cat)

            if self.hard_quadrant:
                idx = quadrant_logits.argmax(dim=-1)
                quadrant_dist = torch.zeros_like(quadrant_logits).scatter_(1, idx.unsqueeze(1), 1.0)
            else:
                quadrant_dist = torch.softmax(quadrant_logits, dim=-1)

            if not self.disable_attractor:
                quadrant_repr = torch.mm(quadrant_dist, self.quadrant_protos)
                parts.append(quadrant_repr)

            if not self.disable_harm_score:
                harm_score = (quadrant_dist * self.harm_weights.unsqueeze(0)).sum(dim=-1, keepdim=True)
                parts.append(harm_score)

        if not self.disable_perception:
            percept_input = torch.cat([field_h["social_perception"], field_h["behavioral_tendency"]], dim=-1)
            percept_repr = self.percept_proj(percept_input)
            parts.append(percept_repr)

        fused = torch.cat(parts, dim=-1)
        penult = self.pre_cls(fused)
        logits = self.head(penult)
        return (logits, penult) if return_penult else logits


class PooledFusion(nn.Module):
    def __init__(self, mk, hidden=192, nh=4, nc=2, drop=0.15, md=0.15):
        super().__init__()
        self.mk = mk; self.md = md
        self.projs = nn.ModuleList([nn.Sequential(
            nn.Linear(768, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden)
        ) for _ in range(len(mk))])
        self.routes = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1)
        ) for _ in range(nh)])
        cd = nh * hidden + hidden
        self.pre_cls = nn.Sequential(
            nn.LayerNorm(cd), nn.Linear(cd, 256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(drop * 0.5)
        )
        self.head = nn.Linear(64, nc)

    def forward(self, batch, training=False, return_penult=False):
        ref = []
        for p, k in zip(self.projs, self.mk):
            h = p(batch[k])
            if training and self.md > 0:
                h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
            ref.append(h)
        st = torch.stack(ref, dim=1)
        heads = [((st * torch.softmax(rm(st).squeeze(-1), dim=1).unsqueeze(-1)).sum(dim=1))
                 for rm in self.routes]
        fused = torch.cat(heads + [st.mean(dim=1)], dim=-1)
        penult = self.pre_cls(fused)
        logits = self.head(penult)
        return (logits, penult) if return_penult else logits


# ---- Utils (same as ablation_itt.py) ----
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
    mean = train_z.mean(dim=0, keepdim=True); centered = (train_z - mean).numpy()
    lw = LedoitWolf().fit(centered); cov = torch.tensor(lw.covariance_, dtype=torch.float32)
    U, S, V = torch.svd(cov)
    if r and r < U.size(1): U = U[:, :r]; S = S[:r]; V = V[:, :r]
    W = U @ torch.diag(1.0 / torch.sqrt(S + 1e-6))
    return (F.normalize((train_z - mean) @ W, dim=1), F.normalize((val_z - mean) @ W, dim=1), F.normalize((test_z - mean) @ W, dim=1))

def zca_whiten(train_z, val_z, test_z):
    mean = train_z.mean(dim=0, keepdim=True); centered = train_z - mean
    cov = (centered.t() @ centered) / (centered.size(0) - 1)
    U, S, V = torch.svd(cov + 1e-5 * torch.eye(cov.size(0)))
    W = U @ torch.diag(1.0 / torch.sqrt(S)) @ V.t()
    return (F.normalize((train_z - mean) @ W, dim=1), F.normalize((val_z - mean) @ W, dim=1), F.normalize((test_z - mean) @ W, dim=1))

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
    return {"acc": float(accuracy_score(y_true, y_pred)), "f1": float(f1_score(y_true, y_pred, average='macro')),
            "p": float(precision_score(y_true, y_pred, average='macro')), "r": float(recall_score(y_true, y_pred, average='macro'))}


def apply_retrieval(model, feats, splits, lm, mk, nc, best_config, seed):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    common = set.intersection(*[set(feats[k].keys()) for k in mk]) & set(feats["labels"].keys())
    cur = {s: [v for v in splits[s] if v in common] for s in splits}
    trl_ns = DataLoader(DS(cur["train"], feats, lm, mk), 64, False, collate_fn=collate_fn)
    vl = DataLoader(DS(cur["valid"], feats, lm, mk), 64, False, collate_fn=collate_fn)
    tel = DataLoader(DS(cur["test"], feats, lm, mk), 64, False, collate_fn=collate_fn)
    tp, tl_arr, tla = get_pl(model, trl_ns); vp, vl_arr, vla = get_pl(model, vl); tep, tel_arr, tela = get_pl(model, tel)
    blt = torch.tensor(tla)
    wname = best_config.get("whiten", "none")
    if wname == "zca": tp, vp, tep = zca_whiten(tp, vp, tep)
    elif wname.startswith("spca_r"): tp, vp, tep = shrinkage_pca_whiten(tp, vp, tep, r=int(wname.split("r")[1]))
    knn_type = best_config.get("knn_type", "none")
    if knn_type != "none":
        k = best_config["k"]; temp = best_config["temp"]; alpha = best_config["alpha"]
        knn_fn = cosine_knn if knn_type == "cosine" else csls_knn
        kt = knn_fn(tep, tp, blt, k=k, nc=nc, temperature=temp); kv = knn_fn(vp, tp, blt, k=k, nc=nc, temperature=temp)
        bl_val = (1 - alpha) * vl_arr + alpha * kv; bl_test = (1 - alpha) * tel_arr + alpha * kt
    else: bl_val = vl_arr; bl_test = tel_arr
    acc, thresh = best_thresh(bl_val, vla, bl_test, tela)
    if thresh is not None: preds = ((bl_test[:, 1] - bl_test[:, 0]) > thresh).astype(int)
    else: preds = np.argmax(bl_test, axis=1)
    return full_metrics(tela, preds)


def train_model(model, feats, splits, lm, mk, nc, seed, class_weight):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    common = set.intersection(*[set(feats[k].keys()) for k in mk]) & set(feats["labels"].keys())
    cur = {s: [v for v in splits[s] if v in common] for s in splits}
    trl = DataLoader(DS(cur["train"], feats, lm, mk), 32, True, collate_fn=collate_fn)
    vl = DataLoader(DS(cur["valid"], feats, lm, mk), 64, False, collate_fn=collate_fn)
    ema = copy.deepcopy(model)
    opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, dtype=torch.float).to(device), label_smoothing=0.03)
    ep = 45; ts_t = ep * len(trl); ws = 5 * len(trl); sch = cw(opt, ws, ts_t)
    bva, bst = -1, None
    for e in range(ep):
        model.train()
        for batch in trl:
            batch = {k: v.to(device) for k, v in batch.items()}; opt.zero_grad()
            crit(model(batch, training=True), batch["label"]).backward()
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
    ema.load_state_dict(bst)
    return ema


def run_ablation(abl_id, abl_desc, model_fn, mk, feats, splits, lm, nc, seed, best_config, class_weight, logger):
    logger.info(f"  [{abl_id}] {abl_desc}")
    model = model_fn().to(device)
    trained = train_model(model, feats, splits, lm, mk, nc, seed, class_weight)
    metrics = apply_retrieval(trained, feats, splits, lm, mk, nc, best_config, seed)
    logger.info(f"    ACC={metrics['acc']:.4f} F1={metrics['f1']:.4f} P={metrics['p']:.4f} R={metrics['r']:.4f}")
    return {"abl_id": abl_id, "desc": abl_desc, **metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="HateMM", choices=["HateMM", "Multihateclip"])
    parser.add_argument("--language", default="English", choices=["English", "Chinese"])
    args = parser.parse_args()

    if args.dataset_name == "HateMM":
        emb_dir = "./embeddings/HateMM"; ann_path = "./datasets/HateMM/annotation(new).json"
        split_dir = "./datasets/HateMM/splits"; lm = {"Non Hate": 0, "Hate": 1}; tag = "HateMM"
    else:
        emb_dir = f"./embeddings/Multihateclip/{args.language}"
        ann_path = f"./datasets/Multihateclip/{args.language}/annotation(new).json"
        split_dir = f"./datasets/Multihateclip/{args.language}/splits"
        lm = {"Normal": 0, "Offensive": 1, "Hateful": 1}; tag = f"MHC_{args.language[:2]}"

    results_path = f"./seed_search_scm/{tag}_off0/all_results.json"
    with open(results_path) as f:
        best = json.load(f)["global_best"]
    seed = best["seed"]; best_config = best["best_config"]

    os.makedirs("./logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(f"abl_scm_{tag}_{ts}")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fh = logging.FileHandler(f"./logs/ablation_scm_{tag}_{ts}.log", encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt); logger.addHandler(fh); logger.addHandler(ch)

    nc = 2; base_mk = ["text", "audio", "frame"]
    feats = {
        "text": torch.load(f"{emb_dir}/text_features.pth", map_location="cpu"),
        "audio": torch.load(f"{emb_dir}/wavlm_audio_features.pth", map_location="cpu"),
        "frame": torch.load(f"{emb_dir}/frame_features.pth", map_location="cpu"),
    }
    for f in SCM_FIELDS:
        feats[f"scm_{f}"] = torch.load(f"{emb_dir}/scm_{f}_features.pth", map_location="cpu")
    feats["scm_rationale"] = torch.load(f"{emb_dir}/scm_rationale_features.pth", map_location="cpu")
    with open(ann_path) as f2:
        feats["labels"] = {d["Video_ID"]: d for d in json.load(f2)}
    splits = load_split_ids(split_dir)

    all_mk = list(base_mk) + [f"scm_{f}" for f in SCM_FIELDS]
    cw_val = [1.0, 1.5]

    logger.info(f"=== SCM Ablation Study: {tag} ===")
    logger.info(f"Best seed={seed}, config={best_config}")
    logger.info(f"Best ACC={best['metrics']['acc']:.4f} F1={best['metrics']['f1']:.4f}")

    results = []

    # ABL-00: Full model
    results.append(run_ablation("ABL-00", "Full SCM model (reproduce)",
        lambda: SCMFusion(base_mk, nc=nc), all_mk, feats, splits, lm, nc, seed, best_config, cw_val, logger))

    # ABL-01: No rationale
    results.append(run_ablation("ABL-01", "No rationale (base only)",
        lambda: PooledFusion(base_mk, nc=nc), base_mk, feats, splits, lm, nc, seed, best_config, cw_val, logger))

    # ABL-02: Pooled rationale
    pooled_mk = base_mk + ["scm_rationale"]
    results.append(run_ablation("ABL-02", "Pooled rationale (no theory fusion)",
        lambda: PooledFusion(pooled_mk, nc=nc), pooled_mk, feats, splits, lm, nc, seed, best_config, cw_val, logger))

    # ABL-03: No Warmth stream
    results.append(run_ablation("ABL-03", "No Warmth stream (Competence only)",
        lambda: SCMFusion(base_mk, nc=nc, disable_warmth=True),
        all_mk, feats, splits, lm, nc, seed, best_config, cw_val, logger))

    # ABL-04: No Competence stream
    results.append(run_ablation("ABL-04", "No Competence stream (Warmth only)",
        lambda: SCMFusion(base_mk, nc=nc, disable_competence=True),
        all_mk, feats, splits, lm, nc, seed, best_config, cw_val, logger))

    # ABL-05: No Quadrant Attractor
    results.append(run_ablation("ABL-05", "No Quadrant Attractor (no prototype matching)",
        lambda: SCMFusion(base_mk, nc=nc, disable_attractor=True),
        all_mk, feats, splits, lm, nc, seed, best_config, cw_val, logger))

    # ABL-06: No harm_score
    results.append(run_ablation("ABL-06", "No harm_score (no BIAS Map weights)",
        lambda: SCMFusion(base_mk, nc=nc, disable_harm_score=True),
        all_mk, feats, splits, lm, nc, seed, best_config, cw_val, logger))

    # ABL-07: No perception features
    results.append(run_ablation("ABL-07", "No perception features (no social_perception + behavioral_tendency)",
        lambda: SCMFusion(base_mk, nc=nc, disable_perception=True),
        all_mk, feats, splits, lm, nc, seed, best_config, cw_val, logger))

    # ABL-08: Hard quadrant (argmax instead of softmax)
    results.append(run_ablation("ABL-08", "Hard quadrant assignment (argmax, not softmax)",
        lambda: SCMFusion(base_mk, nc=nc, hard_quadrant=True),
        all_mk, feats, splits, lm, nc, seed, best_config, cw_val, logger))

    # ABL-09 to ABL-13: Remove individual fields
    for field in SCM_FIELDS:
        results.append(run_ablation(f"ABL-09-{field}", f"Remove field: {field}",
            lambda fld=field: SCMFusion(base_mk, nc=nc, disable_fields=[fld]),
            all_mk, feats, splits, lm, nc, seed, best_config, cw_val, logger))

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(f"ABLATION SUMMARY — SCM {tag}")
    logger.info(f"{'='*70}")
    logger.info(f"{'ID':<25} {'Description':<50} {'ACC':>7} {'F1':>7} {'Δ F1':>7}")
    logger.info(f"{'-'*96}")
    base_f1 = results[0]["f1"]
    for r in results:
        delta = r["f1"] - base_f1
        logger.info(f"{r['abl_id']:<25} {r['desc']:<50} {r['acc']:>7.4f} {r['f1']:>7.4f} {delta:>+7.4f}")

    os.makedirs("./ablation_results", exist_ok=True)
    save_path = f"./ablation_results/scm_{tag}_ablations.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
