"""
Ablation study for SCM model — V3.
ABL-00 uses EXACT original model class from main_scm.py.
Other ablations subclass and override forward to disable components.

Usage:
  conda run -n HVGuard python ablation_scm_v3.py --dataset_name HateMM
  conda run -n HVGuard python ablation_scm_v3.py --dataset_name Multihateclip --language English
  conda run -n HVGuard python ablation_scm_v3.py --dataset_name Multihateclip --language Chinese
"""

import argparse, csv, json, os, random, copy, logging
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader, Dataset
import warnings; warnings.filterwarnings("ignore")

from main_scm import (SCMFusion, DS, collate_fn, cw, load_split_ids, get_pl,
                      shrinkage_pca_whiten, zca_whiten, cosine_knn, csls_knn,
                      best_thresh, full_metrics, SCM_FIELDS)

device = "cuda"


class FlatFusion(nn.Module):
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
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(drop * 0.5))
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


# ---- Ablation subclasses of original SCMFusion ----
# All share the same __init__ (same param names/order), only override forward.

def _base_encode(self, batch, training):
    """Shared base encoding logic."""
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
    base_ctx = base_stack.mean(dim=1)
    field_h = {}
    for f in SCM_FIELDS:
        h = self.field_projs[f](batch[f"scm_{f}"])
        if training and self.md > 0:
            h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
        field_h[f] = h
    return base_repr, base_ctx, field_h


class SCMNoWarmth(SCMFusion):
    """Zero out warmth stream."""
    def forward(self, batch, training=False, return_penult=False):
        base_repr, base_ctx, field_h = _base_encode(self, batch, training)
        warmth_repr = torch.zeros(base_ctx.size(0), self.hidden, device=base_ctx.device)
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
        penult = self.pre_cls(fused)
        logits = self.head(penult)
        return (logits, penult) if return_penult else logits


class SCMNoCompetence(SCMFusion):
    """Zero out competence stream."""
    def forward(self, batch, training=False, return_penult=False):
        base_repr, base_ctx, field_h = _base_encode(self, batch, training)
        warmth_input = torch.cat([field_h["warmth_evidence"], field_h["target_group"], base_ctx], dim=-1)
        warmth_repr = self.warmth_stream(warmth_input)
        comp_repr = torch.zeros(base_ctx.size(0), self.hidden, device=base_ctx.device)
        wc_cat = torch.cat([warmth_repr, comp_repr], dim=-1)
        quadrant_logits = self.quadrant_head(wc_cat)
        quadrant_dist = torch.softmax(quadrant_logits, dim=-1)
        quadrant_repr = torch.mm(quadrant_dist, self.quadrant_protos)
        harm_score = (quadrant_dist * self.harm_weights.unsqueeze(0)).sum(dim=-1, keepdim=True)
        percept_input = torch.cat([field_h["social_perception"], field_h["behavioral_tendency"]], dim=-1)
        percept_repr = self.percept_proj(percept_input)
        fused = torch.cat([base_repr, quadrant_repr, percept_repr, harm_score], dim=-1)
        penult = self.pre_cls(fused)
        logits = self.head(penult)
        return (logits, penult) if return_penult else logits


class SCMNoAttractor(SCMFusion):
    """Replace quadrant_repr with zeros."""
    def forward(self, batch, training=False, return_penult=False):
        base_repr, base_ctx, field_h = _base_encode(self, batch, training)
        warmth_input = torch.cat([field_h["warmth_evidence"], field_h["target_group"], base_ctx], dim=-1)
        warmth_repr = self.warmth_stream(warmth_input)
        comp_input = torch.cat([field_h["competence_evidence"], field_h["target_group"], base_ctx], dim=-1)
        comp_repr = self.competence_stream(comp_input)
        wc_cat = torch.cat([warmth_repr, comp_repr], dim=-1)
        quadrant_logits = self.quadrant_head(wc_cat)
        quadrant_dist = torch.softmax(quadrant_logits, dim=-1)
        quadrant_repr = torch.zeros(base_ctx.size(0), self.hidden, device=base_ctx.device)  # zeroed
        harm_score = (quadrant_dist * self.harm_weights.unsqueeze(0)).sum(dim=-1, keepdim=True)
        percept_input = torch.cat([field_h["social_perception"], field_h["behavioral_tendency"]], dim=-1)
        percept_repr = self.percept_proj(percept_input)
        fused = torch.cat([base_repr, quadrant_repr, percept_repr, harm_score], dim=-1)
        penult = self.pre_cls(fused)
        logits = self.head(penult)
        return (logits, penult) if return_penult else logits


class SCMNoHarmScore(SCMFusion):
    """Replace harm_score with zero."""
    def forward(self, batch, training=False, return_penult=False):
        base_repr, base_ctx, field_h = _base_encode(self, batch, training)
        warmth_input = torch.cat([field_h["warmth_evidence"], field_h["target_group"], base_ctx], dim=-1)
        warmth_repr = self.warmth_stream(warmth_input)
        comp_input = torch.cat([field_h["competence_evidence"], field_h["target_group"], base_ctx], dim=-1)
        comp_repr = self.competence_stream(comp_input)
        wc_cat = torch.cat([warmth_repr, comp_repr], dim=-1)
        quadrant_logits = self.quadrant_head(wc_cat)
        quadrant_dist = torch.softmax(quadrant_logits, dim=-1)
        quadrant_repr = torch.mm(quadrant_dist, self.quadrant_protos)
        harm_score = torch.zeros(base_ctx.size(0), 1, device=base_ctx.device)  # zeroed
        percept_input = torch.cat([field_h["social_perception"], field_h["behavioral_tendency"]], dim=-1)
        percept_repr = self.percept_proj(percept_input)
        fused = torch.cat([base_repr, quadrant_repr, percept_repr, harm_score], dim=-1)
        penult = self.pre_cls(fused)
        logits = self.head(penult)
        return (logits, penult) if return_penult else logits


class SCMNoMiddle(SCMFusion):
    """Zero both attractor and harm_score."""
    def forward(self, batch, training=False, return_penult=False):
        base_repr, base_ctx, field_h = _base_encode(self, batch, training)
        warmth_input = torch.cat([field_h["warmth_evidence"], field_h["target_group"], base_ctx], dim=-1)
        warmth_repr = self.warmth_stream(warmth_input)
        comp_input = torch.cat([field_h["competence_evidence"], field_h["target_group"], base_ctx], dim=-1)
        comp_repr = self.competence_stream(comp_input)
        quadrant_repr = torch.zeros(base_ctx.size(0), self.hidden, device=base_ctx.device)
        harm_score = torch.zeros(base_ctx.size(0), 1, device=base_ctx.device)
        percept_input = torch.cat([field_h["social_perception"], field_h["behavioral_tendency"]], dim=-1)
        percept_repr = self.percept_proj(percept_input)
        fused = torch.cat([base_repr, quadrant_repr, percept_repr, harm_score], dim=-1)
        penult = self.pre_cls(fused)
        logits = self.head(penult)
        return (logits, penult) if return_penult else logits


class SCMNoPerception(SCMFusion):
    """Zero perception features."""
    def forward(self, batch, training=False, return_penult=False):
        base_repr, base_ctx, field_h = _base_encode(self, batch, training)
        warmth_input = torch.cat([field_h["warmth_evidence"], field_h["target_group"], base_ctx], dim=-1)
        warmth_repr = self.warmth_stream(warmth_input)
        comp_input = torch.cat([field_h["competence_evidence"], field_h["target_group"], base_ctx], dim=-1)
        comp_repr = self.competence_stream(comp_input)
        wc_cat = torch.cat([warmth_repr, comp_repr], dim=-1)
        quadrant_logits = self.quadrant_head(wc_cat)
        quadrant_dist = torch.softmax(quadrant_logits, dim=-1)
        quadrant_repr = torch.mm(quadrant_dist, self.quadrant_protos)
        harm_score = (quadrant_dist * self.harm_weights.unsqueeze(0)).sum(dim=-1, keepdim=True)
        percept_repr = torch.zeros(base_ctx.size(0), self.hidden, device=base_ctx.device)  # zeroed
        fused = torch.cat([base_repr, quadrant_repr, percept_repr, harm_score], dim=-1)
        penult = self.pre_cls(fused)
        logits = self.head(penult)
        return (logits, penult) if return_penult else logits


class SCMHardQuadrant(SCMFusion):
    """Use argmax instead of softmax for quadrant."""
    def forward(self, batch, training=False, return_penult=False):
        base_repr, base_ctx, field_h = _base_encode(self, batch, training)
        warmth_input = torch.cat([field_h["warmth_evidence"], field_h["target_group"], base_ctx], dim=-1)
        warmth_repr = self.warmth_stream(warmth_input)
        comp_input = torch.cat([field_h["competence_evidence"], field_h["target_group"], base_ctx], dim=-1)
        comp_repr = self.competence_stream(comp_input)
        wc_cat = torch.cat([warmth_repr, comp_repr], dim=-1)
        quadrant_logits = self.quadrant_head(wc_cat)
        idx = quadrant_logits.argmax(dim=-1)
        quadrant_dist = torch.zeros_like(quadrant_logits).scatter_(1, idx.unsqueeze(1), 1.0)
        quadrant_repr = torch.mm(quadrant_dist, self.quadrant_protos)
        harm_score = (quadrant_dist * self.harm_weights.unsqueeze(0)).sum(dim=-1, keepdim=True)
        percept_input = torch.cat([field_h["social_perception"], field_h["behavioral_tendency"]], dim=-1)
        percept_repr = self.percept_proj(percept_input)
        fused = torch.cat([base_repr, quadrant_repr, percept_repr, harm_score], dim=-1)
        penult = self.pre_cls(fused)
        logits = self.head(penult)
        return (logits, penult) if return_penult else logits


class SCMNoTargetCond(SCMFusion):
    """Replace target_group with zeros in stream inputs."""
    def forward(self, batch, training=False, return_penult=False):
        base_repr, base_ctx, field_h = _base_encode(self, batch, training)
        zero_target = torch.zeros_like(field_h["target_group"])
        warmth_input = torch.cat([field_h["warmth_evidence"], zero_target, base_ctx], dim=-1)
        warmth_repr = self.warmth_stream(warmth_input)
        comp_input = torch.cat([field_h["competence_evidence"], zero_target, base_ctx], dim=-1)
        comp_repr = self.competence_stream(comp_input)
        wc_cat = torch.cat([warmth_repr, comp_repr], dim=-1)
        quadrant_logits = self.quadrant_head(wc_cat)
        quadrant_dist = torch.softmax(quadrant_logits, dim=-1)
        quadrant_repr = torch.mm(quadrant_dist, self.quadrant_protos)
        harm_score = (quadrant_dist * self.harm_weights.unsqueeze(0)).sum(dim=-1, keepdim=True)
        percept_input = torch.cat([field_h["social_perception"], field_h["behavioral_tendency"]], dim=-1)
        percept_repr = self.percept_proj(percept_input)
        fused = torch.cat([base_repr, quadrant_repr, percept_repr, harm_score], dim=-1)
        penult = self.pre_cls(fused)
        logits = self.head(penult)
        return (logits, penult) if return_penult else logits


def train_and_eval(model_fn, feats, splits, lm, mk, nc, seed, class_weight, best_config):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    common = set.intersection(*[set(feats[k].keys()) for k in mk]) & set(feats["labels"].keys())
    cur = {s: [v for v in splits[s] if v in common] for s in splits}
    trl = DataLoader(DS(cur["train"], feats, lm, mk), 32, True, collate_fn=collate_fn)
    vl = DataLoader(DS(cur["valid"], feats, lm, mk), 64, False, collate_fn=collate_fn)
    tel = DataLoader(DS(cur["test"], feats, lm, mk), 64, False, collate_fn=collate_fn)
    trl_ns = DataLoader(DS(cur["train"], feats, lm, mk), 64, False, collate_fn=collate_fn)
    model = model_fn().to(device); ema = copy.deepcopy(model)
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
    tp, tl_arr, tla = get_pl(ema, trl_ns); vp, vl_arr, vla = get_pl(ema, vl)
    tep, tel_arr, tela = get_pl(ema, tel); blt = torch.tensor(tla)
    wname = best_config.get("whiten", "none")
    if wname == "zca": tp, vp, tep = zca_whiten(tp, vp, tep)
    elif wname.startswith("spca_r"): tp, vp, tep = shrinkage_pca_whiten(tp, vp, tep, r=int(wname.split("r")[1]))
    knn_type = best_config.get("knn_type", "none")
    if knn_type != "none":
        k = best_config["k"]; temp = best_config["temp"]; alpha = best_config["alpha"]
        knn_fn = cosine_knn if knn_type == "cosine" else csls_knn
        kt = knn_fn(tep, tp, blt, k=k, nc=nc, temperature=temp)
        kv = knn_fn(vp, tp, blt, k=k, nc=nc, temperature=temp)
        bl_val = (1 - alpha) * vl_arr + alpha * kv; bl_test = (1 - alpha) * tel_arr + alpha * kt
    else: bl_val = vl_arr; bl_test = tel_arr
    acc, thresh = best_thresh(bl_val, vla, bl_test, tela)
    if thresh is not None: preds = ((bl_test[:, 1] - bl_test[:, 0]) > thresh).astype(int)
    else: preds = np.argmax(bl_test, axis=1)
    return full_metrics(tela, preds)


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

    with open(f"./seed_search_scm/{tag}_off0/all_results.json") as f:
        best = json.load(f)["global_best"]
    seed = best["seed"]; best_config = best["best_config"]

    os.makedirs("./logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(f"ablv3_scm_{tag}_{ts}")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fh = logging.FileHandler(f"./logs/ablation_scm_v3_{tag}_{ts}.log", encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt); logger.addHandler(fh); logger.addHandler(ch)

    nc = 2; base_mk = ["text", "audio", "frame"]; cw_val = [1.0, 1.5]
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
    all_scm_mk = [f"scm_{f}" for f in SCM_FIELDS]

    logger.info(f"=== SCM Ablation V3: {tag} ===")
    logger.info(f"Best seed={seed}, config={best_config}")
    logger.info(f"Reported: ACC={best['metrics']['acc']:.4f} F1={best['metrics']['f1']:.4f}")

    results = []
    def run_abl(abl_id, desc, model_fn, mk):
        logger.info(f"  [{abl_id}] {desc}")
        m = train_and_eval(model_fn, feats, splits, lm, mk, nc, seed, cw_val, best_config)
        logger.info(f"    ACC={m['acc']:.4f} F1={m['f1']:.4f} P={m['p']:.4f} R={m['r']:.4f}")
        results.append({"abl_id": abl_id, "desc": desc, **m})

    # ABL-00: Full model — EXACT original class
    run_abl("ABL-00", "Full SCM model (original class)",
        lambda: SCMFusion(base_mk, nc=nc), base_mk + all_scm_mk)

    # ABL-01: No rationale
    run_abl("ABL-01", "No rationale (base only)",
        lambda: FlatFusion(base_mk, nc=nc), base_mk)

    # ABL-02: Pooled rationale
    run_abl("ABL-02", "Pooled rationale (no theory fusion)",
        lambda: FlatFusion(base_mk + ["scm_rationale"], nc=nc), base_mk + ["scm_rationale"])

    # ABL-03: Per-field flat
    run_abl("ABL-03", "Per-field flat (no theory structure)",
        lambda: FlatFusion(base_mk + all_scm_mk, nc=nc), base_mk + all_scm_mk)

    # ABL-04: No Warmth stream
    run_abl("ABL-04", "No Warmth stream (zeroed)",
        lambda: SCMNoWarmth(base_mk, nc=nc), base_mk + all_scm_mk)

    # ABL-05: No Competence stream
    run_abl("ABL-05", "No Competence stream (zeroed)",
        lambda: SCMNoCompetence(base_mk, nc=nc), base_mk + all_scm_mk)

    # ABL-06: No Quadrant Attractor
    run_abl("ABL-06", "No Quadrant Attractor (zeroed)",
        lambda: SCMNoAttractor(base_mk, nc=nc), base_mk + all_scm_mk)

    # ABL-07: No harm_score
    run_abl("ABL-07", "No harm_score (zeroed)",
        lambda: SCMNoHarmScore(base_mk, nc=nc), base_mk + all_scm_mk)

    # ABL-08: No middle module (both zeroed)
    run_abl("ABL-08", "No middle module (attractor+harm zeroed)",
        lambda: SCMNoMiddle(base_mk, nc=nc), base_mk + all_scm_mk)

    # ABL-09: No perception features
    run_abl("ABL-09", "No perception features (zeroed)",
        lambda: SCMNoPerception(base_mk, nc=nc), base_mk + all_scm_mk)

    # ABL-10: Hard quadrant
    run_abl("ABL-10", "Hard quadrant (argmax)",
        lambda: SCMHardQuadrant(base_mk, nc=nc), base_mk + all_scm_mk)

    # ABL-11: No target_group conditioning
    run_abl("ABL-11", "No target_group conditioning (zeroed)",
        lambda: SCMNoTargetCond(base_mk, nc=nc), base_mk + all_scm_mk)

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"ABLATION SUMMARY V3 — SCM {tag}")
    logger.info(f"{'='*80}")
    logger.info(f"{'ID':<15} {'Description':<45} {'ACC':>7} {'F1':>7} {'Δ F1':>7}")
    logger.info(f"{'-'*81}")
    base_f1 = results[0]["f1"]
    for r in results:
        delta = r["f1"] - base_f1
        logger.info(f"{r['abl_id']:<15} {r['desc']:<45} {r['acc']:>7.4f} {r['f1']:>7.4f} {delta:>+7.4f}")

    os.makedirs("./ablation_results", exist_ok=True)
    save_path = f"./ablation_results/scm_v3_{tag}_ablations.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
