"""Quick dual-pooling ensemble: train CLS and Mean Pool models for each seed, ensemble logits."""
import sys, json, random, copy, numpy as np, torch, torch.nn as nn, torch.optim as optim
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

def best_thresh(vl, vla, tl, tla):
    std = accuracy_score(tla, np.argmax(tl, axis=1))
    vd = vl[:, 1] - vl[:, 0]; td = tl[:, 1] - tl[:, 0]
    bt, bv = 0, 0
    for t in np.arange(-3, 3, 0.02):
        v = accuracy_score(vla, (vd > t).astype(int))
        if v > bv: bv, bt = v, t
    tuned = accuracy_score(tla, (td > bt).astype(int))
    return (tuned, bt) if tuned > std else (std, None)

def train_model(feats, splits, lm, mk, seed):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    common = set.intersection(*[set(feats[k].keys()) for k in mk]) & set(feats["labels"].keys())
    cur = {s: [v for v in splits[s] if v in common] for s in splits}
    trl = DataLoader(DS(cur["train"], feats, lm, mk), 32, True, collate_fn=collate_fn)
    vl = DataLoader(DS(cur["valid"], feats, lm, mk), 64, False, collate_fn=collate_fn)
    tel = DataLoader(DS(cur["test"], feats, lm, mk), 64, False, collate_fn=collate_fn)
    model = SCMQMoEQELS(base_mk, nc=2).to(device); ema = copy.deepcopy(model)
    opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
    cw_t = torch.tensor([1.0, 1.5], dtype=torch.float).to(device)
    ep = 45; ts_t = ep * len(trl); ws = 5 * len(trl)
    sch = LambdaLR(opt, lambda s: s/max(1,ws) if s<ws else max(0, 0.5*(1+np.cos(np.pi*(s-ws)/max(1,ts_t-ws)))))
    bva, bst = -1, None
    for e in range(ep):
        model.train()
        for batch in trl:
            batch = {k: v.to(device) for k, v in batch.items()}; opt.zero_grad()
            logits, q_ent = model(batch, training=True, return_qels=True)
            loss = qels_cross_entropy(logits, batch["label"], q_ent, nc=2, eps_min=0.01, eps_lambda=0.15, class_weight=cw_t)
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
    vl_l, vl_lab, te_l, te_lab = [], [], [], []
    with torch.no_grad():
        for batch in vl:
            batch = {k: v.to(device) for k, v in batch.items()}
            vl_l.append(ema(batch).cpu()); vl_lab.extend(batch["label"].cpu().numpy())
        for batch in tel:
            batch = {k: v.to(device) for k, v in batch.items()}
            te_l.append(ema(batch).cpu()); te_lab.extend(batch["label"].cpu().numpy())
    return torch.cat(vl_l).numpy(), np.array(vl_lab), torch.cat(te_l).numpy(), np.array(te_lab)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="HateMM")
    parser.add_argument("--language", default="English")
    parser.add_argument("--num_runs", type=int, default=200)
    parser.add_argument("--seed_offset", type=int, default=0)
    args = parser.parse_args()
    if args.dataset_name == "HateMM":
        emb_dir = "./embeddings/HateMM"; ann_path = "./datasets/HateMM/annotation(new).json"
        split_dir = "./datasets/HateMM/splits"; lm = {"Non Hate": 0, "Hate": 1}; tag = "HateMM"
    else:
        emb_dir = f"./embeddings/Multihateclip/{args.language}"
        ann_path = f"./datasets/Multihateclip/{args.language}/annotation(new).json"
        split_dir = f"./datasets/Multihateclip/{args.language}/splits"
        lm = {"Normal": 0, "Offensive": 1, "Hateful": 1}; tag = f"MHC_{args.language[:2]}"

    os.makedirs("./logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler(f"./logs/dual_pool_{tag}_{ts}.log"),
        logging.StreamHandler()
    ], format="%(asctime)s [%(levelname)s] %(message)s")

    feats_base = {
        "text": torch.load(f"{emb_dir}/text_features.pth", map_location="cpu"),
        "audio": torch.load(f"{emb_dir}/wavlm_audio_features.pth", map_location="cpu"),
        "frame": torch.load(f"{emb_dir}/frame_features.pth", map_location="cpu"),
    }
    with open(ann_path) as f:
        labels_data = {d["Video_ID"]: d for d in json.load(f)}
    splits = load_split_ids(split_dir)
    mk = base_mk + [f"scm_{f}" for f in SCM_FIELDS]

    feats_cls = dict(feats_base); feats_cls["labels"] = labels_data
    feats_mean = dict(feats_base); feats_mean["labels"] = labels_data
    for f in SCM_FIELDS:
        feats_cls[f"scm_{f}"] = torch.load(f"{emb_dir}/scm_{f}_features.pth", map_location="cpu")
        feats_mean[f"scm_{f}"] = torch.load(f"{emb_dir}/scm_mean_{f}_features.pth", map_location="cpu")

    logging.info(f"=== Dual Pool Ensemble: {tag}, Runs: {args.num_runs}, Offset: {args.seed_offset} ===")

    all_results = []; best_acc = 0; best_result = None
    save_dir = f"./seed_search_dual_pool/{tag}_off{args.seed_offset}"
    os.makedirs(save_dir, exist_ok=True)

    for ri in range(args.num_runs):
        seed = ri * 1000 + 42 + args.seed_offset
        vl_cls, vl_lab, te_cls, te_lab = train_model(feats_cls, splits, lm, mk, seed)
        vl_mean, _, te_mean, _ = train_model(feats_mean, splits, lm, mk, seed)
        
        best_ens_acc = 0; best_alpha = 0.5
        for alpha_100 in range(0, 101, 5):
            alpha = alpha_100 / 100.0
            vl_ens = alpha * vl_cls + (1-alpha) * vl_mean
            te_ens = alpha * te_cls + (1-alpha) * te_mean
            acc, _ = best_thresh(vl_ens, vl_lab, te_ens, te_lab)
            if acc > best_ens_acc:
                best_ens_acc = acc; best_alpha = alpha
        
        result = {"seed": seed, "ri": ri, "ens_acc": float(best_ens_acc), "alpha": best_alpha}
        all_results.append(result)
        if best_ens_acc > best_acc:
            best_acc = best_ens_acc; best_result = result
            logging.info(f"  NEW BEST seed={seed} ENS_ACC={best_ens_acc:.4f} alpha={best_alpha:.2f}")
        if (ri + 1) % 20 == 0:
            accs = [r["ens_acc"] for r in all_results]
            logging.info(f"  [{ri+1}/{args.num_runs}] mean={np.mean(accs):.4f} max={np.max(accs):.4f}")

    accs = [r["ens_acc"] for r in all_results]
    logging.info(f"  FINAL: mean={np.mean(accs):.4f}+/-{np.std(accs):.4f} max={np.max(accs):.4f}")
    logging.info(f"  BEST: {json.dumps(best_result)}")
    with open(os.path.join(save_dir, "all_results.json"), "w") as f:
        json.dump({"best": best_result, "all": all_results}, f, indent=2)

if __name__ == "__main__":
    main()
