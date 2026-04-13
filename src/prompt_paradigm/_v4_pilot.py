"""v4 pre-pilot: rank-based noisy-OR fusion on existing v3 scoring artefacts.

Sanity-check candidate aggregators for v4 Gate 1 proposal, using
train_polarity.jsonl as the rank reference (label-free) and test_polarity.jsonl
for oracle/TF/TR evaluation. No new MLLM calls.
"""
import json, os, sys, bisect
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_utils import load_annotations, DATASET_ROOTS
from quick_eval_all import otsu_threshold, gmm_threshold


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def acc_sweep(scores, labels):
    best = (0.0, None)
    for t in sorted(set(scores)):
        preds = [1 if s >= t else 0 for s in scores]
        acc = sum(1 for p, y in zip(preds, labels) if p == y) / len(labels)
        if acc > best[0]:
            best = (acc, t)
    return best


def macro_f1(preds, labels):
    def f1(tp, fp, fn):
        if tp == 0:
            return 0.0
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        return 2 * p * r / (p + r) if p + r > 0 else 0.0

    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    f1_pos = f1(tp, fp, fn)
    f1_neg = f1(tn, fn, fp)
    return (f1_pos + f1_neg) / 2


def apply_thresh(scores, labels, t):
    preds = [1 if s >= t else 0 for s in scores]
    acc = sum(1 for p, y in zip(preds, labels) if p == y) / len(labels)
    return acc, macro_f1(preds, labels)


def rank_of(v, sorted_ref):
    return bisect.bisect_right(sorted_ref, v) / len(sorted_ref)


def main():
    for ds, base_o, base_acc, base_mf in [
        ("MHClip_EN", 0.7764, 0.7640, 0.6532),
        ("MHClip_ZH", 0.8121, 0.8121, 0.7871),
    ]:
        print(f"\n=== {ds}  baseline ACC={base_acc} mF1={base_mf} oracle={base_o} ===")
        ann = load_annotations(ds)
        train = load(f"results/prompt_paradigm/{ds}/train_polarity.jsonl")
        test = load(f"results/prompt_paradigm/{ds}/test_polarity.jsonl")

        tr_pe = sorted([r["p_evidence"] for r in train if r.get("p_evidence") is not None])
        tr_pc = sorted([r["p_compliance"] for r in train if r.get("p_compliance") is not None])

        def score_nor(pe, pc):
            r_e = rank_of(pe, tr_pe)
            r_c = 1.0 - rank_of(pc, tr_pc)
            return 1 - (1 - r_e) * (1 - r_c)

        def score_avg(pe, pc):
            r_e = rank_of(pe, tr_pe)
            r_c = 1.0 - rank_of(pc, tr_pc)
            return (r_e + r_c) / 2

        def score_max(pe, pc):
            r_e = rank_of(pe, tr_pe)
            r_c = 1.0 - rank_of(pc, tr_pc)
            return max(r_e, r_c)

        # multiplicative OR on raw probs (independence assumption)
        def score_raw_nor(pe, pc):
            return 1 - (1 - pe) * pc  # pc high = clean, so (1-pe)*pc = benign mass

        # Load test with annotation alignment
        split = os.path.join(DATASET_ROOTS[ds], "splits", "test_clean.csv")
        ids = [l.strip() for l in open(split) if l.strip()]
        by = {r["video_id"]: r for r in test}

        test_rows = []
        for vid in ids:
            a = ann.get(vid); r = by.get(vid)
            if a is None or r is None:
                continue
            if r.get("p_evidence") is None or r.get("p_compliance") is None:
                continue
            label = a.get("hate_label", a.get("label"))
            y = 1 if label in ("Hateful", "Offensive", 1) else 0
            test_rows.append((r["p_evidence"], r["p_compliance"], y))

        labels = [y for _, _, y in test_rows]
        n_pos = sum(labels)
        n = len(labels)
        print(f"  n_test={n}  n_pos={n_pos}")

        for name, sf in [
            ("noisy_OR_rank", score_nor),
            ("avg_rank", score_avg),
            ("max_rank", score_max),
            ("raw_nor (no rank)", score_raw_nor),
        ]:
            test_scores = [sf(pe, pc) for pe, pc, _ in test_rows]
            train_scores = [sf(r["p_evidence"], r["p_compliance"])
                            for r in train if r.get("p_evidence") is not None
                            and r.get("p_compliance") is not None]

            oracle = acc_sweep(test_scores, labels)
            test_np = np.array(test_scores)
            train_np = np.array(train_scores)

            tf_otsu = otsu_threshold(test_np)
            tf_gmm = gmm_threshold(test_np)
            tr_otsu = otsu_threshold(train_np)
            tr_gmm = gmm_threshold(train_np)

            # AUC (pairwise)
            pos_s = [s for s, y in zip(test_scores, labels) if y == 1]
            neg_s = [s for s, y in zip(test_scores, labels) if y == 0]
            wins = sum(1 for ps in pos_s for ns in neg_s if ps > ns)
            ties = sum(1 for ps in pos_s for ns in neg_s if ps == ns)
            auc = (wins + 0.5 * ties) / (len(pos_s) * len(neg_s))

            print(f"  {name:20s} oracle={oracle[0]:.4f}  AUC={auc:.4f}")
            for tname, tv in [("tf_otsu", tf_otsu), ("tf_gmm", tf_gmm),
                              ("tr_otsu", tr_otsu), ("tr_gmm", tr_gmm)]:
                acc, mf = apply_thresh(test_scores, labels, tv)
                tag = ""
                if acc > base_acc and mf >= base_mf:
                    tag = "  STRICT-BEAT"
                elif acc > base_acc:
                    tag = "  acc-up mF1-regress"
                print(f"    {tname}  t={tv:.4f}  acc={acc:.4f}  mF1={mf:.4f}{tag}")


if __name__ == "__main__":
    main()
