"""ALARM video reproduction — 5-stage pipeline driver.

Upstream: `external_repos/alarm/src/main.py`. That driver is a Hydra
dispatcher that instantiates one of `{Label_Runner, Experience_Runner,
Reference_Runner, InPredict_Runner}` based on `cfg.task`. Each runner
owns its own data loader and sequence of steps. We flatten all of
that into a single Python script that runs the 5 stages in order
per dataset, reading only our project's `data_utils` + per-stage
intermediate files.

Stage order (same as upstream `run/run_FHM.sh`):

  Stage 1 — Label          → results/alarm/<ds>/label.jsonl
  Stage 2 — make_embeddings → results/alarm/<ds>/fea/{image,text,joint}_embed.pt
  Stage 3 — conduct_retrieval → results/alarm/<ds>/retrieve/pairs.jsonl
  Stage 4 — Experience     → results/alarm/<ds>/experience.jsonl
  Stage 5 — Reference      → results/alarm/<ds>/reference.json
  Stage 6 — InPredict      → results/alarm/<ds>/test_alarm.jsonl  [final output]

Upstream runs Label on the **train split** to build the high-
confidence pair pool for Experience/Reference, then runs InPredict
on the **test split** for the final prediction. We follow that
split: stages 1-5 run on `train_clean.csv`, stage 6 on
`test_clean.csv`.

`feedback_meme_to_video_8frames.md` adaptations:
  * 8 frames per video
  * one LLM call per video (Label, InPredict) or per pair (Experience)
  * 16 frames for the Experience pairwise call (8 from A + 8 from B)
  * meme→video prompt rewrites in `stages.py`
"""

import argparse
import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from alarm_video_dataset import ALL_DATASETS, build_video_items  # noqa: E402
from qwen2vl_video_model import DEFAULT_MODEL_ID, Qwen2VLVideoModel  # noqa: E402
import stages  # noqa: E402

PROJECT_ROOT = "/data/jehc223/EMNLP2"
RESULTS_SUBDIR = "alarm"


def _ds_paths(dataset: str):
    root = os.path.join(PROJECT_ROOT, "results", RESULTS_SUBDIR, dataset)
    return {
        "root": root,
        "label": os.path.join(root, "label.jsonl"),
        "fea": os.path.join(root, "fea"),
        "pairs": os.path.join(root, "retrieve", "pairs.jsonl"),
        "experience": os.path.join(root, "experience.jsonl"),
        "reference": os.path.join(root, "reference.json"),
        "test_out": os.path.join(root, "test_alarm.jsonl"),
    }


def run_one_dataset(dataset: str, args, model: Qwen2VLVideoModel):
    paths = _ds_paths(dataset)
    os.makedirs(paths["root"], exist_ok=True)

    logging.info(f"[{dataset}] building train items")
    train_items, train_missing = build_video_items(
        dataset, "train", include_frames=True
    )
    logging.info(
        f"[{dataset}] train items: {len(train_items)}  missing={train_missing}"
    )

    logging.info(f"[{dataset}] building test items")
    test_items, test_missing = build_video_items(
        dataset, "test", include_frames=True
    )
    logging.info(
        f"[{dataset}] test items: {len(test_items)}  missing={test_missing}"
    )

    train_by_id = {it["id"]: it for it in train_items}

    # ---------- Stage 1 — Label (on train) ----------
    if args.do_label:
        logging.info(f"[{dataset}] Stage 1: Label on train split")
        stages.run_label(model, train_items, paths["label"])

    # ---------- Stage 2 — Jina-CLIP embeddings (on train) ----------
    if args.do_embed:
        logging.info(f"[{dataset}] Stage 2: make_embeddings on train split")
        # Flush any pending CUDA errors from Stage 1 and free cache
        # before loading Jina-CLIP alongside the 72B model.
        import torch, gc
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        stages.run_make_embeddings(train_items, paths["fea"])

    # ---------- Stage 3 — conduct_retrieval ----------
    if args.do_retrieve:
        logging.info(f"[{dataset}] Stage 3: conduct_retrieval")
        stages.run_conduct_retrieval(
            label_jsonl_path=paths["label"],
            fea_dir=paths["fea"],
            out_pairs_path=paths["pairs"],
            coverage_rate=args.coverage_rate,
        )

    # ---------- Stage 4 — Experience (pairwise video-vs-video) ----------
    if args.do_experience:
        logging.info(f"[{dataset}] Stage 4: Experience on retrieved pairs")
        stages.run_experience(
            model=model,
            pairs_path=paths["pairs"],
            items_by_id=train_by_id,
            out_path=paths["experience"],
        )

    # ---------- Stage 5 — Reference distillation ----------
    if args.do_reference:
        logging.info(f"[{dataset}] Stage 5: Reference set distillation")
        stages.run_reference(
            model=model,
            experience_path=paths["experience"],
            reference_out_path=paths["reference"],
            size=args.reference_size,
        )

    # ---------- Stage 6 — InPredict (on test) ----------
    if args.do_inpredict:
        logging.info(f"[{dataset}] Stage 6: InPredict on test split")
        stages.run_inpredict(
            model=model,
            test_items=test_items,
            reference_path=paths["reference"],
            out_path=paths["test_out"],
        )
    logging.info(f"[{dataset}] done. Final output → {paths['test_out']}")


def main():
    parser = argparse.ArgumentParser(
        description="ALARM video reproduction — 5-stage pipeline"
    )
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)

    # Per-stage toggles — default all on, but callers can re-run
    # individual stages for debugging / resume.
    parser.add_argument("--do-label", action="store_true", default=True)
    parser.add_argument("--no-do-label", dest="do_label", action="store_false")
    parser.add_argument("--do-embed", action="store_true", default=True)
    parser.add_argument("--no-do-embed", dest="do_embed", action="store_false")
    parser.add_argument("--do-retrieve", action="store_true", default=True)
    parser.add_argument("--no-do-retrieve", dest="do_retrieve", action="store_false")
    parser.add_argument("--do-experience", action="store_true", default=True)
    parser.add_argument("--no-do-experience", dest="do_experience", action="store_false")
    parser.add_argument("--do-reference", action="store_true", default=True)
    parser.add_argument("--no-do-reference", dest="do_reference", action="store_false")
    parser.add_argument("--do-inpredict", action="store_true", default=True)
    parser.add_argument("--no-do-inpredict", dest="do_inpredict", action="store_false")

    # Upstream defaults from config yamls.
    parser.add_argument(
        "--coverage-rate",
        type=float,
        default=0.5,
        help="High-confidence coverage rate for retrieval pool (upstream iterates 0.1..1.0; we pick 0.5 as the midpoint)",
    )
    parser.add_argument(
        "--reference-size",
        type=int,
        default=20,
        help="Max size of the distilled reference set",
    )
    args = parser.parse_args()
    if not args.dataset and not args.all:
        parser.error("Provide --dataset or --all")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    datasets = ALL_DATASETS if args.all else [args.dataset]
    logging.info(
        f"ALARM repro: datasets={datasets} model={args.model}  "
        f"stages=[label={args.do_label} embed={args.do_embed} "
        f"retrieve={args.do_retrieve} experience={args.do_experience} "
        f"reference={args.do_reference} inpredict={args.do_inpredict}]"
    )

    logging.info(f"Loading model {args.model} (Qwen2.5-VL-72B-AWQ via HF)")
    model = Qwen2VLVideoModel(model_id=args.model)
    logging.info("Model loaded")

    for ds in datasets:
        run_one_dataset(ds, args, model)

    logging.info("All datasets done.")


if __name__ == "__main__":
    main()
