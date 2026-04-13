#!/usr/bin/env bash
# prompt_paradigm v6 — Coarse Axes Prompt pipeline runner.
#
# Five waves of Slurm jobs with own-ID polling (no --dependency, no &, no
# scancel of foreign jobs). Re-startable: each wave checks for output files
# and skips already-complete conditions.
#
# Wave 1: score EN-test axes   + ZH-test axes    (2 concurrent GPU jobs)
# Wave 2: score EN-test control + ZH-test control (2 concurrent GPU jobs)
# Wave 3: score EN-train axes   + ZH-train axes    (2 concurrent GPU jobs)
# Wave 4: score EN-train control + ZH-train control (2 concurrent GPU jobs)
# Wave 5: CPU eval (eval_coarse_axes.py)
#
# Usage:
#   bash src/prompt_paradigm/run_v6_pipeline.sh

set -u

ROOT="/data/jehc223/EMNLP2"
RESULTS_DIR="$ROOT/results/prompt_paradigm"
LOG_DIR="$ROOT/logs"
RUN_LOG="$ROOT/docs/experiments/prompt_paradigm_runs.md"
SCORER="src/prompt_paradigm/coarse_axes_prompt.py"
EVALUATOR="src/prompt_paradigm/eval_coarse_axes.py"

CONDA_SH="/data/jehc223/home/miniconda3/etc/profile.d/conda.sh"
ENV="SafetyContradiction"

mkdir -p "$LOG_DIR"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

append_runlog() {
  echo "$1" >> "$RUN_LOG"
}

out_path() {
  local dataset="$1"
  local split="$2"
  local condition="$3"
  echo "$RESULTS_DIR/$dataset/${split}_coarse_axes_${condition}.jsonl"
}

split_total() {
  local dataset="$1"
  local split="$2"
  local f="/data/jehc223/EMNLP2/datasets/$dataset/splits/${split}_clean.csv"
  if [[ -f "$f" ]]; then
    wc -l < "$f" | tr -d ' '
  else
    echo 0
  fi
}

jsonl_count() {
  local f="$1"
  if [[ -f "$f" ]]; then
    wc -l < "$f" | tr -d ' '
  else
    echo 0
  fi
}

condition_complete() {
  local dataset="$1"
  local split="$2"
  local condition="$3"
  local out
  out="$(out_path "$dataset" "$split" "$condition")"
  local total done_n
  total="$(split_total "$dataset" "$split")"
  done_n="$(jsonl_count "$out")"
  if [[ "$total" -gt 0 && "$done_n" -ge "$total" ]]; then
    return 0
  fi
  return 1
}

submit_score() {
  local dataset="$1"
  local split="$2"
  local condition="$3"
  local jobname="v6_${dataset}_${split}_${condition}"
  local logf="$LOG_DIR/${jobname}.out"
  local cmd="source $CONDA_SH && conda activate $ENV && cd $ROOT && python $SCORER --dataset $dataset --split $split --condition $condition"
  local jid
  jid=$(sbatch --parsable --gres=gpu:1 --job-name="$jobname" \
    --output="$logf" --error="$logf" \
    --wrap "$cmd")
  echo "$jid"
}

submit_eval() {
  local jobname="v6_eval"
  local logf="$LOG_DIR/${jobname}.out"
  local cmd="source $CONDA_SH && conda activate $ENV && cd $ROOT && python $EVALUATOR"
  local jid
  jid=$(sbatch --parsable --job-name="$jobname" \
    --output="$logf" --error="$logf" \
    --wrap "$cmd")
  echo "$jid"
}

poll_jobs() {
  local ids="$1"
  if [[ -z "$ids" ]]; then
    return 0
  fi
  log "Polling jobs: $ids"
  while true; do
    local active
    active=$(squeue -h -j "$ids" 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$active" -eq 0 ]]; then
      log "All polled jobs exited queue."
      return 0
    fi
    local total
    total=$(echo "$ids" | tr ',' '\n' | wc -l | tr -d ' ')
    log "  $active/$total still running, sleeping 30s"
    sleep 30
  done
}

run_wave() {
  local wave_name="$1"
  local split="$2"
  local condition="$3"

  log "=== Wave: $wave_name ($split / $condition) ==="
  local ids=()
  for ds in "MHClip_EN" "MHClip_ZH"; do
    if condition_complete "$ds" "$split" "$condition"; then
      log "  $ds/$split/$condition already complete, skipping"
      continue
    fi
    local jid
    jid="$(submit_score "$ds" "$split" "$condition")"
    log "  submitted $ds/$split/$condition -> job $jid"
    append_runlog "- v6 runner $wave_name: $ds/$split/$condition job $jid"
    ids+=("$jid")
  done

  if [[ ${#ids[@]} -gt 0 ]]; then
    local joined
    joined=$(IFS=','; echo "${ids[*]}")
    poll_jobs "$joined"
  fi

  for ds in "MHClip_EN" "MHClip_ZH"; do
    if ! condition_complete "$ds" "$split" "$condition"; then
      log "ERROR: $ds/$split/$condition still incomplete after wave $wave_name"
      return 1
    fi
  done
  log "Wave $wave_name complete."
  return 0
}

append_runlog ""
append_runlog "### v6 pipeline runner started $(date '+%Y-%m-%d %H:%M:%S')"

if ! run_wave "W1 test/axes"    "test"  "axes";    then log "Wave 1 failed"; exit 1; fi
if ! run_wave "W2 test/control" "test"  "control"; then log "Wave 2 failed"; exit 1; fi
if ! run_wave "W3 train/axes"   "train" "axes";    then log "Wave 3 failed"; exit 1; fi
if ! run_wave "W4 train/control" "train" "control"; then log "Wave 4 failed"; exit 1; fi

log "=== Wave 5: eval ==="
REPORT="$RESULTS_DIR/report_v6.json"
if [[ -f "$REPORT" ]]; then
  log "  report_v6.json already exists, re-running eval to refresh"
fi
EVAL_JID="$(submit_eval)"
log "  submitted eval -> job $EVAL_JID"
append_runlog "- v6 runner W5 eval: job $EVAL_JID"
poll_jobs "$EVAL_JID"

if [[ -f "$REPORT" ]]; then
  log "Pipeline complete. Report at $REPORT"
  append_runlog "- v6 runner finished $(date '+%Y-%m-%d %H:%M:%S')"
  exit 0
else
  log "ERROR: eval finished but report missing"
  exit 1
fi
