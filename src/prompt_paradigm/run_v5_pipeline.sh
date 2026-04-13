#!/usr/bin/env bash
# prompt_paradigm v5 — pipeline runner.
#
# Three waves of Slurm jobs with own-ID polling (no --dependency, no &, no scancel).
# Re-startable: each wave checks for existing outputs and skips already-scored
# splits. Must run on login node as foreground bash.
#
# Wave 1: score EN-test + ZH-test (2 concurrent GPU jobs)
# Wave 2: score EN-train + ZH-train (2 concurrent GPU jobs)
# Wave 3: CPU eval (eval_per_rule.py)
#
# Usage:
#   bash src/prompt_paradigm/run_v5_pipeline.sh

set -u

ROOT="/data/jehc223/EMNLP2"
RESULTS_DIR="$ROOT/results/prompt_paradigm"
LOG_DIR="$ROOT/logs"
RUN_LOG="$ROOT/docs/experiments/prompt_paradigm_runs.md"
SCORER="src/prompt_paradigm/per_rule_readout.py"
EVALUATOR="src/prompt_paradigm/eval_per_rule.py"

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
  echo "$RESULTS_DIR/$dataset/${split}_per_rule.jsonl"
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

wave_complete() {
  local dataset="$1"
  local split="$2"
  local out
  out="$(out_path "$dataset" "$split")"
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
  local jobname="v5_${dataset}_${split}"
  local logf="$LOG_DIR/${jobname}.out"
  local cmd="source $CONDA_SH && conda activate $ENV && cd $ROOT && python $SCORER --dataset $dataset --split $split"
  local jid
  jid=$(sbatch --parsable --gres=gpu:1 --job-name="$jobname" \
    --output="$logf" --error="$logf" \
    --wrap "$cmd")
  echo "$jid"
}

submit_eval() {
  local jobname="v5_eval"
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
    log "  $active/$(echo "$ids" | tr ',' '\n' | wc -l | tr -d ' ') still running, sleeping 30s"
    sleep 30
  done
}

run_wave() {
  local wave_name="$1"
  local d1="$2"
  local s1="$3"
  local d2="$4"
  local s2="$5"

  log "=== Wave: $wave_name ==="
  local ids=()
  if wave_complete "$d1" "$s1"; then
    log "  $d1/$s1 already complete, skipping"
  else
    local jid1
    jid1="$(submit_score "$d1" "$s1")"
    log "  submitted $d1/$s1 -> job $jid1"
    append_runlog "- v5 runner $wave_name: $d1/$s1 job $jid1"
    ids+=("$jid1")
  fi
  if wave_complete "$d2" "$s2"; then
    log "  $d2/$s2 already complete, skipping"
  else
    local jid2
    jid2="$(submit_score "$d2" "$s2")"
    log "  submitted $d2/$s2 -> job $jid2"
    append_runlog "- v5 runner $wave_name: $d2/$s2 job $jid2"
    ids+=("$jid2")
  fi
  if [[ ${#ids[@]} -gt 0 ]]; then
    local joined
    joined=$(IFS=','; echo "${ids[*]}")
    poll_jobs "$joined"
  fi

  if ! wave_complete "$d1" "$s1"; then
    log "ERROR: $d1/$s1 still incomplete after wave $wave_name"
    return 1
  fi
  if ! wave_complete "$d2" "$s2"; then
    log "ERROR: $d2/$s2 still incomplete after wave $wave_name"
    return 1
  fi
  log "Wave $wave_name complete."
  return 0
}

append_runlog ""
append_runlog "### v5 pipeline runner started $(date '+%Y-%m-%d %H:%M:%S')"

if ! run_wave "W1 test" "MHClip_EN" "test" "MHClip_ZH" "test"; then
  log "Wave 1 failed, aborting."
  exit 1
fi

if ! run_wave "W2 train" "MHClip_EN" "train" "MHClip_ZH" "train"; then
  log "Wave 2 failed, aborting."
  exit 1
fi

log "=== Wave 3: eval ==="
REPORT="$RESULTS_DIR/report_v5.json"
if [[ -f "$REPORT" ]]; then
  log "  report_v5.json already exists, re-running eval to refresh"
fi
EVAL_JID="$(submit_eval)"
log "  submitted eval -> job $EVAL_JID"
append_runlog "- v5 runner W3 eval: job $EVAL_JID"
poll_jobs "$EVAL_JID"

if [[ -f "$REPORT" ]]; then
  log "Pipeline complete. Report at $REPORT"
  append_runlog "- v5 runner finished $(date '+%Y-%m-%d %H:%M:%S')"
  exit 0
else
  log "ERROR: eval finished but report missing"
  exit 1
fi
