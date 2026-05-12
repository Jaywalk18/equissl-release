#!/bin/bash
# Phase-2 serial queue runner for a single GPU.
# Reads one-job-per-line queue file, runs each sequentially, logs to
# .ops/q_gpu<N>.log. Skips jobs whose target results.pth already exists
# (resumable after machine reboot or mid-queue interrupt). Continues to
# next job even if a job fails (non-zero exit).
#
# Usage (tmux-friendly):
#   GPU=0 bash scripts/run_phase2_queue.sh .ops/q_gpu0.sh
#
# Queue file format: each non-empty non-comment line is a full bash command
# (including stage1 && stage2 chains). Comments starting with # are skipped.
# A job is considered "done" if ANY `--output_dir outputs/<path>` in the
# line has a `<path>/results.pth` present.

set -o pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

GPU="${GPU:?set GPU=<id> before invoking}"
QUEUE="${1:?usage: GPU=<n> $0 <queue_file>}"

mkdir -p .ops
LOG=".ops/q_gpu${GPU}.log"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy NO_PROXY no_proxy
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/../sphere_uformer/src:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU"
export TMPDIR=/mnt/ssd/tmp

{
  echo "============================================"
  echo "[$(date)] Phase-2 queue runner started"
  echo "  GPU: $GPU"
  echo "  Queue: $QUEUE"
  echo "  Project: $PROJECT_ROOT"
  echo "============================================"
} >> "$LOG"

total=$(grep -cvE '^\s*(#|$)' "$QUEUE" || echo 0)
idx=0
done=0
skipped=0
failed=0

while IFS= read -r cmd || [ -n "$cmd" ]; do
  # skip empties / comments
  [[ -z "${cmd// }" || "$cmd" =~ ^[[:space:]]*# ]] && continue
  idx=$((idx + 1))

  # Detect the terminal (last) --output_dir in the command chain (for
  # iBOT 2-stage, the stage2 dir is the one whose results.pth signals done)
  final_out=$(echo "$cmd" | grep -oE -- '--output_dir[[:space:]]+[^[:space:]]+' | tail -1 | awk '{print $2}')
  if [ -n "$final_out" ] && [ -f "${final_out}/results.pth" ]; then
    echo "[$(date)] [$idx/$total] SKIP (done): $final_out" >> "$LOG"
    skipped=$((skipped + 1))
    continue
  fi

  echo "[$(date)] [$idx/$total] RUN: target=$final_out" >> "$LOG"
  echo "[$(date)] [$idx/$total] CMD: $cmd" >> "$LOG"

  bash -c "$cmd" >> "$LOG" 2>&1
  rc=$?

  if [ $rc -eq 0 ]; then
    echo "[$(date)] [$idx/$total] DONE (exit=0): $final_out" >> "$LOG"
    done=$((done + 1))
  else
    echo "[$(date)] [$idx/$total] FAIL (exit=$rc): $final_out — continuing queue" >> "$LOG"
    failed=$((failed + 1))
  fi
done < "$QUEUE"

{
  echo "============================================"
  echo "[$(date)] Queue exhausted"
  echo "  Total entries : $total"
  echo "  Skipped (done): $skipped"
  echo "  Completed     : $done"
  echo "  Failed        : $failed"
  echo "============================================"
} >> "$LOG"
