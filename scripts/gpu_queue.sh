#!/bin/bash
# GPU queue orchestrator: watches a GPU, waits for current job to finish,
# then launches the next task in its queue.
#
# Usage: bash scripts/gpu_queue.sh <gpu_id> <current_pid> <queue_config_file>
# queue_config_file: one bash command per line (comments starting with # ignored)

GPU="$1"; WATCH_PID="$2"; QUEUE_FILE="$3"

LOG="/tmp/gpu${GPU}_queue.log"
echo "[$(date)] queue watcher for GPU $GPU, watching pid $WATCH_PID" > "$LOG"

# Wait for current job
while kill -0 "$WATCH_PID" 2>/dev/null; do
    sleep 300  # 5-min poll
done
echo "[$(date)] pid $WATCH_PID finished, starting queue" >> "$LOG"

# Process queue
while IFS= read -r cmd || [ -n "$cmd" ]; do
    # Skip empty lines and comments
    [[ -z "$cmd" || "$cmd" =~ ^[[:space:]]*# ]] && continue

    echo "[$(date)] RUN: $cmd" >> "$LOG"
    bash -c "$cmd" >> "$LOG" 2>&1
    echo "[$(date)] DONE: $cmd (exit=$?)" >> "$LOG"
done < "$QUEUE_FILE"

echo "[$(date)] queue exhausted" >> "$LOG"
