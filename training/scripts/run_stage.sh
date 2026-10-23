#!/usr/bin/env bash
# Run one training stage under a supervisor that resumes it after a crash.
#
#   scripts/run_stage.sh configs/large/stage2_prune_s80.yaml [--set key=value ...]
#
# The trainer resumes from <save.dir>/latest.pt on every launch (--resume auto).
# Exit 0 = stage finished; exit 3 = stopped by SIGINT/SIGTERM (stays stopped);
# anything else = crash -> relaunch after a short pause, at most MAX_CRASHES times
# in a row (a run that gets past more steps resets the count).
# Stop it cleanly with:  kill -TERM <pid of this script>   (forwarded to the trainer)
set -uo pipefail
cd "$(dirname "$0")/.."
CONFIG="$1"; shift
MAX_CRASHES=${MAX_CRASHES:-5}
NAME=${LOG_PREFIX:-}$(basename "$CONFIG" .yaml)   # LOG_PREFIX keeps same-named stages of another chain apart
LOG_DIR=runs/logs; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/$NAME.log"
source scripts/env.sh
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

child=0
stop_requested=0
on_term() { stop_requested=1; if [ "$child" -ne 0 ]; then kill -TERM "$child" 2>/dev/null; fi; }
trap on_term TERM INT

crashes=0
while [ "$stop_requested" -eq 0 ]; do
    echo "=== $(date '+%F %T') launching $CONFIG $* (crashes in a row: $crashes)" | tee -a "$LOG"
    "$PY" -m sota_train.train "$CONFIG" "$@" >>"$LOG" 2>&1 &
    child=$!
    wait "$child"; rc=$?
    # wait returns early when a trapped signal arrives; wait again for the real exit code
    while kill -0 "$child" 2>/dev/null; do wait "$child"; rc=$?; done
    child=0
    echo "=== $(date '+%F %T') trainer exited with $rc" | tee -a "$LOG"
    if [ "$rc" -eq 0 ]; then exit 0; fi
    if [ "$rc" -eq 3 ] || [ "$stop_requested" -eq 1 ]; then echo "stopped; rerun this script to resume" | tee -a "$LOG"; exit 3; fi
    crashes=$((crashes + 1))
    if [ "$crashes" -ge "$MAX_CRASHES" ]; then echo "giving up after $crashes crashes in a row" | tee -a "$LOG"; exit "$rc"; fi
    sleep 30 & child=$!; wait "$child"; child=0   # interruptible back-off
done
echo "stopped during back-off; rerun this script to resume" | tee -a "$LOG"
exit 3
