#!/usr/bin/env bash
# The full SOTA recipe for one model size, after scripts/00-02 (see run.sh):
#   Stage 1 CTC fine-tune (the teacher) -> teacher.pt -> Stage 2 prune to 80% sparsity ->
#   cut (04_apply_pruning) -> Stage 2b distil with the mask frozen -> Stage 3 CTC + logit KD ->
#   test-set scores for the pruned model and the teacher (runs/eval/<size>{,_teacher}.json).
#
#   scripts/run_chain.sh large|base
#
# Rerunning resumes: finished steps are skipped (DONE / output file), and run_stage.sh
# resumes a stage from its latest.pt. Any step that does not exit 0 stops the chain.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
SIZE=${1:?usage: run_chain.sh large|base}
R=runs/$SIZE
C=configs/$SIZE
[ -d "$C" ] || { echo "no configs in $C"; exit 2; }
export LOG_PREFIX=${SIZE}_   # runs/logs/<size>_<stage>.log
mkdir -p runs/logs runs/eval
log() { echo "=== $(date '+%F %T') $SIZE: $*" | tee -a "runs/logs/chain_$SIZE.log"; }

# a service manager stops the chain by signalling this script only; pass it on so the trainer saves
child=0
on_term() { if [ "$child" -ne 0 ]; then kill -TERM "$child" 2>/dev/null; fi; }
trap on_term TERM INT
run() {
    "$@" & child=$!
    local rc=0
    wait "$child" || rc=$?
    while kill -0 "$child" 2>/dev/null; do rc=0; wait "$child" || rc=$?; done
    child=0
    return "$rc"
}

stage() {
    local name=$1
    if [ -f "$R/$name/DONE" ]; then log "$name already done"; return; fi
    log "start $name"
    run scripts/run_stage.sh "$C/$name.yaml"
    [ -f "$R/$name/DONE" ] || { log "$name exited without DONE"; exit 1; }
}

stage stage1_finetune

if [ ! -f "$R/stage1_finetune/teacher.pt" ]; then
    log "export stage 1 weights -> teacher.pt"
    run "$PY" -m sota_train.ckpt export-weights "$R/stage1_finetune/latest.pt" "$R/stage1_finetune/teacher.pt"
fi

stage stage2_prune_s80

if [ ! -f "$R/stage2_prune_s80/pruned.pt" ]; then
    log "apply pruning"
    run "$PY" scripts/04_apply_pruning.py "$R/stage2_prune_s80/latest.pt" --out "$R/stage2_prune_s80/pruned.pt" --target 0.8
fi

stage stage2b_distill
stage stage3_recover

if [ ! -f "runs/eval/$SIZE.json" ]; then
    log "test-set evaluation"
    run "$PY" scripts/03_evaluate.py "$R/stage3_recover/latest.pt" --name "$SIZE"
fi
if [ ! -f "runs/eval/${SIZE}_teacher.json" ]; then
    log "teacher evaluation"
    run "$PY" scripts/03_evaluate.py "$R/stage1_finetune/teacher.pt" --name "${SIZE}_teacher"
fi
log "chain finished"
