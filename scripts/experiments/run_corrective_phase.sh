#!/usr/bin/env bash
# Phase 5 pipeline (pre-registered in docs/research/research_log.md). Fully resumable: rerun after any interruption.
# Never more than 2 simulation workers at once.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONWARNINGS=ignore
PY=.venv/bin/python
A=artifacts/research_audit/cosine_x0_hold_ema/ema_last.pt
COL=artifacts/corrective; TR=artifacts/corrective_train; EV=artifacts/closed_loop_v3; RC=artifacts/recovery
mkdir -p "$COL" "$TR" "$EV" "$RC"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a artifacts/corrective_phase.log; }

log "collection (resumable)"
$PY -m data.collect_corrective --mode perturb --output $COL/perturb >> $COL/perturb.log 2>&1 &
$PY -m data.collect_corrective --mode dagger  --output $COL/dagger  >> $COL/dagger.log 2>&1 &
wait

C_FRAMES=$($PY -c "import glob,numpy as np;print(sum(len(np.load(f)) for f in glob.glob('$COL/dagger/episodes/episode_*[0-9]/t.npy')))")
log "C frames = $C_FRAMES"

train() {  # name, extra args...
  local name=$1; shift
  if [ -f "$TR/$name/result.json" ]; then log "train $name: done"; return; fi
  rm -rf "$TR/$name"; log "train $name"
  $PY -m training.audit_overfit --output "$TR/$name" --schedule cosine --prediction x0 --padding hold --steps 20000 --seed 17 \
      --batch-size 8 --learning-rate 0.001 --device cuda --eval-interval 2000 --ema-decay 0.999 "$@" > "$TR/$name.log" 2>&1
}
train B  --corrective $COL/perturb --corrective-fraction 0.5
train C  --corrective $COL/dagger  --corrective-fraction 0.5
train Bm --corrective $COL/perturb --corrective-fraction 0.5 --corrective-max-frames "$C_FRAMES"

evaluate() {  # episodes, tag
  for v in C B Bm; do
    $PY -m evaluation.closed_loop --policy tinyrdt --checkpoint $TR/$v/ema_last.pt --skip-replay --episodes $1 --k 1 2 4 8 --max-steps 150 --output $EV/$v
  done
}
log "normal closed-loop evaluation"
evaluate "0 2 3 4 5" > $EV/worker_a.log 2>&1 &
evaluate "6 7 8 9 11" > $EV/worker_b.log 2>&1 &
wait

log "recovery benchmark"
( $PY -m evaluation.recovery --checkpoint $A --output $RC/A; $PY -m evaluation.recovery --checkpoint $TR/B/ema_last.pt --output $RC/B ) > $RC/worker_a.log 2>&1 &
( $PY -m evaluation.recovery --checkpoint $TR/C/ema_last.pt --output $RC/C; $PY -m evaluation.recovery --checkpoint $TR/Bm/ema_last.pt --output $RC/Bm ) > $RC/worker_b.log 2>&1 &
wait
log "phase 5 pipeline complete"
