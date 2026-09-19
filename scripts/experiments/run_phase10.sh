#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONWARNINGS=ignore
PY=.venv/bin/python
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a artifacts/phase10.log; }
mkdir -p artifacts/phase10
train() { local name=$1; shift; local D=artifacts/phase10/$name
  if [ ! -f $D/result.json ]; then rm -rf $D; log "10 train $name"
    $PY -m training.audit_overfit --output $D --train-vision --vision-learning-rate 1e-4 --schedule cosine --prediction x0 --padding hold --steps 20000 --seed 17 \
      --batch-size 8 --learning-rate 0.001 --device cuda --eval-interval 2000 --ema-decay 0.999 "$@" > $D.log 2>&1; fi
  [ -f $D/diagnostics_ema_last/diagnostics.json ] || $PY -m evaluation.research_audit --checkpoint $D/ema_last.pt --output $D/diagnostics_ema_last --device cuda > $D/diagnostics.log 2>&1; }
evaluate() { local name=$1; local CK=artifacts/phase10/$name/ema_last.pt
  log "10 $name normal"
  for eps in "0 2 3 4 5" "6 7 8 9 11"; do
    $PY -m evaluation.closed_loop --policy tinyrdt --checkpoint $CK --skip-replay --episodes $eps --k 1 2 4 8 --max-steps 150 --output artifacts/closed_loop_v3/$name >> artifacts/phase10/$name.eval.log 2>&1 &
  done; wait
  log "10 $name random starts"
  $PY -m evaluation.closed_loop --policy tinyrdt --checkpoint $CK --start-seed 777 --k 4 8 --max-steps 150 --episodes 0 2 3 4 5 --output artifacts/closed_loop_v3/${name}_start777 >> artifacts/phase10/$name.eval.log 2>&1 &
  $PY -m evaluation.closed_loop --policy tinyrdt --checkpoint $CK --start-seed 777 --k 4 8 --max-steps 150 --episodes 6 7 8 9 11 --output artifacts/closed_loop_v3/${name}_start777 >> artifacts/phase10/$name.eval.log 2>&1 &
  wait
  log "10 $name recovery"
  for eps in "0 2 3 4 5" "6 7 8 9 11"; do
    $PY -m evaluation.recovery --checkpoint $CK --output artifacts/recovery/$name --episodes $eps >> artifacts/phase10/$name.eval.log 2>&1 &
  done; wait
  $PY -m evaluation.recovery --checkpoint $CK --output artifacts/recovery/$name >> artifacts/phase10/$name.eval.log 2>&1; }
train V1
train V2 --vision-tokens spatial &
evaluate V1
wait
evaluate V2
log "phase 10 complete"
