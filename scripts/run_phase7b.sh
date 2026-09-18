#!/usr/bin/env bash
# Phase 7b: state-dropout TinyRDT (pre-registered). Resumable; waits on files/log markers only.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore
PY=.venv/bin/python; D=artifacts/phase7/statedrop_p03
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a artifacts/phase7.log; }
if [ ! -f $D/result.json ]; then
  rm -rf $D; log "7b train state dropout p=0.3"
  $PY -m training.audit_overfit --output $D --state-dropout 0.3 --schedule cosine --prediction x0 --padding hold --steps 20000 --seed 17 \
      --batch-size 8 --learning-rate 0.001 --device cuda --eval-interval 2000 --ema-decay 0.999 > $D.log 2>&1
fi
[ -f $D/diagnostics_ema_last/diagnostics.json ] || $PY -m evaluation.research_audit --checkpoint $D/ema_last.pt --output $D/diagnostics_ema_last --device cuda > $D/diagnostics.log 2>&1
until grep -q "7a complete" artifacts/phase7.log 2>/dev/null; do sleep 60; done
log "7b closed-loop"
for eps in "0 2 3 4 5" "6 7 8 9 11"; do
  $PY -m evaluation.closed_loop --policy tinyrdt --checkpoint $D/ema_last.pt --skip-replay --episodes $eps --k 1 2 4 8 --max-steps 150 --output artifacts/closed_loop_v3/SD >> artifacts/phase7/SD_closed_loop.log 2>&1 &
done; wait
log "7b recovery"
for eps in "0 2 3 4 5" "6 7 8 9 11"; do
  $PY -m evaluation.recovery --checkpoint $D/ema_last.pt --output artifacts/recovery/SD --episodes $eps >> artifacts/phase7/SD_recovery.log 2>&1 &
done; wait
$PY -m evaluation.recovery --checkpoint $D/ema_last.pt --output artifacts/recovery/SD >> artifacts/phase7/SD_recovery.log 2>&1
log "phase 7b complete"
