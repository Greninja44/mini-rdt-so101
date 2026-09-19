#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore
PY=.venv/bin/python; D=artifacts/phase7/imageonly
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a artifacts/phase7.log; }
if [ ! -f $D/result.json ]; then
  rm -rf $D; log "7c train image-only"
  $PY -m training.audit_overfit --output $D --state-dropout 1.0 --schedule cosine --prediction x0 --padding hold --steps 20000 --seed 17 \
      --batch-size 8 --learning-rate 0.001 --device cuda --eval-interval 2000 --ema-decay 0.999 > $D.log 2>&1
fi
[ -f $D/diagnostics_ema_last/diagnostics.json ] || $PY -m evaluation.research_audit --checkpoint $D/ema_last.pt --output $D/diagnostics_ema_last --device cuda > $D/diagnostics.log 2>&1
log "7c closed-loop"
for eps in "0 2 3 4 5" "6 7 8 9 11"; do
  $PY -m evaluation.closed_loop --policy tinyrdt --checkpoint $D/ema_last.pt --skip-replay --episodes $eps --k 1 2 4 8 --max-steps 150 --output artifacts/closed_loop_v3/IO >> artifacts/phase7/IO_closed_loop.log 2>&1 &
done; wait
log "7c recovery + prefix"
for eps in "0 2 3 4 5" "6 7 8 9 11"; do
  $PY -m evaluation.recovery --checkpoint $D/ema_last.pt --output artifacts/recovery/IO --episodes $eps >> artifacts/phase7/IO_recovery.log 2>&1 &
done; wait
$PY -m evaluation.recovery --checkpoint $D/ema_last.pt --output artifacts/recovery/IO >> artifacts/phase7/IO_recovery.log 2>&1
$PY -m evaluation.recovery --checkpoint $D/ema_last.pt --output artifacts/recovery_prefix/IO --magnitudes 0 --joints shoulder_pan >> artifacts/phase7/IO_prefix.log 2>&1
log "phase 7c complete"
