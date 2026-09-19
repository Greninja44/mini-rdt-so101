#!/usr/bin/env bash
# Phase 6 (pre-registered in docs/research/research_log.md): oracle split + spatial vision tokens. Resumable; <=2 sim workers.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONWARNINGS=ignore
PY=.venv/bin/python; S=artifacts/phase6/spatial_clean10
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a artifacts/phase6.log; }
# Wait on log markers / files, not process names (launcher shells can contain script names).
until grep -q "phase 5 remaining complete" artifacts/corrective_phase.log 2>/dev/null; do sleep 60; done
log "6a oracle split"
$PY -m evaluation.oracle_ablation --mode gripper_oracle --output artifacts/phase6/oracle/gripper_oracle >> artifacts/phase6/oracle_gripper.log 2>&1 &
$PY -m evaluation.oracle_ablation --mode arm_oracle --output artifacts/phase6/oracle/arm_oracle >> artifacts/phase6/oracle_arm.log 2>&1 &
wait
if [ ! -f $S/result.json ]; then
  log "spatial training incomplete - retraining"; rm -rf $S
  $PY -m training.audit_overfit --output $S --vision-tokens spatial --schedule cosine --prediction x0 --padding hold --steps 20000 --seed 17 \
      --batch-size 8 --learning-rate 0.001 --device cuda --eval-interval 2000 --ema-decay 0.999 > $S.log 2>&1
fi
log "6b spatial offline diagnostics"
[ -f $S/diagnostics_ema_last/diagnostics.json ] || $PY -m evaluation.research_audit --checkpoint $S/ema_last.pt --output $S/diagnostics_ema_last --device cuda > $S/diagnostics.log 2>&1
log "6b spatial closed-loop"
for eps in "0 2 3 4 5" "6 7 8 9 11"; do
  $PY -m evaluation.closed_loop --policy tinyrdt --checkpoint $S/ema_last.pt --skip-replay --episodes $eps --k 1 2 4 8 --max-steps 150 --output artifacts/closed_loop_v3/S >> artifacts/phase6/S_closed_loop.log 2>&1 &
done; wait
log "6b spatial recovery"
for eps in "0 2 3 4 5" "6 7 8 9 11"; do
  $PY -m evaluation.recovery --checkpoint $S/ema_last.pt --output artifacts/recovery/S --episodes $eps >> artifacts/phase6/S_recovery.log 2>&1 &
done; wait
$PY -m evaluation.recovery --checkpoint $S/ema_last.pt --output artifacts/recovery/S >> artifacts/phase6/S_recovery.log 2>&1
log "phase 6 complete"
