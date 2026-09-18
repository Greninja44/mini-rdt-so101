#!/usr/bin/env bash
# Remaining Phase 5 work after the recovery filename fix: C40k normal eval + full recovery benchmark. Resumable, 2 workers.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore
PY=.venv/bin/python
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a artifacts/corrective_phase.log; }
ck() { case $1 in A) echo artifacts/research_audit/cosine_x0_hold_ema/ema_last.pt;; *) echo artifacts/corrective_train/$1/ema_last.pt;; esac; }
log "C40k normal evaluation"
for eps in "0 2 3 4 5" "6 7 8 9 11"; do
  $PY -m evaluation.closed_loop --policy tinyrdt --checkpoint $(ck C40k) --skip-replay --episodes $eps --k 1 2 4 8 --max-steps 150 --output artifacts/closed_loop_v3/C40k >> artifacts/closed_loop_v3/C40k.log 2>&1 &
done; wait
for v in A C C40k B Bm; do
  log "recovery $v"
  for eps in "0 2 3 4 5" "6 7 8 9 11"; do
    $PY -m evaluation.recovery --checkpoint $(ck $v) --output artifacts/recovery/$v --episodes $eps >> artifacts/recovery/$v.log 2>&1 &
  done; wait
  $PY -m evaluation.recovery --checkpoint $(ck $v) --output artifacts/recovery/$v >> artifacts/recovery/$v.log 2>&1   # all done: rewrites the full summary
done
log "phase 5 remaining complete"
