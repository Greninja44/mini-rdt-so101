#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore
PY=.venv/bin/python; A=artifacts/research_audit/cosine_x0_hold_ema/ema_last.pt
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a artifacts/phase7.log; }
log "7a prefix sweep"
for pair in "1 2" "3 5"; do
  ( for n in $pair; do $PY -m evaluation.recovery --checkpoint $A --output artifacts/phase7/prefix_$n --magnitudes 0 --joints shoulder_pan --t0 $n --duration 0; done ) >> artifacts/phase7/prefix.log 2>&1 &
done; wait
log "7a complete"
