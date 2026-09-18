#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore
PY=.venv/bin/python
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a artifacts/phase6.log; }
until grep -q "phase 6 complete" artifacts/phase6.log 2>/dev/null; do sleep 60; done
log "6c expert-prefix control"
$PY -m evaluation.recovery --checkpoint artifacts/research_audit/cosine_x0_hold_ema/ema_last.pt --output artifacts/recovery_prefix/A --magnitudes 0 --joints shoulder_pan >> artifacts/phase6/prefix.log 2>&1 &
$PY -m evaluation.recovery --checkpoint artifacts/phase6/spatial_clean10/ema_last.pt --output artifacts/recovery_prefix/S --magnitudes 0 --joints shoulder_pan >> artifacts/phase6/prefix.log 2>&1 &
wait
log "phase 6c complete"
