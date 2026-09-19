#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore
PY=.venv/bin/python
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a artifacts/phase8.log; }
log "8b K=16"
( $PY -m evaluation.closed_loop --policy tinyrdt --checkpoint artifacts/research_audit/cosine_x0_hold_ema/ema_last.pt --skip-replay --k 16 --max-steps 150 --episodes 0 2 3 4 5 6 7 8 9 11 --output artifacts/closed_loop_v3/K16_A
  $PY -m evaluation.closed_loop --policy tinyrdt --checkpoint artifacts/corrective_train/C40k/ema_last.pt --skip-replay --k 16 --max-steps 150 --episodes 0 2 3 4 5 6 7 8 9 11 --output artifacts/closed_loop_v3/K16_C40k
  $PY -m evaluation.closed_loop --policy tinyrdt --checkpoint artifacts/phase8/train80/ema_last.pt --skip-replay --k 16 --max-steps 150 --episodes 10 29 41 49 68 72 79 89 90 96 --output artifacts/closed_loop_v3/K16_D80_validation ) >> artifacts/phase8/k16_a.log 2>&1 &
( $PY -m evaluation.closed_loop --policy tinyrdt --checkpoint artifacts/phase6/spatial_clean10/ema_last.pt --skip-replay --k 16 --max-steps 150 --episodes 0 2 3 4 5 6 7 8 9 11 --output artifacts/closed_loop_v3/K16_S
  $PY -m evaluation.closed_loop --policy tinyrdt --checkpoint artifacts/phase8/train80/ema_last.pt --skip-replay --k 16 --max-steps 150 --episodes 0 2 3 4 5 6 7 8 9 11 --output artifacts/closed_loop_v3/K16_D80_memorised ) >> artifacts/phase8/k16_b.log 2>&1 &
wait
log "phase 8b complete"
