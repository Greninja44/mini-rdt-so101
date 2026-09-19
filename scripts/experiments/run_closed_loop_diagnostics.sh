#!/usr/bin/env bash
# Balanced memorized-scene closed-loop sweep: every policy x every K on the SAME 10 seeds.
# Resumable (finished rollouts are skipped). Two workers max on this 7.8 GB host.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONWARNINGS=ignore
PY=.venv/bin/python
OUT=artifacts/closed_loop_v2
mkdir -p "$OUT"
worker() {
  local eps="$1" tag="$2"
  $PY -m evaluation.closed_loop --policy tinyrdt --checkpoint artifacts/research_audit/cosine_x0_hold_ema/ema_last.pt \
      --episodes $eps --k 1 2 4 8 --max-steps 150 --output $OUT/tinyrdt_ema
  $PY -m evaluation.closed_loop --policy bc --checkpoint artifacts/research_audit/bc_rgb/best.pt --skip-replay \
      --episodes $eps --k 1 2 4 8 --max-steps 150 --output $OUT/bc_rgb
  $PY -m evaluation.closed_loop --policy bc --checkpoint artifacts/research_audit/bc_privileged/best.pt --skip-replay \
      --episodes $eps --k 1 2 4 8 --max-steps 150 --output $OUT/bc_privileged
}
worker "0 2 3 4 5" a > "$OUT/worker_a.log" 2>&1 &
worker "6 7 8 9 11" b > "$OUT/worker_b.log" 2>&1 &
wait
