#!/usr/bin/env bash
# Service entrypoint for the optimization scaling study.
#
# Runs whichever stage manifest the pointer names, then derives the next stage from measured training
# health and runs that too, until the staged logic has nothing left to queue. The pointer file lets a
# reboot resume the stage that was in flight without editing the unit; the loop means a completed stage
# advances on its own rather than waiting for a hand-written pointer.
#
# build_next_stage.py decides what comes next using only training-health quantities (Amendment 3(b)).
# It emits no runs while any screening run of a capacity is still incomplete, so the loop cannot
# advance past a stage that has not finished.
set -uo pipefail
cd "$(dirname "$0")/../.."
POINTER=artifacts/optimization_scaling_v1/current_stage
LOG=artifacts/optimization_scaling_v1/pipeline.log
PY=.venv/bin/python

for _ in $(seq 1 12); do
  [ -f "$POINTER" ] || { echo "no stage pointer at $POINTER"; exit 0; }
  MANIFEST=$(cat "$POINTER")
  [ -f "$MANIFEST" ] || { echo "stage manifest $MANIFEST missing"; exit 1; }

  /bin/bash scripts/experiments/run_optimization_stage.sh "$MANIFEST" || exit 1

  # Refresh the gate summary, then ask the staged logic what to run next.
  $PY evaluation/optimization_analysis.py >/dev/null 2>&1
  PLAN=$($PY scripts/experiments/build_next_stage.py --write 2>&1)
  echo "$PLAN"
  NEXT=$(echo "$PLAN" | sed -n 's/^wrote \(docs\/research\/[a-z_0-9]*\.json\) .*/\1/p' | tail -1)
  if [ -z "$NEXT" ]; then
    echo "[$(date -u +%FT%TZ)] staged logic has nothing further to queue; study screening complete" >> "$LOG"
    exit 0
  fi
  if [ "$NEXT" = "$MANIFEST" ]; then
    echo "[$(date -u +%FT%TZ)] next stage equals current manifest ($NEXT); stopping to avoid a loop" >> "$LOG"
    exit 0
  fi
  echo "$NEXT" > "$POINTER"
  echo "[$(date -u +%FT%TZ)] advancing to $NEXT" >> "$LOG"
done
