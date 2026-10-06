#!/usr/bin/env bash
# Service entrypoint for the optimization scaling study: runs whichever stage manifest the pointer names.
# The pointer file lets a reboot resume the stage that was in flight without editing the unit.
set -uo pipefail
cd "$(dirname "$0")/../.."
POINTER=artifacts/optimization_scaling_v1/current_stage
[ -f "$POINTER" ] || { echo "no stage pointer at $POINTER"; exit 0; }
MANIFEST=$(cat "$POINTER")
[ -f "$MANIFEST" ] || { echo "stage manifest $MANIFEST missing"; exit 1; }
exec /bin/bash scripts/experiments/run_optimization_stage.sh "$MANIFEST"
