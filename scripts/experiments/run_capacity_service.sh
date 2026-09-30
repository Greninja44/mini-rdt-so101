#!/usr/bin/env bash
# User-systemd entrypoint for the resumable capacity study.  A successful,
# already-finished pipeline is deliberately a no-op; interrupted work resumes
# from its per-run checkpoints via run_capacity_scaling.sh.
set -uo pipefail
cd "$(dirname "$0")/../.."

if grep -q "capacity scaling pipeline complete" artifacts/capacity_scaling/pipeline.log 2>/dev/null; then
  exit 0
fi

exec bash scripts/experiments/run_capacity_scaling.sh
