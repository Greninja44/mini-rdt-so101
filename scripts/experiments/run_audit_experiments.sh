#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for setting in cosine_x0 linear_epsilon cosine_epsilon; do
  schedule="${setting%_*}"
  prediction="${setting#*_}"
  .venv/bin/python -m training.audit_overfit --output "artifacts/research_audit/${setting}" --schedule "$schedule" --prediction "$prediction" --device cuda > "artifacts/research_audit/${setting}.log" 2>&1
  .venv/bin/python -m evaluation.research_audit --checkpoint "artifacts/research_audit/${setting}/last.pt" --output "artifacts/research_audit/${setting}/diagnostics" --device cuda > "artifacts/research_audit/${setting}_diagnostics.log" 2>&1
done
for baseline in privileged rgb state; do
  .venv/bin/python -m training.audit_overfit --output "artifacts/research_audit/bc_${baseline}" --baseline "$baseline" --device cuda > "artifacts/research_audit/bc_${baseline}.log" 2>&1
done
