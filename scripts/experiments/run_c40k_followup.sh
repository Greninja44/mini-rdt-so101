#!/usr/bin/env bash
# Follow-up variant C40k (see docs/research/research_log.md): runs after the Phase 5 pipeline; resumable.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONWARNINGS=ignore
PY=.venv/bin/python; CK=artifacts/corrective_train/C40k/ema_last.pt
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a artifacts/corrective_phase.log; }
while pgrep -f run_corrective_phase.sh >/dev/null || pgrep -f "audit_overfit --output artifacts/corrective_train/C40k" >/dev/null; do sleep 60; done
[ -f artifacts/corrective_train/C40k/result.json ] || { log "C40k training missing result.json - rerun training first"; exit 1; }
log "C40k evaluation"
$PY -m evaluation.closed_loop --policy tinyrdt --checkpoint $CK --skip-replay --episodes 0 2 3 4 5 --k 1 2 4 8 --max-steps 150 --output artifacts/closed_loop_v3/C40k > artifacts/closed_loop_v3/C40k_a.log 2>&1 &
$PY -m evaluation.closed_loop --policy tinyrdt --checkpoint $CK --skip-replay --episodes 6 7 8 9 11 --k 1 2 4 8 --max-steps 150 --output artifacts/closed_loop_v3/C40k > artifacts/closed_loop_v3/C40k_b.log 2>&1 &
wait
$PY -m evaluation.recovery --checkpoint $CK --output artifacts/recovery/C40k --episodes 0 2 3 4 5 > artifacts/recovery/C40k_a.log 2>&1 &
$PY -m evaluation.recovery --checkpoint $CK --output artifacts/recovery/C40k --episodes 6 7 8 9 11 > artifacts/recovery/C40k_b.log 2>&1 &
wait
log "C40k follow-up complete"
