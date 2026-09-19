#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore
PY=.venv/bin/python; D=artifacts/phase9/vs10_60k
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a artifacts/phase9.log; }
if [ ! -f $D/result.json ]; then
  rm -rf $D; log "9b train VS 60k"
  $PY -m training.audit_overfit --dataset artifacts/varied_start10 --train-all --output $D --schedule cosine --prediction x0 --padding hold \
      --steps 60000 --seed 17 --batch-size 8 --learning-rate 0.001 --device cuda --eval-interval 6000 --ema-decay 0.999 > $D.log 2>&1
fi
[ -f $D/diagnostics_ema_last/diagnostics.json ] || $PY -m evaluation.research_audit --checkpoint $D/ema_last.pt --dataset artifacts/varied_start10 --output $D/diagnostics_ema_last --device cuda > $D/diagnostics.log 2>&1
W=$($PY -c "import json;r=json.load(open('$D/diagnostics_ema_last/diagnostics.json'))['full_training_windows'][0]['all'];print(max(r['per_joint_mae'][:5]))")
log "9b offline worst joint $W"
if ! $PY -c "import sys; sys.exit(0 if $W <= 0.02 else 1)"; then log "9b precision criterion not met - stopping"; log "phase 9b complete"; exit 0; fi
log "9b closed-loop HOME start"
for eps in "0 2 3 4 5" "6 7 8 9 11"; do
  $PY -m evaluation.closed_loop --policy tinyrdt --checkpoint $D/ema_last.pt --skip-replay --episodes $eps --k 1 2 4 8 --max-steps 150 --output artifacts/closed_loop_v3/VS60k >> artifacts/phase9/home60k.log 2>&1 &
done; wait
log "9b closed-loop unseen random starts"
$PY -m evaluation.closed_loop --policy tinyrdt --checkpoint $D/ema_last.pt --start-seed 777 --k 4 8 --max-steps 150 --episodes 0 2 3 4 5 --output artifacts/closed_loop_v3/VS60k_start777 >> artifacts/phase9/start60k.log 2>&1 &
$PY -m evaluation.closed_loop --policy tinyrdt --checkpoint $D/ema_last.pt --start-seed 777 --k 4 8 --max-steps 150 --episodes 6 7 8 9 11 --output artifacts/closed_loop_v3/VS60k_start777 >> artifacts/phase9/start60k.log 2>&1 &
wait
log "phase 9b complete"
