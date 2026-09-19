#!/usr/bin/env bash
# Phase 8 density diagnostic (pre-registered). Resumable; <=2 sim workers.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONWARNINGS=ignore
PY=.venv/bin/python; D=artifacts/phase8/train80
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a artifacts/phase8.log; }
mkdir -p artifacts/phase8
if [ ! -f $D/result.json ]; then
  rm -rf $D; log "8 train on 80 training episodes"
  $PY -m training.audit_overfit --output $D --train-episodes 80 --schedule cosine --prediction x0 --padding hold --steps 40000 --seed 17 \
      --batch-size 8 --learning-rate 0.001 --device cuda --eval-interval 4000 --ema-decay 0.999 > $D.log 2>&1
fi
VAL=$($PY -c "from data.ml_dataset import make_episode_splits as m;print(' '.join(map(str,m('artifacts/pickcube_smoke100_rgb160',17).validation)))")
log "8 validation ids: $VAL"
set -- $VAL
log "8 closed-loop memorised CLEAN10 seeds"
for eps in "0 2 3 4 5" "6 7 8 9 11"; do
  $PY -m evaluation.closed_loop --policy tinyrdt --checkpoint $D/ema_last.pt --skip-replay --episodes $eps --k 1 2 4 8 --max-steps 150 --output artifacts/closed_loop_v3/D80_memorised >> artifacts/phase8/memorised.log 2>&1 &
done; wait
log "8 closed-loop held-out validation seeds (with expert replay control)"
$PY -m evaluation.closed_loop --policy tinyrdt --checkpoint $D/ema_last.pt --episodes $1 $2 $3 $4 $5 --k 1 2 4 8 --max-steps 150 --output artifacts/closed_loop_v3/D80_validation >> artifacts/phase8/validation.log 2>&1 &
$PY -m evaluation.closed_loop --policy tinyrdt --checkpoint $D/ema_last.pt --episodes $6 $7 $8 $9 ${10} --k 1 2 4 8 --max-steps 150 --output artifacts/closed_loop_v3/D80_validation >> artifacts/phase8/validation.log 2>&1 &
wait
log "phase 8 complete"
