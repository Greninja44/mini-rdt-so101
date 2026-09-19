#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONWARNINGS=ignore
PY=.venv/bin/python; D=artifacts/phase9/vs10; A=artifacts/research_audit/cosine_x0_hold_ema/ema_last.pt
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a artifacts/phase9.log; }
mkdir -p artifacts/phase9
log "9 collect VS10"
$PY -m data.collect_varied_start --episodes 0 2 3 4 5 >> artifacts/phase9/collect.log 2>&1 &
$PY -m data.collect_varied_start --episodes 6 7 8 9 11 >> artifacts/phase9/collect.log 2>&1 &
wait
log "9 VS10 episodes: $(ls artifacts/varied_start10/episodes | wc -l); failed: $(cat artifacts/varied_start10/failed.jsonl 2>/dev/null | wc -l)"
if [ ! -f $D/result.json ]; then
  rm -rf $D; log "9 train VS"
  $PY -m training.audit_overfit --dataset artifacts/varied_start10 --train-all --output $D --schedule cosine --prediction x0 --padding hold \
      --steps 20000 --seed 17 --batch-size 8 --learning-rate 0.001 --device cuda --eval-interval 2000 --ema-decay 0.999 > $D.log 2>&1
fi
log "9 closed-loop HOME start"
for eps in "0 2 3 4 5" "6 7 8 9 11"; do
  $PY -m evaluation.closed_loop --policy tinyrdt --checkpoint $D/ema_last.pt --skip-replay --episodes $eps --k 1 2 4 8 --max-steps 150 --output artifacts/closed_loop_v3/VS >> artifacts/phase9/home.log 2>&1 &
done; wait
log "9 closed-loop unseen random starts"
$PY -m evaluation.closed_loop --policy tinyrdt --checkpoint $D/ema_last.pt --start-seed 777 --k 4 8 --max-steps 150 --episodes 0 2 3 4 5 6 7 8 9 11 --output artifacts/closed_loop_v3/VS_start777 >> artifacts/phase9/start.log 2>&1 &
$PY -m evaluation.closed_loop --policy tinyrdt --checkpoint $A --start-seed 777 --k 4 8 --max-steps 150 --episodes 0 2 3 4 5 6 7 8 9 11 --output artifacts/closed_loop_v3/A_start777 >> artifacts/phase9/start.log 2>&1 &
wait
log "phase 9 complete"
