#!/usr/bin/env bash
# Physics-v2 baseline: validate dataset -> train the same 2M TinyRDT on CLEAN10 -> offline gate -> closed-loop K sweep.
# Resumable; waits on files and log markers only; <= 2 simulation workers.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONWARNINGS=ignore
PY=.venv/bin/python; DS=artifacts/pickcube_physics_v2_rgb160; D=artifacts/physics_v2/tinyrdt_clean10; EV=artifacts/physics_v2/closed_loop
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a artifacts/physics_v2/pipeline.log; }
mkdir -p artifacts/physics_v2
until [ "$(ls -d $DS/episodes/episode_* 2>/dev/null | wc -l)" -ge 100 ] || [ "$(( $(ls -d $DS/episodes/episode_* 2>/dev/null | wc -l) + $(cat $DS/failed.jsonl 2>/dev/null | wc -l) ))" -ge 100 ]; do sleep 60; done
log "dataset complete: $(ls -d $DS/episodes/episode_* | wc -l) episodes"
$PY -m data.validate_v2 $DS > artifacts/physics_v2/dataset_validation.json; log "dataset validation exit $?"
if [ ! -f $D/result.json ]; then
  rm -rf $D; log "train TinyRDT CLEAN10 (physics-v2)"
  $PY -m training.audit_overfit --dataset $DS --output $D --schedule cosine --prediction x0 --padding hold --steps 20000 --seed 17 \
      --batch-size 8 --learning-rate 0.001 --device cuda --eval-interval 2000 --ema-decay 0.999 > $D.log 2>&1
fi
[ -f $D/diagnostics_ema_last/diagnostics.json ] || $PY -m evaluation.research_audit --checkpoint $D/ema_last.pt --dataset $DS --output $D/diagnostics_ema_last --device cuda > $D/diagnostics.log 2>&1
$PY -m evaluation.gate_check --diagnostics $D/diagnostics_ema_last/diagnostics.json --gate $D/gate_definition.json > $D/gate_result.json
log "offline gate done"
log "closed-loop K sweep"
$PY -m evaluation.closed_loop --policy tinyrdt --checkpoint $D/ema_last.pt --dataset $DS --episodes 0 2 3 4 5 --k 1 2 4 8 16 --max-steps 150 --output $EV >> artifacts/physics_v2/closed_loop_a.log 2>&1 &
$PY -m evaluation.closed_loop --policy tinyrdt --checkpoint $D/ema_last.pt --dataset $DS --episodes 6 7 8 9 11 --k 1 2 4 8 16 --max-steps 150 --output $EV >> artifacts/physics_v2/closed_loop_b.log 2>&1 &
wait
log "physics-v2 pipeline complete"
