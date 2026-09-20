#!/usr/bin/env bash
# POST-HOC extension (not pre-registered; decided after tier 1 + diagnostic D, before the pre-registered tiers finished).
# The 80k checkpoint is the only TRAIN80 model that solves its own training scenes (18/20), so its sampler-seed
# repeatability and K sensitivity on the SAME frozen test positions are measured. No position, model or threshold changes.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONWARNINGS=ignore; unset PYTHONPATH
PY=.venv/bin/python; G=artifacts/physics_v2_generalization; TEST=artifacts/pickcube_physics_v2_gen_test_rgb160
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $G/pipeline.log; }
until grep -q "generalization pipeline complete" $G/pipeline.log; do sleep 120; done
EPS=$($PY -c "from data.ml_dataset import episode_ids; print(*episode_ids('$TEST'))")
EVEN=$(echo $EPS | tr ' ' '\n' | awk 'NR%2==1' | tr '\n' ' '); ODD=$(echo $EPS | tr ' ' '\n' | awk 'NR%2==0' | tr '\n' ' ')
run() {  # k-list seed
  local tag=""; [ $2 -gt 0 ] && tag="--tag _s$2"
  local args="--policy tinyrdt --checkpoint $G/models/TRAIN80_80k/ema_last.pt --dataset $TEST --k $1 --max-steps 150 --policy-seed $2 --workspace-margin 0.015 --skip-replay --no-media --output $G/closed_loop/TRAIN80_80k $tag"
  $PY -m evaluation.closed_loop $args --episodes $EVEN >> $G/closed_loop/TRAIN80_80k/posthoc_a.log 2>&1 & local pa=$!
  $PY -m evaluation.closed_loop $args --episodes $ODD >> $G/closed_loop/TRAIN80_80k/posthoc_b.log 2>&1; wait $pa; log "post-hoc TRAIN80_80k K=[$1] seed $2 done"
}
log "post-hoc P1: TRAIN80_80k K=8 seeds 1,2"; run 8 1; run 8 2
log "post-hoc P2: TRAIN80_80k K sweep seed 0"; run "1 2 4 16" 0
log "post-hoc complete"
