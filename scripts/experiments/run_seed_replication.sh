#!/usr/bin/env bash
# Physics-v2 training-seed replication (docs/research/training_seed_replication_spec.md).
#
# Trains the 16 missing cells of the 4 scales x 5 seeds matrix (seed 17 already exists) at the frozen exposure-matched budgets,
# and evaluates every model on the 10 shared CLEAN10 training scenes and the identical 56-position held-out benchmark (K=8, policy seed 0).
#
# Two concurrent loops: a GPU trainer and a simulation evaluator (<= 2 simulation workers). Seed-major order, so interruptions leave
# complete replicates. Resumable: trained models and finished rollouts are never recomputed.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONWARNINGS=ignore; unset PYTHONPATH
PY=.venv/bin/python; R=artifacts/seed_replication; DS=artifacts/pickcube_physics_v2_rgb160
TEST=artifacts/pickcube_physics_v2_gen_test_rgb160; VAL=artifacts/pickcube_physics_v2_gen_val_rgb160
SPLIT=docs/research/physics_v2_generalization_split.json
SEEDS="1 2 3 4"; SCALES="10 20 40 80"
mkdir -p $R/models $R/closed_loop $R/train_scenes $R/offline
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $R/pipeline.log; }
subset() { case $1 in 10) echo CLEAN10;; 20) echo TRAIN20;; 40) echo TRAIN40;; 80) echo TRAIN80;; esac; }
steps()  { case $1 in 10) echo 10055;; 20) echo 20037;; 40) echo 40018;; 80) echo 80000;; esac; }

# ---- trainer (GPU, sequential) ------------------------------------------------------------------
trainer() {
  for s in $SEEDS; do for n in $SCALES; do
    local D=$R/models/train${n}_seed${s}; [ -f $D/result.json ] && continue
    rm -rf $D; log "train train${n}_seed${s}: $(subset $n), $(steps $n) steps"
    $PY -m training.audit_overfit --dataset $DS --output $D --train-subset $(subset $n) --split-file $SPLIT \
        --schedule cosine --prediction x0 --padding hold --steps $(steps $n) --seed $s --batch-size 8 \
        --learning-rate 0.001 --device cuda --eval-interval 2000 --ema-decay 0.999 > $D.log 2>&1
    [ -f $D/result.json ] || log "WARNING: train${n}_seed${s} produced no result.json (infrastructure failure, see $D.log)"
  done; done
  log "training matrix complete"
}

# ---- evaluator (2 simulation workers) -----------------------------------------------------------
cl() {  # model dataset episodes outdir [extra]
  local ck=$1 ds=$2 eps="$3" out=$4; shift 4; mkdir -p $out
  local e=$(echo $eps | tr ' ' '\n' | awk 'NR%2==1' | tr '\n' ' '); local o=$(echo $eps | tr ' ' '\n' | awk 'NR%2==0' | tr '\n' ' ')
  local args="--policy tinyrdt --checkpoint $ck --dataset $ds --k 8 --max-steps 150 --policy-seed 0 --skip-replay --no-media --output $out $*"
  $PY -m evaluation.closed_loop $args --episodes $e >> $out/worker_a.log 2>&1 & local pa=$!
  $PY -m evaluation.closed_loop $args --episodes $o >> $out/worker_b.log 2>&1
  wait $pa
}
evaluator() {
  local C10=$($PY -c "import json; print(*json.load(open('$SPLIT'))['train_subsets']['CLEAN10'])")
  local EPS=$($PY -c "from data.ml_dataset import episode_ids; print(*episode_ids('$TEST'))")
  for s in $SEEDS; do for n in $SCALES; do
    local m=train${n}_seed${s}; local D=$R/models/$m
    until [ -f $D/result.json ]; do sleep 60; done
    cl $D/ema_last.pt $DS "$C10" $R/train_scenes/$m; log "train-scene diagnostic $m done"
    cl $D/ema_last.pt $TEST "$EPS" $R/closed_loop/$m --workspace-margin 0.015; log "held-out $m done"
    for part in train val test; do
      case $part in train) src="--dataset $DS --train-ids";; val) src="--dataset $VAL";; test) src="--dataset $TEST";; esac
      $PY -m evaluation.offline_generalization --checkpoint $D/ema_last.pt $src --output $R/offline/${m}_${part}.json >> $R/offline.log 2>&1
    done
    log "offline $m done"
  done; done
  log "evaluation matrix complete"
}

trainer & TP=$!
evaluator & EP=$!
wait $TP $EP
log "seed replication pipeline complete"
