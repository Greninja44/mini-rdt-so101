#!/usr/bin/env bash
# Physics-v2 exposure-matched data scaling (docs/research/exposure_matched_data_scaling_spec.md).
#
# Trains TRAIN10/20/40 at the derived exposure-matched budgets (the 80-demo / 80k anchor is reused unchanged), runs the
# training-scene closed-loop diagnostic, the offline evaluations, and the held-out benchmark at the pre-registered primary K=8, seed 0.
# Resumable; at most 2 simulation workers; waits on files, never on process names.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONWARNINGS=ignore; unset PYTHONPATH
PY=.venv/bin/python; S=artifacts/exposure_matched_scaling; G=artifacts/physics_v2_generalization
DS=artifacts/pickcube_physics_v2_rgb160; TEST=artifacts/pickcube_physics_v2_gen_test_rgb160; VAL=artifacts/pickcube_physics_v2_gen_val_rgb160
SPLIT=docs/research/physics_v2_generalization_split.json
mkdir -p $S/models $S/offline $S/closed_loop $S/train_scenes
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $S/pipeline.log; }
ckpt() { [ "$1" = TRAIN80_e ] && echo $G/models/TRAIN80_80k/ema_last.pt || echo $S/models/$1/ema_last.pt; }
done_flag() { [ "$1" = TRAIN80_e ] && echo $G/models/TRAIN80_80k/result.json || echo $S/models/$1/result.json; }

# ---- 1. exposure-matched training (GPU, sequential) ---------------------------------------------
train() {  # name subset steps
  local D=$S/models/$1; [ -f $D/result.json ] && return
  rm -rf $D; log "train $1 on $2 for $3 steps (E = $3 x 8 / windows = 147.87 passes)"
  $PY -m training.audit_overfit --dataset $DS --output $D --train-subset $2 --split-file $SPLIT --schedule cosine --prediction x0 --padding hold \
      --steps $3 --seed 17 --batch-size 8 --learning-rate 0.001 --device cuda --eval-interval 2000 --ema-decay 0.999 --snapshot-interval 4000 > $D.log 2>&1
  log "trained $1: $(tail -c 220 $D/result.json | tr -d '\n ')"
}
( train TRAIN10_e CLEAN10 10055; train TRAIN20_e TRAIN20 20037; train TRAIN40_e TRAIN40 40018; log "training complete" ) & TRAIN_PID=$!

# ---- 2. training-scene closed-loop diagnostic ---------------------------------------------------
# Scenes inside each model's OWN training data. Primary: the 10 CLEAN10 scenes (in every subset). Secondary: the 20 TRAIN20 scenes.
scenes() { $PY -c "import json; print(*json.load(open('$SPLIT'))['train_subsets']['$1'])"; }
cl() {  # model dataset episodes outdir [extra]
  local m=$1 ds=$2 eps="$3" out=$4; shift 4; mkdir -p $out
  until [ -f "$(done_flag $m)" ]; do sleep 60; done
  local e=$(echo $eps | tr ' ' '\n' | awk 'NR%2==1' | tr '\n' ' '); local o=$(echo $eps | tr ' ' '\n' | awk 'NR%2==0' | tr '\n' ' ')
  local args="--policy tinyrdt --checkpoint $(ckpt $m) --dataset $ds --k 8 --max-steps 150 --policy-seed 0 --skip-replay --no-media --output $out $*"
  $PY -m evaluation.closed_loop $args --episodes $e >> $out/worker_a.log 2>&1 & local pa=$!
  $PY -m evaluation.closed_loop $args --episodes $o >> $out/worker_b.log 2>&1
  wait $pa; log "closed loop $m -> $out done"
}
for m in TRAIN10_e TRAIN20_e TRAIN40_e; do
  [ "$m" = TRAIN10_e ] && SC=$(scenes CLEAN10) || SC=$(scenes TRAIN20)
  cl $m $DS "$SC" $S/train_scenes/$m
done
log "training-scene diagnostic complete (TRAIN80_e reuses $G/diagnostic_train_scenes/TRAIN80_80k)"

# ---- 3. offline evaluation (GPU) ----------------------------------------------------------------
wait $TRAIN_PID
( for m in TRAIN10_e TRAIN20_e TRAIN40_e; do
    $PY -m evaluation.offline_generalization --checkpoint $(ckpt $m) --dataset $DS --train-ids --output $S/offline/${m}_train.json >> $S/offline.log 2>&1
    $PY -m evaluation.offline_generalization --checkpoint $(ckpt $m) --dataset $VAL --output $S/offline/${m}_val.json >> $S/offline.log 2>&1
    $PY -m evaluation.offline_generalization --checkpoint $(ckpt $m) --dataset $TEST --output $S/offline/${m}_test.json >> $S/offline.log 2>&1
  done; log "offline evaluation complete" ) &

# ---- 4. held-out benchmark, primary K=8 seed 0 --------------------------------------------------
EPS=$($PY -c "from data.ml_dataset import episode_ids; print(*episode_ids('$TEST'))")
for m in TRAIN10_e TRAIN20_e TRAIN40_e; do cl $m $TEST "$EPS" $S/closed_loop/$m --workspace-margin 0.015; done
log "held-out benchmark complete (TRAIN80_e reuses $G/closed_loop/TRAIN80_80k: 28/56)"
wait
log "exposure-matched scaling pipeline complete"
