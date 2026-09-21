#!/usr/bin/env bash
# Physics-v2 controlled demonstration-density sweep (docs/research/controlled_density_sweep_spec.md).
#
# 1. Expert-collects, per condition, an 80-demonstration training set and its evaluation positions (2 simulation workers).
# 2. Trains the same ~2M TinyRDT with 3 seeds per condition at exposure-matched budgets derived from MEASURED window counts
#    (2 concurrent GPU jobs; training is bit-deterministic given a seed).
# 3. Runs the training-scene diagnostic and the primary K=8 closed-loop evaluation on each condition's own evaluation positions.
#
# Resumable; waits on files, never on process names.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONWARNINGS=ignore; unset PYTHONPATH
PY=.venv/bin/python; D=artifacts/density_sweep
CONDS="r2.5 r5.0 r7.5 r10.0 r15.0 r20.0 r7.5_onesided"
SEEDS="0 1 2"
mkdir -p $D/data $D/models $D/closed_loop $D/train_scenes $D/offline
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $D/pipeline.log; }

# ---- 1. expert collection --------------------------------------------------------------------
collect_condition() {  # condition
  local c=$1 pos=$D/positions/$c.json
  for split in train eval; do
    local root=$D/data/${c}_${split}
    local want=$($PY -c "import json;print(len(json.load(open('$pos'))['$split']))")
    local have=$(ls -d $root/episodes/episode_* 2>/dev/null | wc -l)
    local failed=$(cat $root/failed.jsonl 2>/dev/null | wc -l)
    [ $((have + failed)) -ge $want ] && continue
    $PY -m data.collect_v2 --positions $pos --split $split --output $root --dataset-version pickcube-physics-v2-density-1 \
        --count 200 >> $D/collect_${c}_${split}.log 2>&1
  done
  log "collected $c: train $(ls -d $D/data/${c}_train/episodes/episode_* 2>/dev/null | wc -l), eval $(ls -d $D/data/${c}_eval/episodes/episode_* 2>/dev/null | wc -l)"
}
( for c in r2.5 r7.5 r15.0 r7.5_onesided; do collect_condition $c; done ) & C1=$!
( for c in r5.0 r10.0 r20.0; do collect_condition $c; done ) & C2=$!
wait $C1 $C2
log "collection complete"
$PY -m evaluation.density_expert_control --output $D/expert_control.json >> $D/pipeline.log 2>&1
log "expert control written"

# ---- 2. training: 2 concurrent GPU jobs, exposure matched from measured windows ----------------
steps_for() {  # condition -> optimizer steps at 147.874 passes
  $PY -c "
import json,glob,numpy as np
w=sum(len(np.load(f)) for f in glob.glob('$D/data/$1_train/episodes/*/action.npy'))
print(round(147.8743068391867*w/8))"
}
train_one() {  # condition seed
  local c=$1 s=$2 M=$D/models/${c}_seed${s}
  [ -f $M/result.json ] && return
  local st=$(steps_for $c); rm -rf $M
  log "train ${c}_seed${s}: $st steps (147.87 passes over measured windows)"
  $PY -m training.audit_overfit --dataset $D/data/${c}_train --output $M --train-all --schedule cosine --prediction x0 --padding hold \
      --steps $st --seed $s --batch-size 8 --learning-rate 0.001 --device cuda --eval-interval 2000 --ema-decay 0.999 > $M.log 2>&1
  [ -f $M/result.json ] || log "WARNING: ${c}_seed${s} produced no result.json (see $M.log)"
}
trainer() {  # half of the matrix, so two run concurrently
  local which=$1 i=0
  for c in $CONDS; do for s in $SEEDS; do
    i=$((i + 1)); [ $((i % 2)) -eq $which ] && train_one $c $s
  done; done
  log "trainer $which complete"
}
trainer 0 & T0=$!
trainer 1 & T1=$!

# ---- 3. evaluation: training scenes + the condition's own evaluation positions -----------------
cl() {  # checkpoint dataset episodes outdir
  local ck=$1 ds=$2 eps="$3" out=$4; mkdir -p $out
  local e=$(echo $eps | tr ' ' '\n' | awk 'NR%2==1' | tr '\n' ' '); local o=$(echo $eps | tr ' ' '\n' | awk 'NR%2==0' | tr '\n' ' ')
  local args="--policy tinyrdt --checkpoint $ck --dataset $ds --k 8 --max-steps 150 --policy-seed 0 --skip-replay --no-media --output $out"
  $PY -m evaluation.closed_loop $args --episodes $e >> $out/worker_a.log 2>&1 & local pa=$!
  $PY -m evaluation.closed_loop $args --episodes $o >> $out/worker_b.log 2>&1
  wait $pa
}
evaluator() {
  for c in $CONDS; do for s in $SEEDS; do
    local M=$D/models/${c}_seed${s}
    until [ -f $M/result.json ]; do sleep 60; done
    local TR=$($PY -c "from data.ml_dataset import episode_ids; print(*episode_ids('$D/data/${c}_train')[:10])")
    local EV=$($PY -c "from data.ml_dataset import episode_ids; print(*episode_ids('$D/data/${c}_eval'))")
    cl $M/ema_last.pt $D/data/${c}_train "$TR" $D/train_scenes/${c}_seed${s}; log "train-scene diagnostic ${c}_seed${s} done"
    cl $M/ema_last.pt $D/data/${c}_eval "$EV" $D/closed_loop/${c}_seed${s}; log "held-out ${c}_seed${s} done"
    $PY -m evaluation.offline_generalization --checkpoint $M/ema_last.pt --dataset $D/data/${c}_train --train-ids --output $D/offline/${c}_seed${s}_train.json >> $D/offline.log 2>&1
    $PY -m evaluation.offline_generalization --checkpoint $M/ema_last.pt --dataset $D/data/${c}_eval --output $D/offline/${c}_seed${s}_eval.json >> $D/offline.log 2>&1
  done; done
  log "evaluation complete"
}
evaluator & E=$!
wait $T0 $T1 $E
log "density sweep pipeline complete"
