#!/usr/bin/env bash
# Physics-v2 spatial generalization study (docs/research/physics_v2_generalization_spec.md).
#
# Stages:
# 1. Expert collection of VAL/TEST positions (2 simulation workers), in parallel with GPU training of TRAIN80, TRAIN20, TRAIN40
#    and TRAIN80_80k.
# 2. Dataset validation and a determinism check.
# 3. Offline evaluation.
# 4. Closed-loop tiers 1-5 (2 simulation workers, disjoint halves of the test positions).
#
# Resumable: finished rollouts, evaluations and trained models are never recomputed. It waits on files and PIDs, never on process names.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONWARNINGS=ignore; unset PYTHONPATH
PY=.venv/bin/python; G=artifacts/physics_v2_generalization; DS=artifacts/pickcube_physics_v2_rgb160
SPLIT=docs/research/physics_v2_generalization_split.json
VAL=artifacts/pickcube_physics_v2_gen_val_rgb160; TEST=artifacts/pickcube_physics_v2_gen_test_rgb160
CLEAN10=artifacts/physics_v2/tinyrdt_clean10/ema_last.pt
mkdir -p $G/models $G/offline $G/closed_loop
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $G/pipeline.log; }
ckpt() { [ "$1" = CLEAN10 ] && echo $CLEAN10 || echo $G/models/$1/ema_last.pt; }

# ---- 1. expert control (collection) || training -------------------------------------------------
collect() { $PY -m data.collect_v2 --positions $SPLIT --split $1 --output $2 --dataset-version $3 --workspace-margin 0.015 --start $4 --count $5 >> $G/collect_${1}_$4.log 2>&1; }
log "collect validation + test positions with the physics-v2 expert"
( collect validation $VAL pickcube-physics-v2-gen-val-1 0 10; collect test $TEST pickcube-physics-v2-gen-test-1 0 30 ) & C1=$!
( collect test $TEST pickcube-physics-v2-gen-test-1 30 30 ) & C2=$!

train() {  # name steps subset
  local D=$G/models/$1; [ -f $D/result.json ] && return
  rm -rf $D; log "train $1 ($2 steps on $3)"
  $PY -m training.audit_overfit --dataset $DS --output $D --train-subset $3 --split-file $SPLIT --schedule cosine --prediction x0 --padding hold \
      --steps $2 --seed 17 --batch-size 8 --learning-rate 0.001 --device cuda --eval-interval 2000 --ema-decay 0.999 --snapshot-interval 4000 > $D.log 2>&1
  log "trained $1: $(tail -c 300 $D/result.json | tr -d '\n ')"
}
( train TRAIN80 20000 TRAIN80; train TRAIN20 20000 TRAIN20; train TRAIN40 20000 TRAIN40; train TRAIN80_80k 80000 TRAIN80 ) & TRAIN_PID=$!

wait $C1 $C2
log "collection done: val $(ls -d $VAL/episodes/episode_* | wc -l)/10, test $(ls -d $TEST/episodes/episode_* | wc -l)/60 (failed: $(cat $TEST/failed.jsonl 2>/dev/null | wc -l))"
$PY -m data.validate_v2 $VAL > $G/val_dataset_validation.json; log "val dataset validation exit $?"
$PY -m data.validate_v2 $TEST > $G/test_dataset_validation.json; log "test dataset validation exit $?"
$PY -m evaluation.generalization_expert_control --split $SPLIT --test $TEST --val $VAL --original $DS --output $G/expert_control.json >> $G/pipeline.log 2>&1; log "expert control summary written"
TEST_EPS=$($PY -c "from data.ml_dataset import episode_ids; print(*episode_ids('$TEST'))")
EVEN=$(echo $TEST_EPS | tr ' ' '\n' | awk 'NR%2==1' | tr '\n' ' '); ODD=$(echo $TEST_EPS | tr ' ' '\n' | awk 'NR%2==0' | tr '\n' ' ')

# ---- 3. offline evaluation (GPU, background) ---------------------------------------------------
offline() {  # model
  local ck=$(ckpt $1)
  $PY -m evaluation.offline_generalization --checkpoint $ck --dataset $DS --train-ids --output $G/offline/$1_train.json >> $G/offline.log 2>&1
  $PY -m evaluation.offline_generalization --checkpoint $ck --dataset $VAL --output $G/offline/$1_val.json >> $G/offline.log 2>&1
  $PY -m evaluation.offline_generalization --checkpoint $ck --dataset $TEST --output $G/offline/$1_test.json >> $G/offline.log 2>&1
}
until [ -f $G/models/TRAIN80/result.json ]; do sleep 60; done
( offline CLEAN10; offline TRAIN80
  for s in $G/models/TRAIN80/ema_step*.pt; do $PY -m evaluation.offline_generalization --checkpoint $s --dataset $VAL --output $G/offline/TRAIN80_val_$(basename $s .pt).json >> $G/offline.log 2>&1; done
  wait $TRAIN_PID; for m in TRAIN20 TRAIN40 TRAIN80_80k; do offline $m; done
  for s in $G/models/TRAIN80_80k/ema_step*.pt; do $PY -m evaluation.offline_generalization --checkpoint $s --dataset $VAL --output $G/offline/TRAIN80_80k_val_$(basename $s .pt).json >> $G/offline.log 2>&1; done
  log "offline evaluation complete" ) &

# ---- 4. closed loop ----------------------------------------------------------------------------
cl() {  # model k-list seed [extra args]
  local m=$1 ks=$2 s=$3; shift 3; local tag=""; [ $s -gt 0 ] && tag="_s$s"
  local out=$G/closed_loop/$m; mkdir -p $out
  until [ -f $(ckpt $m) ] && { [ $m = CLEAN10 ] || [ -f $G/models/$m/result.json ]; }; do sleep 60; done
  local args="--policy tinyrdt --checkpoint $(ckpt $m) --dataset $TEST --k $ks --max-steps 150 --policy-seed $s --workspace-margin 0.015 --output $out $*"
  [ -n "$tag" ] && args="$args --tag $tag"
  $PY -m evaluation.closed_loop $args --episodes $EVEN >> $out/worker_a.log 2>&1 &
  local pa=$!
  $PY -m evaluation.closed_loop $args --episodes $ODD >> $out/worker_b.log 2>&1
  wait $pa; log "closed loop $m K=[$ks] seed $s done"
}
log "tier 1"; cl TRAIN80 8 0; cl CLEAN10 8 0 --skip-replay; log "tier 1 complete"
first=$(echo $TEST_EPS | cut -d' ' -f1)
$PY -m evaluation.closed_loop --policy tinyrdt --checkpoint $(ckpt TRAIN80) --dataset $TEST --episodes $first --k 8 --max-steps 150 --workspace-margin 0.015 --skip-replay --no-media --output $G/determinism >> $G/determinism.log 2>&1
$PY -c "
import numpy as np; a=np.load('$G/closed_loop/TRAIN80/ep$(printf %03d $first)_k8.npz'); b=np.load('$G/determinism/ep$(printf %03d $first)_k8.npz')
print('determinism: executed identical =', a['executed'].shape==b['executed'].shape and bool((a['executed']==b['executed']).all()))" | tee -a $G/pipeline.log
# Diagnostic D (POST-HOC, added after tier 1; training scenes only, no test data): does TRAIN80 still solve the scenes it was trained on?
cl_train() {  # model
  local out=$G/diagnostic_train_scenes/$1; mkdir -p $out; local ids=$($PY -c "import json; print(*json.load(open('$SPLIT'))['train_subsets']['TRAIN20'])")
  local e=$(echo $ids | tr ' ' '\n' | awk 'NR%2==1' | tr '\n' ' '); local o=$(echo $ids | tr ' ' '\n' | awk 'NR%2==0' | tr '\n' ' ')
  local args="--policy tinyrdt --checkpoint $(ckpt $1) --dataset $DS --k 8 --max-steps 150 --skip-replay --no-media --output $out"
  $PY -m evaluation.closed_loop $args --episodes $e >> $out/worker_a.log 2>&1 & local pa=$!
  $PY -m evaluation.closed_loop $args --episodes $o >> $out/worker_b.log 2>&1; wait $pa; log "diagnostic D $1 on TRAIN20 training scenes done"
}
log "diagnostic D"; cl_train TRAIN80; cl_train TRAIN80_80k; log "diagnostic D complete"
# Execution-order change (logged; no analysis change): diagnostic D showed TRAIN80@20k solves only 7/20 of its own training scenes,
# TRAIN80@80k 18/20, so the pre-registered tier-4 80k arm runs first. Every tier still runs.
log "tier 4 (80k arm moved first)"; cl TRAIN80_80k 8 0 --skip-replay --no-media
log "tier 2"; cl TRAIN80 "1 2 4 16" 0 --skip-replay; log "tier 2 complete"
log "tier 3"; for s in 1 2; do cl TRAIN80 8 $s --skip-replay --no-media; cl CLEAN10 8 $s --skip-replay --no-media; done; log "tier 3 complete"
log "tier 4"; for m in TRAIN20 TRAIN40 TRAIN80_80k; do cl $m 8 0 --skip-replay --no-media; done; log "tier 4 complete"
log "tier 5"; cl CLEAN10 "1 2 4 16" 0 --skip-replay --no-media; log "tier 5 complete"
wait
log "generalization pipeline complete"
