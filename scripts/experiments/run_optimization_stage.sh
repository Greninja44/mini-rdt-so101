#!/usr/bin/env bash
# Optimization scaling study (docs/research/optimization_scaling_spec.md): run one pre-registered stage.
#
# Usage: run_optimization_stage.sh <manifest.json>
# The manifest lists runs: {name, hidden, layers, heads, condition, seed, steps, extra, eval_train_scenes, eval_heldout}.
# Trains with the existing trainer (only optimizer flags vary), then evaluates training scenes and, when asked, held-out positions.
# One training worker and one evaluation worker. Resumable: finished runs and rollouts are never recomputed.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONWARNINGS=ignore; unset PYTHONPATH
PY=.venv/bin/python; O=artifacts/optimization_scaling_v1; DS=artifacts/density_sweep/data
MANIFEST=${1:?manifest required}
mkdir -p $O/models $O/train_scenes $O/closed_loop $O/offline
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $O/pipeline.log; }
field() { $PY -c "import json,sys;print(json.load(open('$MANIFEST'))[$1]['$2'])"; }
count=$($PY -c "import json;print(len(json.load(open('$MANIFEST'))))")
log "stage manifest $MANIFEST: $count runs"

train_one() {
  local i=$1 name=$(field $1 name) M
  M=$O/models/$name; [ -f $M/result.json ] && return
  local hid=$(field $i hidden) lay=$(field $i layers) hea=$(field $i heads) cond=$(field $i condition) seed=$(field $i seed) steps=$(field $i steps) extra=$(field $i extra)
  local resume=""
  if [ -f $M/last.pt ] && $PY -c "
import sys, torch
ck = torch.load('$M/last.pt', map_location='cpu', weights_only=False)
sys.exit(0 if 'ema_model' in ck and ck.get('ema_decay') == 0.999 else 1)" 2>/dev/null; then
    resume="--resume $M/last.pt"; log "resume $name from its atomic checkpoint"
  else
    rm -rf $M; log "train $name: d=$hid L=$lay h=$hea on $cond seed $seed, $steps steps, extra: $extra"
  fi
  $PY -m training.audit_overfit --dataset $DS/${cond}_train --output $M --train-all --schedule cosine --prediction x0 --padding hold \
      --steps $steps --seed $seed --batch-size 8 --learning-rate 0.001 --device cuda --eval-interval 2000 --ema-decay 0.999 \
      --hidden-dim $hid --layers $lay --heads $hea $extra $resume > $M.log 2>&1
  [ -f $M/result.json ] && log "trained $name: $($PY -c "import json;r=json.load(open('$M/result.json'));print('wall %.0f min'%(r['elapsed_s']/60),'loss %.4f'%r['loss'],'trainMAE %.5f'%r['sampled']['action_mae'])")" \
                        || log "WARNING: $name produced no result.json (see $M.log)"
}

evaluate_one() {
  local i=$1 name=$(field $1 name) cond=$(field $1 condition) M=$O/models/$(field $1 name)
  until [ -f $M/result.json ] || grep -q "WARNING: $name " $O/pipeline.log 2>/dev/null; do sleep 60; done
  [ -f $M/result.json ] || return
  local args="--policy tinyrdt --checkpoint $M/ema_last.pt --k 8 --max-steps 150 --policy-seed 0 --skip-replay --no-media"
  if [ "$(field $i eval_train_scenes)" = "True" ]; then
    local TR=$($PY -c "from data.ml_dataset import episode_ids; print(*episode_ids('$DS/${cond}_train')[:10])")
    mkdir -p $O/train_scenes/$name
    $PY -m evaluation.closed_loop $args --dataset $DS/${cond}_train --episodes $TR --output $O/train_scenes/$name >> $O/train_scenes/$name/worker.log 2>&1
    $PY -m evaluation.offline_generalization --checkpoint $M/ema_last.pt --dataset $DS/${cond}_train --train-ids --output $O/offline/${name}_train.json >> $O/offline.log 2>&1
    log "train-scene diagnostic $name: $(ls $O/train_scenes/$name/ep*_k8.json 2>/dev/null | wc -l) rollouts"
  fi
  if [ "$(field $i eval_heldout)" = "True" ]; then
    local EV=$($PY -c "from data.ml_dataset import episode_ids; print(*episode_ids('$DS/${cond}_eval'))")
    mkdir -p $O/closed_loop/$name
    $PY -m evaluation.closed_loop $args --dataset $DS/${cond}_eval --episodes $EV --output $O/closed_loop/$name >> $O/closed_loop/$name/worker.log 2>&1
    $PY -m evaluation.offline_generalization --checkpoint $M/ema_last.pt --dataset $DS/${cond}_eval --output $O/offline/${name}_eval.json >> $O/offline.log 2>&1
    log "held-out evaluation $name done"
  fi
}

( for i in $(seq 0 $((count - 1))); do train_one $i; done; log "stage training complete" ) & T=$!
( for i in $(seq 0 $((count - 1))); do evaluate_one $i; done; log "stage evaluation complete" ) & E=$!
wait $T $E
log "stage complete: $MANIFEST"
