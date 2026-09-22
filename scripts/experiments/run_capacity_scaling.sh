#!/usr/bin/env bash
# Physics-v2 targeted capacity x density scaling (docs/research/capacity_density_scaling_spec.md).
#
# Trains the 4.3M / 9.1M / 19.5M members of the TinyRDT scaling family (head dim 32, MLP 4d, layers = round(d/48)) on the frozen PR #9
# density datasets, at matched data exposure (80,222 steps, batch 8), 3 seeds each; the 2M baseline runs are reused unchanged.
# Every model gets the training-scene diagnostic, the K=8 evaluation on its condition's positions, and offline metrics.
# Two concurrent GPU trainers, one evaluator using <= 2 simulation workers. Resumable; waits on files, never on process names.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONWARNINGS=ignore; unset PYTHONPATH
PY=.venv/bin/python; C=artifacts/capacity_scaling; DS=artifacts/density_sweep/data
SIZES="m4.3 m9.1 m19.5"; CONDS="r10.0 r15.0 r20.0 r7.5_onesided r7.5"; SEEDS="0 1 2"; STEPS=80222
mkdir -p $C/models $C/closed_loop $C/train_scenes $C/offline
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $C/pipeline.log; }
arch() { case $1 in m4.3) echo "256 5 8";; m9.1) echo "320 7 10";; m19.5) echo "416 9 13";; esac; }
QUEUE=(); for z in $SIZES; do for c in $CONDS; do for s in $SEEDS; do QUEUE+=("$z $c $s"); done; done; done

train_one() {  # size condition seed
  local z=$1 c=$2 s=$3 M=$C/models/${1}_${2}_seed${3}; [ -f $M/result.json ] && return
  set -- $(arch $z); rm -rf $M; log "train ${z}_${c}_seed${s}: d=$1 L=$2 heads=$3, $STEPS steps"
  $PY -m training.audit_overfit --dataset $DS/${c}_train --output $M --train-all --schedule cosine --prediction x0 --padding hold \
      --steps $STEPS --seed $s --batch-size 8 --learning-rate 0.001 --device cuda --eval-interval 2000 --ema-decay 0.999 \
      --hidden-dim $1 --layers $2 --heads $3 > $M.log 2>&1
  [ -f $M/result.json ] && log "trained ${z}_${c}_seed${s}: $(python3 -c "import json;r=json.load(open('$M/result.json'));print('params',r['policy_parameters'],'wall %.0f min'%(r['elapsed_s']/60),'vram %.0f MB'%(r['peak_vram_bytes']/1e6),'loss %.4f'%r['loss'])")" \
                         || log "WARNING: ${z}_${c}_seed${s} produced no result.json (see $M.log)"
}
trainer() { local i=0; for item in "${QUEUE[@]}"; do i=$((i+1)); [ $((i % 2)) -eq $1 ] && train_one $item; done; log "trainer $1 complete"; }
trainer 0 & T0=$!
trainer 1 & T1=$!

cl() {  # checkpoint dataset episodes outdir
  local ck=$1 ds=$2 eps="$3" out=$4; mkdir -p $out
  local e=$(echo $eps | tr ' ' '\n' | awk 'NR%2==1' | tr '\n' ' '); local o=$(echo $eps | tr ' ' '\n' | awk 'NR%2==0' | tr '\n' ' ')
  local args="--policy tinyrdt --checkpoint $ck --dataset $ds --k 8 --max-steps 150 --policy-seed 0 --skip-replay --no-media --output $out"
  $PY -m evaluation.closed_loop $args --episodes $e >> $out/worker_a.log 2>&1 & local pa=$!
  [ -n "$o" ] && $PY -m evaluation.closed_loop $args --episodes $o >> $out/worker_b.log 2>&1
  wait $pa
}
evaluator() {
  for item in "${QUEUE[@]}"; do
    set -- $item; local z=$1 c=$2 s=$3 m=${1}_${2}_seed${3}; local M=$C/models/$m
    until [ -f $M/result.json ] || grep -q "WARNING: $m " $C/pipeline.log 2>/dev/null; do sleep 60; done
    [ -f $M/result.json ] || continue
    local TR=$($PY -c "from data.ml_dataset import episode_ids; print(*episode_ids('$DS/${c}_train')[:10])")
    local EV=$($PY -c "from data.ml_dataset import episode_ids; print(*episode_ids('$DS/${c}_eval'))")
    cl $M/ema_last.pt $DS/${c}_train "$TR" $C/train_scenes/$m
    cl $M/ema_last.pt $DS/${c}_eval "$EV" $C/closed_loop/$m; log "evaluated $m"
    $PY -m evaluation.offline_generalization --checkpoint $M/ema_last.pt --dataset $DS/${c}_train --train-ids --output $C/offline/${m}_train.json >> $C/offline.log 2>&1
    $PY -m evaluation.offline_generalization --checkpoint $M/ema_last.pt --dataset $DS/${c}_eval --output $C/offline/${m}_eval.json >> $C/offline.log 2>&1
  done
  log "evaluation complete"
}
evaluator & E=$!
wait $T0 $T1 $E
$PY -m evaluation.latency_benchmark --output $C/latency.json >> $C/pipeline.log 2>&1; log "latency benchmark written"
log "capacity scaling pipeline complete"
