#!/usr/bin/env bash
# Capacity-density infrastructure recovery (docs/research/capacity_density_recovery_spec.md).
#
# Retrains, from the original seed, every run the committed integrity audit flags as resumed or hash-mismatched - selected by the audit,
# never by outcome - and re-evaluates it. Originals are preserved; all new output lands under artifacts/capacity_recovery_v1/.
#
# One training worker and one evaluation worker (two total, per protocol). Resumable; waits on files, never on process names.
# The old launcher and watchdog stay disabled.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONWARNINGS=ignore; unset PYTHONPATH
PY=.venv/bin/python; R=artifacts/capacity_recovery_v1; DS=artifacts/density_sweep/data; STEPS=80222
mkdir -p $R/models $R/closed_loop $R/train_scenes $R/offline
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $R/pipeline.log; }
arch() { case $1 in m4.3) echo "256 5 8";; m9.1) echo "320 7 10";; m19.5) echo "416 9 13";; esac; }

# Selection comes from the committed audit, so it cannot depend on results.
mapfile -t RUNS < <($PY -c "
import json
a = json.load(open('docs/research/capacity_integrity_audit.json'))['runs']
sel = [k for k, v in a.items() if v.get('resume') or not v['closed_loop']['checkpoint_hashes_match'] or not v['train_scenes']['checkpoint_hashes_match']]
print('\n'.join(sorted(sel)))")
log "recovery selection: ${#RUNS[@]} runs -> ${RUNS[*]}"

trainer() {
  for run in "${RUNS[@]}"; do
    local M=$R/models/$run; [ -f $M/result.json ] && continue
    local size=${run%%_*} rest=${run#*_}; local cond=${rest%_seed*} seed=${run##*_seed}
    set -- $(arch $size); rm -rf $M
    log "train $run from scratch: d=$1 L=$2 heads=$3, $STEPS steps, seed $seed"
    $PY -m training.audit_overfit --dataset $DS/${cond}_train --output $M --train-all --schedule cosine --prediction x0 --padding hold \
        --steps $STEPS --seed $seed --batch-size 8 --learning-rate 0.001 --device cuda --eval-interval 2000 --ema-decay 0.999 \
        --hidden-dim $1 --layers $2 --heads $3 > $M.log 2>&1
    if [ -f $M/result.json ]; then
      log "trained $run: $($PY -c "import json;r=json.load(open('$M/result.json'));print('params',r['policy_parameters'],'wall %.0f min'%(r['elapsed_s']/60),'vram %.0f MB'%(r['peak_vram_bytes']/1e6),'loss %.4f'%r['loss'])")"
    else
      log "WARNING: $run produced no result.json (see $M.log)"
    fi
  done
  log "recovery training complete"
}

evaluator() {
  for run in "${RUNS[@]}"; do
    local M=$R/models/$run; local rest=${run#*_}; local cond=${rest%_seed*}
    until [ -f $M/result.json ] || grep -q "WARNING: $run " $R/pipeline.log 2>/dev/null; do sleep 60; done
    [ -f $M/result.json ] || continue
    local TR=$($PY -c "from data.ml_dataset import episode_ids; print(*episode_ids('$DS/${cond}_train')[:10])")
    local EV=$($PY -c "from data.ml_dataset import episode_ids; print(*episode_ids('$DS/${cond}_eval'))")
    local args="--policy tinyrdt --checkpoint $M/ema_last.pt --k 8 --max-steps 150 --policy-seed 0 --skip-replay --no-media"
    mkdir -p $R/train_scenes/$run $R/closed_loop/$run
    $PY -m evaluation.closed_loop $args --dataset $DS/${cond}_train --episodes $TR --output $R/train_scenes/$run >> $R/train_scenes/$run/worker.log 2>&1
    $PY -m evaluation.closed_loop $args --dataset $DS/${cond}_eval --episodes $EV --output $R/closed_loop/$run >> $R/closed_loop/$run/worker.log 2>&1
    $PY -m evaluation.offline_generalization --checkpoint $M/ema_last.pt --dataset $DS/${cond}_train --train-ids --output $R/offline/${run}_train.json >> $R/offline.log 2>&1
    $PY -m evaluation.offline_generalization --checkpoint $M/ema_last.pt --dataset $DS/${cond}_eval --output $R/offline/${run}_eval.json >> $R/offline.log 2>&1
    log "evaluated $run"
  done
  log "recovery evaluation complete"
}

trainer & T=$!
evaluator & E=$!
wait $T $E
$PY -m scripts.capacity_integrity_audit --root $R --output docs/research/capacity_recovery_integrity_audit.json >> $R/pipeline.log 2>&1
log "recovery integrity audit written"
log "capacity recovery pipeline complete"
