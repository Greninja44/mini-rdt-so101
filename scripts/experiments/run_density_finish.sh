#!/usr/bin/env bash
# Waits for the density sweep, then runs the analysis, figures and digest. Detached.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONWARNINGS=ignore; unset PYTHONPATH
PY=.venv/bin/python; D=artifacts/density_sweep
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $D/pipeline.log; }
until grep -q "density sweep pipeline complete" $D/pipeline.log; do sleep 180; done
log "finisher: analysis + figures"
$PY -m evaluation.density_analysis > $D/analysis_stdout.json 2> $D/analysis.log || log "analysis failed, see analysis.log"
$PY - > $D/RESULTS_DIGEST.txt 2>> $D/analysis.log <<'PYEOF'
import json
s = json.load(open("artifacts/density_sweep/summary.json"))
for name, c in s["conditions"].items():
    print(f"\n== {name} (nominal {c['nominal_radius_mm']:.1f} mm, {c['geometry']})")
    print(f"  measured d1: {c['measured_d1_mm']['median']:.2f} mm | eval positions {c['evaluation_positions']} | demos {c['demonstrations']} | excluded {c['expert_excluded']}")
    print(f"  per seed {c['per_seed_successes']} -> {c['successes']}/{c['trials']} = {100*c['rate']:.1f}%  Wilson {[round(100*v,1) for v in c['wilson95_naive']]}")
    print(f"  training scenes {c['train_scene_success_per_seed']} of {c['train_scene_trials_per_seed']} | offline {c['offline_mae']}")
    print(f"  failures {c['failures']} | close lateral {c['close_lateral_mm']} | validity {c['validity']}")
print("\ncontrolled curve:", json.dumps(s["controlled_curve"], indent=1))
print("\nobservational curve:", json.dumps(s["observational_curve"]["distance_mm_at_probability"], indent=1))
print("\ntwo-way logistic:", json.dumps(s["controlled_logistic_two_way"], indent=1))
print("\nsupport geometry:", json.dumps(s.get("support_geometry"), indent=1))
print("\nvariance decomposition:", json.dumps(s["variance_decomposition"], indent=1))
print("\npredictors:", json.dumps({k: {q: {kk: vv for kk, vv in v[q].items() if kk not in ('n_rollouts','n_positions','n_seeds')} for q in v} for k, v in s["predictors"].items()}, indent=1)[:3000])
PYEOF
log "density finisher complete: $D/summary.json, $D/RESULTS_DIGEST.txt, docs/assets/v2_density_*.png"
