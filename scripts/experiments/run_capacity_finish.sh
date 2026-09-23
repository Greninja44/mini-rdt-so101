#!/usr/bin/env bash
# Waits for the capacity-scaling pipeline, then runs the analysis, figures and digest. Detached.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONWARNINGS=ignore; unset PYTHONPATH
PY=.venv/bin/python; C=artifacts/capacity_scaling
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $C/pipeline.log; }
until grep -q "capacity scaling pipeline complete" $C/pipeline.log; do sleep 300; done
log "finisher: analysis + figures"
$PY -m evaluation.capacity_analysis > $C/analysis_stdout.json 2> $C/analysis.log || log "analysis failed, see analysis.log"
$PY - > $C/RESULTS_DIGEST.txt 2>> $C/analysis.log <<'PYEOF'
import json
s = json.load(open("artifacts/capacity_scaling/summary.json"))
print("missing cells:", s["missing"] or "none")
for size in ["m2.0", "m4.3", "m9.1", "m19.5"]:
    print(f"\n== {size} ({s['sizes'][size]['params']:,} params)")
    for cond in ["r10.0", "r15.0", "r20.0", "r7.5_onesided", "r7.5"]:
        c = s["cells"][f"{size}|{cond}"]
        print(f"  {cond:14s} per seed {c['per_seed']} -> {c['successes']}/{c['trials']}  train-scenes {c['train_scene_per_seed']}  offline-eval {[round(v,5) for v in c['offline_eval']]}")
    cv = s["curves"].get(size)
    if cv: print("  curve:", {k: round(v["estimate"], 1) for k, v in cv["distance_mm_at_probability"].items()}, "slope %.2f" % cv["slope_per_10mm"])
print("\ninteraction:", json.dumps(s["interaction"], indent=1))
print("\none-sided capacity:", json.dumps(s.get("one_sided_capacity"), indent=1))
print("\nsupport gap:", json.dumps(s["support_gap"], indent=1))
print("\nposition change vs 2M:", json.dumps(s["position_change_vs_2M"], indent=1))
print("\noffline vs closed loop:", json.dumps(s.get("offline_vs_closed_loop"), indent=1))
print("\nefficiency:", json.dumps(s["efficiency"], indent=1))
print("\ncost:", json.dumps(s["cost"], indent=1))
print("\nvalidity:", json.dumps(s["validity"]))
PYEOF
log "capacity finisher complete: $C/summary.json, $C/RESULTS_DIGEST.txt, docs/assets/v2_capacity_*.png"
