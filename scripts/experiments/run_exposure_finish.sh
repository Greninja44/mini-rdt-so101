#!/usr/bin/env bash
# Waits for the exposure-matched scaling pipeline, then runs the secondary sampler seeds on discordant positions
# (pre-registered) and produces the analysis, figures and digest. Detached, so results exist without an interactive session.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONWARNINGS=ignore; unset PYTHONPATH
PY=.venv/bin/python; S=artifacts/exposure_matched_scaling; G=artifacts/physics_v2_generalization
TEST=artifacts/pickcube_physics_v2_gen_test_rgb160
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $S/pipeline.log; }
until grep -q "exposure-matched scaling pipeline complete" $S/pipeline.log; do sleep 120; done
$PY -m evaluation.scaling_analysis > $S/analysis_stdout.json 2> $S/analysis.log || log "analysis failed, see analysis.log"

# Pre-registered secondary: sampler seeds 1 and 2 on DISCORDANT positions (the four models disagree at seed 0).
DISC=$($PY - <<'PYEOF'
import json
s = json.load(open("artifacts/exposure_matched_scaling/summary.json"))
by = {}
for r in s["rows"]: by.setdefault(r["episode"], []).append(r["success"])
print(*[e for e, v in sorted(by.items()) if len(set(v)) > 1])
PYEOF
)
log "discordant positions: $(echo $DISC | wc -w)"
for m in TRAIN10_e TRAIN20_e TRAIN40_e; do
  for seed in 1 2; do
    out=$S/closed_loop/$m; e=$(echo $DISC | tr ' ' '\n' | awk 'NR%2==1' | tr '\n' ' '); o=$(echo $DISC | tr ' ' '\n' | awk 'NR%2==0' | tr '\n' ' ')
    args="--policy tinyrdt --checkpoint $S/models/$m/ema_last.pt --dataset $TEST --k 8 --max-steps 150 --policy-seed $seed --workspace-margin 0.015 --skip-replay --no-media --output $out --tag _s$seed"
    [ -n "$e" ] && $PY -m evaluation.closed_loop $args --episodes $e >> $out/seeds_a.log 2>&1 & pa=$!
    [ -n "$o" ] && $PY -m evaluation.closed_loop $args --episodes $o >> $out/seeds_b.log 2>&1
    wait $pa; log "secondary seed $seed for $m on discordant positions done"
  done
done
$PY - > $S/RESULTS_DIGEST.txt 2>> $S/analysis.log <<'PYEOF'
import json, glob
S = "artifacts/exposure_matched_scaling"
s = json.load(open(f"{S}/summary.json"))
print("exposure-matched budgets:", s["budgets"], "windows:", s["exposure"])
for m, d in s["models"].items():
    h = d["heldout"]
    print(f"\n== {m} ({d['demos']} demos, {d['windows']} windows, {d['steps']} steps, E={d['exposure_passes']})")
    print("  held-out K=8 seed0: %d/%d  A %d/%d  B %d/%d  C %d/%d" % (h["overall"]["k"], h["overall"]["n"],
          h["A_interpolation"]["k"], h["A_interpolation"]["n"], h["B_sparse_interpolation"]["k"], h["B_sparse_interpolation"]["n"],
          h["C_extrapolation"]["k"], h["C_extrapolation"]["n"]))
    print("  training scenes (CLEAN10):", d["train_scenes"]["clean10_scenes"]["k"], "/", d["train_scenes"]["clean10_scenes"]["n"],
          "| own diagnostic scenes:", d["train_scenes"]["all_own_diagnostic_scenes"]["k"], "/", d["train_scenes"]["all_own_diagnostic_scenes"]["n"])
    print("  offline train/val/test:", {k: round(v, 5) for k, v in d["offline"].items() if k in ("train", "val", "test")})
    print("  nearest demo (mm):", {k: round(v, 2) if isinstance(v, float) else v for k, v in d["nearest_demo_mm"].items()})
    print("  distance bins:", {k: f"{v['k']}/{v['n']}" for k, v in d["distance_bins"].items() if v["n"]})
    print("  failures:", d["failure_totals"], "| validity:", d["validity"])
print("\npaired McNemar:", json.dumps(s["paired_mcnemar"], indent=1))
print("\ncoverage-aware logistic:", json.dumps(s["coverage_aware_logistic"], indent=1))
PYEOF
log "exposure-matched finisher complete: $S/summary.json, $S/RESULTS_DIGEST.txt, docs/assets/v2_scaling_curve.png"
