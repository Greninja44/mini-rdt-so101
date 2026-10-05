#!/usr/bin/env bash
# Waits for the capacity recovery to finish, then runs the corrected analysis, figures and a plain-text digest. Detached.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONWARNINGS=ignore; unset PYTHONPATH
PY=.venv/bin/python; R=artifacts/capacity_recovery_v1
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $R/pipeline.log; }
until [ "$(ls $R/models/*/result.json 2>/dev/null | wc -l)" -ge 16 ] && grep -q "recovery evaluation complete" $R/pipeline.log; do sleep 300; done
log "finisher: corrected analysis + figures"
$PY -m evaluation.capacity_recovery_analysis > $R/analysis_stdout.json 2> $R/analysis.log || log "analysis failed, see analysis.log"
$PY - > $R/RESULTS_DIGEST.txt 2>> $R/analysis.log <<'PYEOF'
import json
s = json.load(open("artifacts/capacity_recovery_v1/summary.json"))
SIZES = ["m2.0", "m4.3", "m9.1", "m19.5"]; CONDS = ["r10.0", "r15.0", "r20.0", "r7.5_onesided", "r7.5"]
print("matrix:", json.dumps(s["matrix"]), "\nmissing:", s["missing"] or "none")
print("\n== ORIGINAL vs RECOVERED (audit-flagged runs only)")
for c in s["integrity_comparison"]:
    o, r = c["original"], c["recovered"]
    print(f"  {c['run']:28s} held-out {o['successes']}/{o['trials']} -> {r['successes']}/{r['trials']} | train MAE {o['train_mae']:.4f} -> {r['train_mae']:.4f} "
          f"| scenes {o['train_scenes'][0]}/{o['train_scenes'][1]} -> {r['train_scenes'][0]}/{r['train_scenes'][1]} | EMA changed {c['ema_checkpoint_changed']}")
chg = [c for c in s["integrity_comparison"] if c["delta_successes"] or c["delta_train_scenes"]]
print(f"  runs whose outcome changed: {len(chg)}/{len(s['integrity_comparison'])}")
print("\n== CORRECTED MATRIX")
for size in SIZES:
    print(f"\n{size} ({s['cells'][size+'|r10.0']['train_mae'] and ''}params)")
    for cond in CONDS:
        c = s["cells"][f"{size}|{cond}"]
        print(f"  {cond:14s} seeds {c['per_seed']} -> {c['successes']}/{c['trials']} ({100*(c['rate'] or 0):.1f}%)  scenes {c['train_scene_per_seed']}  trainMAE {[round(v,4) for v in c['train_mae']]}  src {set(c['sources'])}")
print("\ncurves:", json.dumps({k: {"slope": round(v["slope_per_10mm"], 2), **{p: round(q["estimate"], 1) for p, q in v["distance_mm_at_probability"].items()}} for k, v in s["curves"].items()}, indent=1))
print("\ninteraction:", json.dumps(s["interaction"], indent=1)[:1800])
print("\nsupport gap:", json.dumps({k: {"gap": round(v["gap"], 3), "surrounded": v["surrounded"]["rate"], "one_sided": v["one_sided"]["rate"]} for k, v in s["support_gap"].items()}, indent=1))
print("\nposition change vs 2M:", json.dumps(s["position_change_vs_2M"], indent=1))
print("\noffline vs closed loop:", json.dumps(s.get("offline_vs_closed_loop"), indent=1))
print("\ncost:", json.dumps(s["cost"], indent=1))
print("\nmechanism (lateral at close):", json.dumps({k: v["lateral_at_close_mm"] for k, v in s["mechanism"].items()}, indent=1))
print("\nvalidity:", json.dumps(s["validity"]))
PYEOF
log "recovery finisher complete: $R/summary.json, $R/RESULTS_DIGEST.txt, docs/assets/v2_capacity_corrected_*.png"
