#!/usr/bin/env bash
# Waits for the seed-replication matrix, then runs the analysis, figures and a plain-text digest. Detached.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONWARNINGS=ignore; unset PYTHONPATH
PY=.venv/bin/python; R=artifacts/seed_replication
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $R/pipeline.log; }
until grep -q "seed replication pipeline complete" $R/pipeline.log; do sleep 180; done
log "finisher: analysis + figures"
$PY -m evaluation.seed_replication_analysis > $R/analysis_stdout.json 2> $R/analysis.log || log "analysis failed, see analysis.log"
$PY - > $R/RESULTS_DIGEST.txt 2>> $R/analysis.log <<'PYEOF'
import json
s = json.load(open("artifacts/seed_replication/summary.json"))
print("seeds:", s["seeds"], "| missing cells:", s["missing_cells"] or "none")
for scale, d in s["by_scale"].items():
    print(f"\n== TRAIN{scale} ({d['seeds_present']} seeds)")
    print("  held-out per seed:", d["per_seed"], "-> mean %.1f/56  SD %s  range %d-%d  boot95 [%.1f, %.1f]" % (
        d["mean"], ("%.2f" % d["sd"]) if d["sd"] is not None else "n/a", d["min"], d["max"], *d["bootstrap95_mean"]))
    for c, v in d["per_seed_by_category"].items():
        print(f"  {c[0]}: {v} mean {d['category_mean'][c]:.1f} SD {d['category_sd'][c]}")
    print("  training scenes per seed:", d["train_scenes_per_seed"], "range", d["train_scenes_range"])
print("\nvariance decomposition:", json.dumps(s["variance_decomposition"], indent=1))
print("\nposition frequency histogram (successes across seeds):", json.dumps(s["position_frequency_histogram"], indent=1))
print("\ncoverage model:", json.dumps(s["coverage_model"], indent=1))
print("\npaired scale tests:", json.dumps(s["paired_scale_tests"], indent=1))
print("\ndistance bins:", json.dumps(s["distance_bins"], indent=1))
print("\ngeneralization curve:", json.dumps(s["generalization_curve"], indent=1))
print("\npredictors:", json.dumps(s.get("predictors_of_heldout_success"), indent=1))
print("\nextrapolation:", json.dumps(s["extrapolation"], indent=1))
print("\nfailure modes by scale:", json.dumps(s["failure_modes_by_scale"], indent=1))
print("same failure mode across seeds:", json.dumps(s["failures_same_mode_across_seeds"]))
print("\nTRAIN40 category B matrix:", json.dumps(s["train40_category_B_matrix"], indent=1)[:2000])
PYEOF
log "seed finisher complete: $R/summary.json, $R/RESULTS_DIGEST.txt, docs/assets/v2_seed_*.png"
