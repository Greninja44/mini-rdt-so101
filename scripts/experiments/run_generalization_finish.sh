#!/usr/bin/env bash
# Waits for the pre-registered pipeline AND the post-hoc extension, then produces every derived artifact:
# summary.json, figures, representative media, and a plain-text digest. Runs detached, so results exist on disk
# even if no interactive session is attached.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONWARNINGS=ignore; unset PYTHONPATH
PY=.venv/bin/python; G=artifacts/physics_v2_generalization
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $G/pipeline.log; }
until grep -q "generalization pipeline complete" $G/pipeline.log && grep -q "post-hoc complete" $G/pipeline.log; do sleep 180; done
log "finisher: analysis + figures + media"
$PY -m evaluation.generalization_analysis --media > $G/analysis_stdout.json 2> $G/analysis.log
$PY -m evaluation.generalization_probe >> $G/analysis.log 2>&1
$PY - <<'PYEOF' > $G/RESULTS_DIGEST.txt 2>> $G/analysis.log
import json, glob
G = "artifacts/physics_v2_generalization"
s = json.load(open(f"{G}/summary.json")); sp = json.load(open("docs/research/physics_v2_generalization_split.json"))
cat = {r["index"]: r["category"][0] for r in sp["test"]}
print("benchmark positions:", s["test_positions_in_benchmark"], "outside:", s["outside_benchmark"])
print("expert replay:", s["expert_replay"]["k"], "/", s["expert_replay"]["n"])
for m, d in s["closed_loop"].items():
    print(f"\n== {m}")
    for k, v in d["by_k_seed0"].items(): print(f"  K={k:<2} {v['k']}/{v['n']}  " + " ".join(f"{c[0]}:{d['by_k_seed0_by_category'][k][c]['k']}/{d['by_k_seed0_by_category'][k][c]['n']}" for c in ("A_interpolation","B_sparse_interpolation","C_extrapolation")))
    if "k8_by_seed" in d: print("  K=8 by sampler seed:", {k: f"{v['k']}/{v['n']}" for k, v in d["k8_by_seed"].items()})
    print("  outcomes K=8:", d["outcomes_k8_seed0_by_category"])
    print("  validity:", d["validity"])
print("\npaired TRAIN80 vs CLEAN10 (K=8, seed 0):", s.get("paired_TRAIN80_vs_CLEAN10_k8_seed0"))
print("\noffline:", {m: {p: round(o[p]["action_mae"], 4) for p in ("train","val","test") if p in o} for m, o in s["offline"].items()})
for m in ("TRAIN80","TRAIN80_80k"):
    r = [json.load(open(f)) for f in glob.glob(f"{G}/diagnostic_train_scenes/{m}/ep*_k8.json")]
    if r: print(f"diagnostic D {m} on training scenes: {sum(x['success'] for x in r)}/{len(r)}")
print("\nmedia:", s.get("media"))
PYEOF
log "finisher complete: $G/summary.json, $G/RESULTS_DIGEST.txt, docs/assets/v2_gen_*.png"
