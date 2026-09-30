# Capacity-density infrastructure recovery protocol

Registered 2026-09-30 after the original results were inspected, before any
recovery training. This is a disclosed infrastructure repair, not a new blind
pre-registration or a change to the original hypotheses.

## Frozen evidence and inclusion rule

Baseline `2c86475` and original capacity artifacts remain read-only. Audit
commit `7e2256713ddaee6d4408f2bbe52c6b8e75f715a8` records the defects.
Select **every** run whose committed integrity audit records a resume or a
checkpoint/evaluation hash mismatch, regardless of its success rate. This
selects 16 runs (3 at 4.3M, 4 at 9.1M, 9 at 19.5M); the two provenance
mismatches are included in those 16. Reuse the other 29 runs unchanged.
Never choose between original and recovered outcomes based on performance.
Publish both versions and their differences.

## Unchanged scientific protocol

Original H1–H4, physics-v2 validity, datasets, normalization, seeds 0/1/2,
frozen vision encoder, H=16, K=8, DDIM-10, rollout limit 150, training-scene
diagnostic and offline metrics remain as originally specified. No new data.
The model configurations and exact trainable counts remain d256/L5/h8:
4,304,902; d320/L7/h10: 9,137,286; d416/L9/h13: 19,517,062.
Frozen parameters are 927,008 per model. Baseline remains 2,009,670 trainable.

Restart affected runs from their original seed, not legacy raw checkpoints,
because those cannot restore exact EMA. Use the existing trainer, AdamW
lr .001, weight decay .0001, clipping 1, fp32, EMA .999 and batch 8.
Complete 80,222 updates for every recovery run. Preserve the actual historical
within-condition exposure: r10 148.181944 passes, r15 148.011070,
r20 147.704488, one-sided 148.696942, r7.5 surrounded 148.456165.
Do not silently correct the original cross-condition exposure discrepancy.
This matches data exposure within conditions, not FLOPs.

## Execution gates and preservation

New outputs only under `artifacts/capacity_recovery_v1/`; smoke outputs in a
separate subdirectory. Commit this protocol before training. Require existing
EMA/atomic checkpoint tests and an actual TinyRDT interrupted/uninterrupted
trainer test to pass before expensive runs. Compare raw weights, EMA,
optimizer and random-stream states exactly at the same final update. Test
all three recovery architectures. A failure stops recovery launch and is
documented, not bypassed. Validate checkpoint/dataset hashes before evaluation.

Maximum two workers total; default one training worker to limit WSL pressure.
Use setsid for detached work. Never reduce batch size, alter the optimizer,
restart poor scientific results, or overwrite historical outputs. Infrastructure
continuations must use same-transaction embedded EMA checkpoints. Preserve
attempt logs and cumulative runtime; monitor RAM, GPU memory, temperature,
NaNs and process exit status. Completion requires verified outputs, not just
a pipeline completion marker. Do not enable the old launcher or watchdog.

## Analysis and stop

Use the original condition-specific raw counts and every seed; d50 remains
the estimated distance associated with 50% closed-loop success, not a radius.
Use condition-stratified position resampling shared across capacities, and
seed resampling independently within capacity, retaining seed IDs across
conditions within each capacity; 4,000 replicates with fixed RNG seed 20260930.
Report unsupported/extrapolated distance estimates explicitly. Correct the
seed-clustering implementation with regression tests before recovery analysis.
All comparisons are qualified as post-audit recovery. Preserve original figures
and summaries; recovery reports/figures use new paths. Include mechanisms,
position rescue/regression, matched trajectories and compute limitations.

Stop after the 16 selected recoveries, their evaluations, integrity checks,
corrected analysis and recovery report/PR are complete. Do not train 40M or
add conditions/seeds/architectures. The original 40M recommendation gate remains
unchanged and authorizes a recommendation only.
