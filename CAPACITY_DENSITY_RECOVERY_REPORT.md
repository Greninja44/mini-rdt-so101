# Capacity × density recovery: the EMA defect was real, and it does not explain the large-model collapse

**Headline: the infrastructure defect was genuine — all 16 flagged runs produced different EMA checkpoints when retrained correctly — but
only 2 of 16 changed their outcome, and the corrected 45-run matrix is materially identical to the qualified one. Larger TinyRDT policies
still fail to fit their own training data under this recipe, so the capacity comparison is optimization-confounded (CASE D) and must not be
read as "capacity hurts generalization".**

Recovery protocol: `docs/research/capacity_density_recovery_spec.md` (Codex, commit `7e9853e`). Audit that triggered it: `4469c35`.
Frozen baseline: `2c86475`. Raw numbers: `artifacts/capacity_recovery_v1/{summary.json,RESULTS_DIGEST.txt}`,
`docs/research/capacity_recovery_integrity_audit.json`, `docs/research/capacity_recovery_equivalence.json`.

---

## 1–2. Why recovery was required, and the defect
Evaluation uses EMA weights. Interrupted training resumed **without restoring EMA state**: the raw model, optimizer and RNG streams were
restored, but the exponential moving average was re-initialised from the resumed weights. Any run that resumed therefore did not produce the
EMA trajectory the protocol specifies, and its evaluated checkpoint was not the specified model.

## 3–4. Affected runs, and why their concentration mattered
The committed audit flagged **16 of 45** runs: **3 at 4.3M, 4 at 9.1M, 9 at 19.5M**. The defect was concentrated in exactly the capacities
carrying the headline result, so the original matrix could not support a causal capacity claim — a 19.5M collapse could have been an
artifact of its own interruptions. That is what this recovery tests.

## 5. Selection rule (outcome-independent)
Runs were selected **only** by the committed audit: a recorded resume, or a checkpoint/evaluation hash mismatch. Not by success rate, MAE,
model size, or whether a result looked wrong. The selection is recomputed programmatically by the runner from
`docs/research/capacity_integrity_audit.json`, so it cannot drift:

```
m19.5: r10.0 seeds 0,1,2 · r15.0 seeds 0,1 · r20.0 seeds 0,1 · r7.5_onesided seed 0 · r7.5 seed 2     (9)
m9.1 : r10.0 seeds 0,1 · r7.5_onesided seeds 1,2                                                      (4)
m4.3 : r7.5 seeds 0,1,2                                                                               (3)
```

No run was added because its result looked suspicious, and none was dropped because recovery did not help it.

## 6–7. The fix and atomic checkpoints
EMA state is now written **inside the same checkpoint transaction** as the raw weights and optimizer, saved via write-then-atomic-replace,
and restored on resume. A resume whose checkpoint lacks a matching embedded EMA is **refused** rather than silently re-initialised, so the
defect cannot recur quietly. Unit tests cover the mechanism (`tests/test_ema_resume.py`, `tests/test_atomic_checkpoint.py`).

## 8–9. Real-trainer equivalence gate (bit-exact)
The protocol requires more than unit tests: an interrupted/uninterrupted comparison on the **actual TinyRDT trainer**, for every recovery
architecture, before spending compute. `scripts/capacity_recovery_equivalence.py` trains each architecture twice to the same final update —
once straight through, once interrupted at step 200 and resumed — and compares every stateful tensor.

| architecture | raw weights | EMA weights | optimizer | train RNG | noise RNG | torch RNG | CUDA RNG | max abs diff |
|---|---|---|---|---|---|---|---|---|
| d256 / L5 / h8 (4.3M) | identical | identical | identical | identical | identical | identical | identical | **0.0** |
| d320 / L7 / h10 (9.1M) | identical | identical | identical | identical | identical | identical | identical | **0.0** |
| d416 / L9 / h13 (19.5M) | identical | identical | identical | identical | identical | identical | identical | **0.0** |

**Gate passed before any recovery training started.** This is what makes the later in-flight resumes scientifically harmless.

## 10–11. Execution and interruptions
All 16 runs were retrained from their original seeds into `artifacts/capacity_recovery_v1/`, 80,222 updates each, batch 8, AdamW lr 1e-3,
wd 1e-4, clip 1.0, EMA 0.999, fp32 — identical to the original recipe. **The learning rate was not touched**, deliberately: changing it here
would confound checkpoint recovery with capacity-specific optimisation.

Three further WSL virtual-machine shutdowns occurred during recovery (2026-10-03 ~05:35, and twice around 2026-10-05 06:30–06:50). Unlike
the historical interruptions, the in-flight runs carried embedded EMA in their atomic checkpoints and were **continued exactly**
(`m4.3_r7.5_seed0`, `m9.1_r7.5_onesided_seed1`, `m9.1_r7.5_onesided_seed2`). The recovery integrity audit records those resumes; by the gate
above they are bit-identical to uninterrupted training and are **not** instances of the defect. After the first shutdown the runner was moved
to an enabled user service so the queue resumes automatically at boot, which is what carried it through the later two.

Worker ceiling: one training and one evaluation worker for most of the run. For the final two 9.1M runs the user asked for full GPU use, so a
second training worker was enabled (3 workers total, above the protocol's ceiling of 2). Training is bit-deterministic given a seed, so this
changed wall time only; it is logged in `docs/research/capacity_density_scaling_deviation.md`.

## 12. Original vs recovered, per run

| run | capacity | held-out orig → rec | train MAE orig → rec | training scenes orig → rec | EMA hash changed |
|---|---|---|---|---|---|
| m19.5_r10.0_seed0 | 19.5M | 3/8 → 3/8 | 0.0131 → 0.0131 | 4/10 → 4/10 | yes |
| m19.5_r10.0_seed1 | 19.5M | 5/8 → 5/8 | 0.0161 → 0.0161 | 6/10 → 6/10 | yes |
| m19.5_r10.0_seed2 | 19.5M | 4/8 → 4/8 | 0.0103 → 0.0103 | 7/10 → 7/10 | yes |
| m19.5_r15.0_seed0 | 19.5M | 1/4 → 1/4 | 0.0131 → 0.0131 | 4/10 → 4/10 | yes |
| **m19.5_r15.0_seed1** | 19.5M | **0/4 → 2/4** | 0.0125 → 0.0125 | **0/10 → 4/10** | yes |
| m19.5_r20.0_seed0 | 19.5M | 1/3 → 1/3 | 0.0119 → 0.0119 | 5/10 → 5/10 | yes |
| m19.5_r20.0_seed1 | 19.5M | 0/3 → 0/3 | 0.0122 → 0.0122 | 3/10 → 3/10 | yes |
| m19.5_r7.5_seed2 | 19.5M | 3/10 → 3/10 | 0.0143 → 0.0143 | 3/10 → 3/10 | yes |
| m19.5_r7.5_onesided_seed0 | 19.5M | 3/10 → 3/10 | 0.0104 → 0.0104 | 4/10 → 4/10 | yes |
| m9.1_r10.0_seed0 | 9.1M | 7/8 → 7/8 | 0.0074 → 0.0074 | 10/10 → 10/10 | yes |
| m9.1_r10.0_seed1 | 9.1M | 5/8 → 5/8 | 0.0107 → 0.0107 | 7/10 → 7/10 | yes |
| m9.1_r7.5_onesided_seed1 | 9.1M | 2/10 → 2/10 | 0.0108 → 0.0108 | 9/10 → 9/10 | yes |
| m9.1_r7.5_onesided_seed2 | 9.1M | 3/10 → 3/10 | 0.0100 → 0.0100 | 9/10 → 9/10 | yes |
| m4.3_r7.5_seed0 | 4.3M | 10/10 → 10/10 | 0.0101 → 0.0101 | 10/10 → 10/10 | yes |
| **m4.3_r7.5_seed1** | 4.3M | **9/10 → 10/10** | 0.0088 → 0.0088 | **7/10 → 9/10** | yes |
| m4.3_r7.5_seed2 | 4.3M | 10/10 → 10/10 | 0.0074 → 0.0074 | 9/10 → 9/10 | yes |

**16/16 EMA checkpoints changed; 2/16 outcomes changed, both upward.** Training MAE is identical to four decimals in all 16 — mechanically
expected, because that metric comes from the raw weights, which resume restored correctly. Only the average was lost.

**Why so few outcomes moved:** at decay 0.999 the EMA has a memory of roughly 1,000 updates. A run interrupted early in its 80,222 steps
re-converges to essentially the specified average even after the reset; only interruptions within the last few thousand updates leave a
materially different EMA. That is the mechanism behind both changed runs, and it is why the defect was real but mostly inconsequential.

Figure: `docs/assets/v2_capacity_recovery_integrity.png`.

## 13–14. Corrected 45-run matrix
Assembled programmatically: a cell uses its **recovered** run when the audit flagged it, otherwise its **original** run; the 2M baseline
cells are reused unchanged from the density sweep. The contaminated versions appear only in §12, never as extra observations.

- recovered: **16** · reused original: **29** · protocol matrix: **45 new runs** · plus 15 reused 2M baseline cells = 60 cells · missing: none.

| capacity | r10 | r15 | r20 | r7.5 one-sided | r7.5 control | primary total |
|---|---|---|---|---|---|---|
| 2.0M | 18/24 (75%) | 7/12 (58%) | 3/9 (33%) | 17/30 (57%) | 28/30 (93%) | 45/75 (60.0%) |
| **4.3M** | **21/24 (88%)** | **8/12 (67%)** | **4/9 (44%)** | **20/30 (67%)** | **30/30 (100%)** | **53/75 (70.7%)** |
| 9.1M | 18/24 (75%) | 3/12 (25%) | 3/9 (33%) | 13/30 (43%) | 18/30 (60%) | 37/75 (49.3%) |
| 19.5M | 12/24 (50%) | 3/12 (25%) | 3/9 (33%) | 11/30 (37%) | 9/30 (30%) | 29/75 (38.7%) |

Only two cells differ from the qualified matrix: 19.5M r15 (1/12 → 3/12) and 4.3M r7.5 control (29/30 → 30/30). **The qualified conclusion
survives correction.**

## 15–16. Do the large models train?
| capacity | final train MAE (mean) | offline train MAE | offline eval MAE | training-scene success |
|---|---|---|---|---|
| 2.0M | 0.0079 | 0.00477 | 0.00644 | **139/150 (93%)** |
| 4.3M | 0.0087 | 0.00532 | 0.00664 | **146/150 (97%)** |
| 9.1M | 0.0102 | 0.00682 | 0.00788 | 120/150 (80%) |
| 19.5M | 0.0124 | 0.00956 | 0.01006 | **70/150 (47%)** |

**No.** After correct EMA semantics, the 19.5M models still fit their training data roughly twice as poorly as the 2M baseline and execute
fewer than half of the scenes they were trained on. A model that cannot reproduce its own demonstrations cannot be used to measure
generalization. This is the single most important result of the recovery.

## 17–19. Held-out results and corrected distance-response curves
Per-seed counts are in `summary.json`; aggregates above. Logistic fits over the four shared surrounded conditions, with position-cluster
bootstrap intervals:

| capacity | slope / 10 mm | d90 | d75 | d50 | d25 | d10 |
|---|---|---|---|---|---|---|
| 2.0M | −2.35 | 7.0 | 11.7 | **16.4** | 21.1 | 25.7 |
| 4.3M | −2.97 | 10.0 | 14.2 | **18.3** | 22.4 | 26.6 |
| 9.1M | −1.35 | −3.5 | 4.7 | **12.8** | 20.9 | 29.0 |
| 19.5M | −0.15 | unsupported | unsupported | **unsupported** (algebraic −27.3) | unsupported | unsupported |

d50 is the estimated distance associated with 50% success, **not a radius**. The 19.5M fit is not interpretable: its rates are flat, low and
non-monotonic across distance, so a distance-response curve has nothing to latch onto. Values outside the measured 7.5–20 mm band are
extrapolations.

## 20. Capacity × distance interaction (corrected matrix, corrected clustering)
300 surrounded rollouts, 25 positions, 4,000 replicates, fixed RNG seed 20260930. Positions are resampled condition-stratified and shared
across capacities; seeds are resampled independently **within** capacity — this fixes the defect the audit recorded, where seed IDs were
treated as shared clusters across sizes. Five regression tests pin the structure (`tests/test_recovery_bootstrap.py`).

| model | coefficient | 95% CI |
|---|---|---|
| distance (per 10 mm) | −1.55 | [−3.21, −0.31] |
| log2(params) | **−0.67** | [−1.06, −0.38] |
| with interaction — distance | −2.99 | [−6.35, −1.02] |
| with interaction — log2(params) | **−1.64** | [−3.37, −0.52] |
| with interaction — distance × log2(params) | **+0.83** | [+0.03, +1.96] |

The capacity main effect is **negative** and the interaction is positive. The positive interaction does **not** mean larger models
generalize further: it means their success declines less steeply with distance because they are already failing at the easiest distance
(a floor effect). Read together with §15, these coefficients describe under-trained models, not a capacity–generalization law.

## 21. Support geometry
| capacity | r7.5 surrounded | r7.5 one-sided | gap |
|---|---|---|---|
| 2.0M | 93.3% | 56.7% | **+36.7 pp** |
| 4.3M | 100% | 66.7% | +33.3 pp |
| 9.1M | 60.0% | 43.3% | +16.7 pp |
| 19.5M | 30.0% | 36.7% | −6.7 pp |

Capacity effect on one-sided support alone: **−0.31 logits per doubling [−0.90, +0.16]** — no benefit. The gap shrinks at large capacity only
because surrounded performance collapses toward the one-sided level (28 → 30 → 18 → 9 of 30). **CASE E: the support-geometry penalty is not
solved by capacity**; CASE F is not supported.

## 22. Lateral aiming error
Median lateral error of the grasp centre at the first close (mm):

| capacity | r7.5 | r10 | r15 | r20 | one-sided |
|---|---|---|---|---|---|
| 2.0M | 3.7 | 7.4 | 8.7 | 9.9 | 10.0 |
| 4.3M | 5.4 | 6.6 | 8.1 | 11.9 | 8.7 |
| 9.1M | 9.5 | 7.7 | 16.4 | 12.9 | 12.8 |
| 19.5M | 19.0 | 9.5 | 19.6 | 17.7 | 17.3 |

Capacity does **not** reduce the aiming error that sparse coverage produces; beyond 4.3M it grows substantially, including at the easy
7.5 mm control (3.7 → 19.0 mm). This is the mechanistic signature of under-fitting rather than of a capacity-limited representation.

## 23. Position-level change versus 2M (35 shared positions, 3 seeds each)
| capacity | improved | unchanged | regressed | rescued 0/3 → ≥2/3 | lost 3/3 → ≤1/3 |
|---|---|---|---|---|---|
| 4.3M | **8** | 26 | 1 | 2 | 0 |
| 9.1M | 6 | 13 | **16** | 1 | 7 |
| 19.5M | 4 | 14 | **17** | 1 | **12** |

4.3M's gain is broad and nearly monotone across positions; the larger models lose positions the 2M baseline solved reliably.

## 24. Offline versus closed loop
Across all 60 corrected cells the correlation between offline evaluation-position MAE and closed-loop success is **−0.60**, and mean offline
MAE rises with capacity (0.0064 → 0.0066 → 0.0079 → 0.0101). Within capacity: −0.64 (2.0M), −0.72 (4.3M), −0.48 (9.1M), **−0.02 (19.5M)**.

Unlike the earlier phases — where offline error was blind to closed-loop competence — here it does track the degradation, because the large
models are genuinely worse at fitting, not merely worse at executing. That is consistent with an optimisation failure and is **not** a
retraction of the earlier finding, which concerned models that fit equally well.

## 25–26. Compute
| capacity | train wall (min, mean) | peak VRAM | predict latency CUDA | CPU |
|---|---|---|---|---|
| 2.0M | 49.5 | 966 MB | 112.0 ms | 65.0 ms |
| 4.3M | 48.9 | 1,242 MB | 117.7 ms | 80.7 ms |
| 9.1M | 68.2 | 1,575 MB | 122.3 ms | 112.4 ms |
| 19.5M | 76.8 | 2,419 MB | 141.1 ms | 165.6 ms |

Wall times mix original runs (whose timers reset across resumes and whose concurrency varied) with clean recovery runs, so they are
indicative only; VRAM and latency are architecture properties and comparable. Latency was measured with an identical DDIM-10 sampler at batch
1 on the same hardware. At K = 8 the controller replans every 8 × 50 ms = 400 ms, so every size remains feasible in simulation; this says
nothing about a real robot.

## 27. Physical validity
420 corrected rollouts. **Zero successes carried an invalid reason.** One rollout exceeded a tolerance (pad–cube 1.573 mm against the
1.5 mm limit) and was correctly rejected as a failure. Max robot–table penetration 0.00 mm; max cube–table 0.767 mm. Physics-v2 criteria were
unchanged throughout.

## 28. Statistics
The inferential dataset is the 45 protocol-valid new runs plus the 15 reused 2M baseline cells. The contaminated versions of the 16 flagged
runs appear only in §12; they are never counted as additional observations. All intervals use the corrected cluster structure described in
§20; Wilson intervals on raw counts are reported in `summary.json` and are not the primary inference.

## 29. Deviations
- Three WSL shutdowns during recovery; in-flight runs were continued from embedded-EMA atomic checkpoints (gate-certified bit-exact), not
  restarted. Logged.
- A second training worker for the final two runs at the user's request (3 workers vs the protocol's 2). Bit-determinism makes this a wall-time
  change only. Logged.
- Latency reuses the measurement from the original phase; the architectures and sampler are identical and latency does not depend on weights.
- Everything else followed the protocol: selection rule, seeds, exposure, optimizer, evaluation configuration, K, physics criteria.

## 30. Limitations
- **The capacity comparison remains uninterpretable as a generalization result**, because 9.1M and 19.5M do not fit their training data under
  this recipe. No conclusion about capacity and spatial generalization can be drawn from it, in either direction.
- One recipe (lr 1e-3, batch 8, 80,222 updates, no warmup or schedule), chosen for 2M. The pre-registration flagged this as a scaling risk and
  this recovery confirms the risk materialised; it does not isolate which hyperparameter is responsible.
- Three seeds per cell; 8/4/3/10/10 evaluation positions per condition, so single cells are noisy.
- Matched data exposure, not matched compute.
- The 4.3M improvement over 2.0M (53/75 vs 45/75 on primary conditions, 8 positions improved vs 1 regressed) is suggestive but comes from one
  comparison without multiplicity control.

## 31. Reproduction
```bash
# gate (must pass before any recovery training)
.venv/bin/python scripts/capacity_recovery_equivalence.py

# recovery: selection is recomputed from the committed audit, never from outcomes
scripts/experiments/run_capacity_recovery.sh          # or: systemctl --user start mini-rdt-recovery.service
scripts/capacity_integrity_audit.py --root artifacts/capacity_recovery_v1 \
    --output docs/research/capacity_recovery_integrity_audit.json

# corrected analysis, figures and digest
.venv/bin/python -m evaluation.capacity_recovery_analysis
env -u PYTHONPATH .venv/bin/python -m pytest -q tests/
```

## 32. Artifacts
Recovered models, rollouts and offline metrics: `artifacts/capacity_recovery_v1/`. Originals preserved untouched in
`artifacts/capacity_scaling/` (90 entries, unchanged). Integrity records: `docs/research/capacity_integrity_audit.json` (original),
`docs/research/capacity_recovery_integrity_audit.json` (recovery: 16 runs, every `last.pt` carries an embedded EMA, all rollout hashes match
their checkpoints, baseline and dataset hashes verified, 80,221 final step, 147.70–148.70 passes), and
`docs/research/capacity_recovery_equivalence.json`. Figures: `docs/assets/v2_capacity_recovery_integrity.png`,
`v2_capacity_corrected_by_condition.png`, `v2_capacity_corrected_analysis.png`.

## 33. Decision gate and recommendation
- **CASE A — rejected.** The EMA defect does not explain the large-model collapse: 14 of 16 outcomes were unchanged and the corrected matrix
  matches the qualified one.
- **CASE D — applies, and dominates.** 9.1M and 19.5M still fail to fit their own training data (training-scene 80% and 47%; train MAE 1.3×
  and 2.0× the baseline). The capacity experiment is optimization-confounded.
- **CASE E — applies.** The surrounded-vs-one-sided penalty is not solved by capacity; the apparent gap shrinkage is surrounded collapse.
- **CASE B, C, F — not supported.** No evidence that capacity expands range or improves weak-support geometry.
- **CASE G — not reproduced in this phase.** Here offline error does track the degradation, because the large models fit worse; the earlier
  offline/closed-loop disconnect concerned equally-fit models and stands unchanged.

**Is 40M justified? No.** Scaling further while 19.5M cannot fit 80 demonstrations would measure the optimisation failure, not capacity.
**40M was not trained**, as pre-registered.

**Recommended next experiment: a separately pre-registered optimisation study at fixed capacity.** Take 9.1M and 19.5M on one or two density
conditions and test whether standard remedies — learning-rate scaling with width, warmup, a decay schedule, weight-decay adjustment, gradient-norm
monitoring, or a longer horizon — restore training-scene execution to the ≥90% that 2.0M and 4.3M reach. Only if a larger model can fit its
training data does the capacity question become answerable. The density datasets, benchmark and evaluation protocol are reusable unchanged.
