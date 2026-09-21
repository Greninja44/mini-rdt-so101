# Physics-v2 training-seed replication: what is data geometry, and what was one lucky run?

**Headline: the coverage conclusion replicates, and the most striking single-seed result does not.** Across 5 independent training seeds per
dataset scale (20 models, 1,120 held-out rollouts), the distance to the nearest demonstration remains the dominant predictor
(−2.51 logit per 10 mm, 95% CI [−3.53, −1.79]), while dataset size contributes **+0.06 [−0.22, +0.32]** once distance is conditioned on.
TRAIN40's B = 14/20 did **not** replicate (14, 3, 3, 7, 5 → mean 6.4). Training-seed variance is real but much smaller than position
difficulty, except at 20 demonstrations, where it is large enough to overturn conclusions drawn from one run.

Pre-registration: `docs/research/training_seed_replication_spec.md` (commit `147ca2a`, before any model here was trained).
Frozen prior result: `docs/research/exposure_matched_scaling_freeze.json` (commit `c2526f5`).
Raw numbers: `artifacts/seed_replication/{summary.json,RESULTS_DIGEST.txt}`.

---

## 1. Research question
**Primary:** how much closed-loop variance comes from TinyRDT training initialisation / optimisation randomness at each dataset scale?
**Secondary:** after averaging over seeds, does local demonstration distance still explain the apparent benefit of larger datasets?
**Tertiary:** is TRAIN40's sparse-region result reproducible, or was it seed-specific?

## 2. Motivation
The previous phase drew its conclusions from one training run per data scale. Two results were suspicious: category B inverted between
scales (TRAIN40 14/20 vs TRAIN80 4/20) at identical local coverage, and matched exposure failed to equalise training-scene competence
(9/10, 7/10, 6/10, 10/10). Sampler seeds had already been ruled out as the cause, which left the training run itself.

## 3. Previous single-seed result (frozen, unchanged)
TRAIN10 15/56, TRAIN20 17/56, TRAIN40 26/56, TRAIN80 28/56, at K = 8, sampler seed 0.

## 4. Pre-registration
Fixed in advance: five seeds per scale, the exposure-matched budgets, the shared training-scene diagnostic, the frozen 56-position
benchmark, K = 8, a fixed evaluation sampler seed, the statistics (two-way cluster bootstrap, per-position frequencies, permutation tests),
the distance bins, and the decision gate. Predictions: **H1** seed SD ≥ 3/56 at every scale, **H2** TRAIN40 B = 14/20 does not replicate,
**H3** the dataset-size coefficient stays near zero after conditioning on distance, **H4** training-scene success predicts held-out success
better than offline MAE, **H5** extrapolation stays absent. **H2, H3 and H5 hold; H1 holds at 3 of 4 scales; H4 is not supported as stated
(§21).**

## 5. Exact seeds
**17, 1, 2, 3, 4** at every scale. Seed 17 is the existing exposure-matched run, counted as one replicate because its procedure, budget and
configuration are exactly those pre-registered; the other 16 models were trained here. Each seed sets Python `random`, NumPy, the torch
global RNG (parameter init), the batch-order stream (`seed+100`), the diffusion-noise stream (`seed+200`) and the state-dropout stream
(`seed+300`, unused). The evaluation sampler seed stayed 0 for all 20 models, so training variance is never mixed with sampling variance.

## 6. Reproducibility
Two 201-step runs of TRAIN10 with seed 1 under identical settings produced **bit-identical weights** (max abs difference 0.0), identical
losses at steps 0/100/200 (0.976265192032, 0.072721101344, 0.076263666153) and identical sampled metrics. TF32 is off, cuDNN autotuning is
off, and features are pre-extracted, so the training pipeline is deterministic given a seed. No residual nondeterminism had to be documented.

## 7. Training matrix
4 scales × 5 seeds = 20 models, each at its frozen exposure-matched budget (10,055 / 20,037 / 40,018 / 80,000 steps; 147.87 passes).
No early stopping, no run extended, no seed retrained, no seed dropped.

## 8. Compute and failures
No infrastructure failures: all 16 new models trained and all evaluations completed. 16 × (10 training-scene + 56 held-out) = 1,056 new
rollouts, plus the 4 reused seed-17 cells. Training ≈ 4.3 h on one GPU; evaluation ≈ 12 h on 2 simulation workers, at the bit-exact software
renderer.

## 9. Training-scene diagnostic (10 shared CLEAN10 scenes, K = 8)

| scale | per seed (17, 1, 2, 3, 4) | range |
|---|---|---|
| TRAIN10 | 9, 7, 8, 10, 10 | 7–10 |
| TRAIN20 | 7, 7, 8, 9, **5** | 5–9 |
| TRAIN40 | **6**, 9, 9, 8, 8 | 6–9 |
| TRAIN80 | 10, 10, 10, 10, 10 | **10–10** |

The previous pattern (9 / 7 / 6 / 10) was **partly seed-specific**: TRAIN40's 6/10 is its worst seed, not typical (mean 8.0). What does
replicate is that **only the 80-demo models reliably execute their own training scenes — 10/10 in every seed** — while the 20- and 40-demo
models never exceed 9/10. Matched exposure genuinely does not equalise trainability.

## 10. Offline metrics
Held-out MAE is nearly constant within a scale and separates scales cleanly: ≈0.034 (10 demos), ≈0.014 (20), ≈0.010 (40), ≈0.008 (80; the
seed-17 model at 0.0101 is its own worst). Training MAE: ≈0.0073, 0.0061, 0.0053, 0.0054. Validation tracks held-out. §21 shows what this
does and does not predict.

## 11–12. Held-out results and per-scale variance (K = 8, sampler seed 0, 56 positions)

| scale | per seed (17, 1, 2, 3, 4) | mean | SD | median | range | bootstrap 95% CI of the mean |
|---|---|---|---|---|---|---|
| TRAIN10 | 15, 10, 9, 11, 8 | **10.6/56** | 2.70 | 10 | 8–15 | [8.8, 13.0] |
| TRAIN20 | 17, 24, 21, 25, **8** | **19.0/56** | **6.89** | 21 | 8–25 | [13.2, 23.8] |
| TRAIN40 | 26, 23, 25, 28, 27 | **25.8/56** | 1.92 | 26 | 23–28 | [24.2, 27.2] |
| TRAIN80 | 28, 28, 31, 33, 28 | **29.6/56** | 2.30 | 28 | 28–33 | [28.0, 31.6] |

Figure: `docs/assets/v2_seed_variance.png` (every seed plotted, mean and SD overlaid).

**How large is training-seed variance?** SD is 2–3 successes out of 56 at 10, 40 and 80 demonstrations — small relative to the 19-success gap
between the 10- and 80-demo means. At **20 demonstrations it is 6.9**, with a range of 8–25: a single run there can look worse than the
10-demo mean or better than the 40-demo mean. The previous single-seed TRAIN20 number (17) sat mid-range, but one run at that scale carries
almost no information.

## 13. A / B / C by scale and seed

| scale | A interpolation (20) | mean ± SD | B sparse (20) | mean ± SD | C extrapolation (16) | mean ± SD |
|---|---|---|---|---|---|---|
| TRAIN10 | 11, 10, 8, 10, 8 | 9.4 ± 1.3 | 2, 0, 0, 1, 0 | 0.6 ± 0.9 | 2, 0, 1, 0, 0 | 0.6 ± 0.9 |
| TRAIN20 | 12, 14, 17, 16, 6 | 13.0 ± 4.4 | 2, 9, 2, 6, 0 | 3.8 ± 3.6 | 3, 1, 2, 3, 2 | 2.2 ± 0.8 |
| TRAIN40 | 10, 16, 20, 20, 18 | 16.8 ± 4.1 | **14**, 3, 3, 7, 5 | 6.4 ± 4.6 | 2, 4, 2, 1, 4 | 2.6 ± 1.3 |
| TRAIN80 | **20, 20, 20, 20, 20** | **20.0 ± 0.0** | 4, 3, 7, 9, 6 | 5.8 ± 2.4 | 4, 5, 4, 4, 2 | 3.8 ± 1.1 |

The strongest result in the whole project is now the most robust one: **80 demonstrations solve all 20 dense-interpolation positions in
every one of five independent training runs.** Category B is the noisiest cell at every scale, and B does not separate 40 from 80 demos
(6.4 vs 5.8, overlapping spreads).

## 14. The TRAIN40 anomaly
**It did not replicate.** TRAIN40 category B by seed: **14**, 3, 3, 7, 5. The 14/20 was the single best of five runs, nearly 2 SD above its
own mean, and no other seed came close. Per-position across the 5 TRAIN40 seeds (`docs/assets/v2_seed_trainB_matrix.png`), only 3 of the 20
B positions are solved by all five seeds, 7 by none, and the rest intermittently.

The prior report's caution was correct: that inversion was **a seed-specific optimisation outcome, not a property of 40 demonstrations**.
TRAIN40 was not modified to explain the result, and the seed-17 run stays in every average.

## 15. Position stability across independently trained policies
Number of the 56 positions solved by k of 5 seeds:

| scale | 0/5 | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 |
|---|---|---|---|---|---|---|
| TRAIN10 | 37 | 5 | 5 | 3 | 1 | 5 |
| TRAIN20 | 22 | 7 | 6 | 12 | 5 | 4 |
| TRAIN40 | 18 | 7 | 4 | 6 | 9 | 12 |
| TRAIN80 | **17** | 7 | 2 | 5 | 3 | **22** |

Outcomes are strongly bimodal at 80 demos: 39 of 56 positions are all-or-nothing across five independently trained policies. A
**variance decomposition** confirms the same thing: the share of explained outcome variance attributable to position rather than seed is
98% (10 demos), 90% (20), 99% (40) and 99% (80). Position difficulty, not the training run, is what decides most rollouts.

## 16. Coverage-conditioned model (all replicates)
Pooled over 1,120 rollouts (56 positions × 5 seeds × 4 scales), logistic regression with a **two-way cluster bootstrap resampling both
positions and training seeds**:

| model | coefficient | 95% CI |
|---|---|---|
| distance only | **−2.55 per 10 mm** | [−3.49, −1.90] |
| log2(demos) only | **+0.51 per doubling** | [+0.35, +0.71] |
| **both** — distance | **−2.51 per 10 mm** | [−3.53, −1.79] |
| **both** — log2(demos) | **+0.06 per doubling** | [−0.22, +0.32] |

Identical in structure to the single-seed result, with five times the data: dataset size looks beneficial alone and contributes essentially
nothing once local coverage is conditioned on.

For completeness, the raw scale effect is real and significant at the position level — paired permutation tests on per-position success
frequencies give 20 vs 10 demos p = 0.002, 40 vs 20 p = 0.0002, 80 vs 40 p = 0.026, 80 vs 10 p < 1e-4. **More demonstrations do help; the
mechanism is coverage.**

## 17. Does the coverage hypothesis survive replication?
**SUPPORTS.** With 5 seeds the distance coefficient is unchanged in sign, magnitude and precision, and the residual dataset-size coefficient
is centred near zero with a CI spanning zero in both directions. Nothing in the replication suggests a hidden diversity benefit.

Honest bound: distance and dataset size remain correlated by construction in a nested design, so this rules out a *large* residual effect,
not a small one. The CI [−0.22, +0.32] excludes anything close to the +0.51 raw effect.

## 18. Distance response (aggregated over all seeds, pre-registered bins)

| nearest training cube | trials | successes | rate |
|---|---|---|---|
| 0–2.5 mm | 10 | 10 | **100%** |
| 2.5–5 mm | 140 | 112 | 80% |
| 5–7.5 mm | 215 | 118 | 55% |
| 7.5–10 mm | 115 | 64 | 56% |
| 10–15 mm | 355 | 113 | 32% |
| > 15 mm | 285 | 8 | **2.8%** |

Trials share positions and seeds, so the naive intervals in `summary.json` are optimistic; the monotone decline (apart from the flat
5–10 mm region) is the point.

## 19. Generalization curve
A logistic fit of success against nearest-demo distance (position-bootstrap CIs):

| model-predicted success | distance | 95% CI |
|---|---|---|
| 90% | 0.2 mm | [−2.8, 2.7] |
| 75% | 4.5 mm | [2.4, 6.3] |
| **50%** | **8.8 mm** | [7.2, 10.3] |
| 25% | 13.1 mm | [11.3, 15.0] |
| 10% | 17.4 mm | [15.0, 20.2] |

**Success probability decreases smoothly with nearest-demo distance.** This is a fitted curve, not evidence of a hard generalization radius;
the 90% extrapolation is outside the data and its CI includes negative distances, which is exactly why no threshold claim is made. These
estimates are the natural input to the density sweep.

## 20. Training-scene success vs held-out success
Across all 20 models the correlation is +0.35 (Spearman +0.55) with a bootstrap CI of [−0.17, +0.75] — not significant. Pooled **within**
scale (removing the data-scale effect) it is +0.44 [−0.18, +0.76]. The one scale where it matters most is suggestive: at 20 demonstrations,
training-scene success correlates +0.88 with held-out success across the five seeds, and the worst run (5/10 training scenes) is exactly the
run that collapsed to 8/56 held out.

So the diagnostic looks useful for catching a badly optimised run, but with 5 models per scale this is not established. It is not used to
select or drop any seed here.

## 21. Offline MAE vs closed-loop success — the important correction
Across all 20 models, held-out offline MAE correlates strongly with held-out success (Pearson −0.83, CI [−0.97, −0.63]). **That correlation
is entirely a data-scale effect.** Within a scale, where the only difference is the training run, it vanishes and even flips sign:

| pooled within-scale predictor | Pearson | 95% CI |
|---|---|---|
| offline held-out MAE | **+0.15** | [−0.21, +0.52] |
| training-scene closed-loop success | +0.44 | [−0.18, +0.76] |

Per scale, offline MAE vs success is +0.29 (10 demos), +0.48 (20), +0.34 (40), −0.32 (80) — no consistent sign. This refines the project's
standing finding: **offline MAE ranks data scales but carries no information about which training run will execute well.** Neither predictor
is established as reliable at n = 5 per scale; offline MAE is disqualified for within-scale model selection, and the training-scene
diagnostic remains the better candidate without being proven.

## 22. Failure modes across seeds
One mode dominates at every scale and shrinks with data: `wrong_lateral_alignment` 210 → 166 → 137 → 116 of the failures at 10 → 20 → 40 → 80
demos. The rest are single digits (`cube_tilt` 10 → 5 → 5 → 1, `early_close` 2 → 6 → 6 → 8, `cube_nudge_displacement` 5 → 3 → 2 → 5,
`wrong_height` 0 → 3 → 0 → 0).

When independently trained policies fail at the same position, they fail **the same way in 78.5%** of position × scale cells (128 of 163).
Failure is mostly a property of the position, not of the run — the same conclusion the variance decomposition reaches.

## 23. Extrapolation
Across all 20 models and 320 category-C rollouts there are 46 successes (by scale: 3, 11, 13, 19). **Every one of them has a training cube
within 14.85 mm** (median 12.8 mm), i.e. at the inner edge of the 3–15 mm extrapolation band. No replicate at any scale succeeded
substantially farther outside the demonstrated region, and the > 15 mm bin is 8/285 = 2.8% across the whole study.

**The conclusion is preserved: the policy interpolates locally and shows no evidence of robust extrapolation.**

## 24. Physical validity
1,120 primary rollouts, all enforcing physics-v2.
- **Successes with an invalid reason: 0** in every one of the 20 models.
- 5 rollouts exceeded a tolerance and were correctly rejected (max cube–table penetration 1.48 mm against the 1.0 mm limit); they count as
  failures.
- Max robot–table penetration across the study: 0.21 mm. Max pad–cube: 1.43 mm (limit 1.5).
- Every counted success is a valid opposing side pinch held 5 steps with the cube lifted.

## 25. Limitations
- **5 seeds per scale.** SD estimates from n = 5 are themselves uncertain, especially the TRAIN20 SD of 6.9.
- The within-scale predictor analyses have 5 models each; at 80 demos the training-scene diagnostic is constant (10/10), so its correlation
  is undefined there.
- Nested subsets keep dataset size and coverage correlated; the regression bounds the residual effect rather than removing the confound.
- One evaluation sampler seed per model (by design, to isolate training variance); the previous phase covered sampler variability.
- 56 positions, one task, one cube, fixed yaw, one camera; "generalization" means cube position only.
- Extrapolation spans only 3–15 mm beyond the workspace.
- The distance-response curve is fitted across scales pooled; per-scale curves would need more positions.

## 26. Deviations from pre-registration
- **None in design.** Seeds, budgets, benchmark, K, sampler seed, bins, statistics and the gate are exactly as registered. All 20 cells are
  in the analysis; no seed was retrained or dropped.
- One tooling fix during analysis: the distance-response figure crashed on a negative Wilson error bar (clipped at zero). No number changed.
- The analysis was run twice (before and after the figure fix); `summary.json` is from the final run.

## 27. Reproduction and hashes
```bash
# determinism check (bit-identical weights for a repeated seed)
.venv/bin/python -m training.audit_overfit --dataset artifacts/pickcube_physics_v2_rgb160 --output /tmp/det_a \
    --train-subset CLEAN10 --split-file docs/research/physics_v2_generalization_split.json --schedule cosine \
    --prediction x0 --padding hold --steps 201 --seed 1 --batch-size 8 --learning-rate 0.001 --device cuda \
    --eval-interval 100 --ema-decay 0.999

# the 4 x 5 matrix, training-scene diagnostic, held-out benchmark and offline metrics (resumable, <= 2 sim workers)
scripts/experiments/run_seed_replication.sh
scripts/experiments/run_seed_finish.sh
.venv/bin/python -m evaluation.seed_replication_analysis
env -u PYTHONPATH .venv/bin/python -m pytest -q tests/
```
Models: `artifacts/seed_replication/models/train{10,20,40,80}_seed{1,2,3,4}` (each with config, subset ids, seed, step count, loss history,
`last.pt`, `ema_last.pt`); seed-17 cells stay in `artifacts/exposure_matched_scaling/models/` and
`artifacts/physics_v2_generalization/models/TRAIN80_80k`. Hashes and the frozen benchmark: `docs/research/exposure_matched_scaling_freeze.json`
and `docs/artifact_manifest.json`. Results: `artifacts/seed_replication/{summary.json,RESULTS_DIGEST.txt}`.

## 28. Decision gate and recommendation
- **CASE A — the coverage result replicates.** Distance dominates (−2.51 per 10 mm); dataset size adds +0.06 [−0.22, +0.32] after
  conditioning. → the density sweep is the right next experiment.
- **CASE E — extrapolation remains absent.** 46/320 C successes, none beyond 14.85 mm from a demonstration.
- **CASE C — partially, at one scale only.** TRAIN20's SD of 6.9/56 (range 8–25) means single-run conclusions at that scale are unsafe;
  10, 40 and 80 demos are stable (SD 1.9–2.7), and 80 demos is perfectly stable on category A.
- **CASE D — not supported as stated.** The training-scene diagnostic is the better of the two predictors within a scale (+0.44 vs +0.15)
  and flagged the collapsed TRAIN20 run, but its CI includes zero at n = 5.
- **CASE B — rejected.** No independent dataset-size effect survives conditioning.

Recommended next, in order:
1. **The density sweep**, now properly motivated: hold demonstration count fixed and vary spacing directly, targeting the 4.5–13 mm range
   where §19 puts the 75%→25% transition. This is the experiment that converts "distance predicts success" into a causal statement.
2. **Report ≥ 3 seeds for any future claim at ≤ 20 demonstrations**, and at least 2 elsewhere; this study shows one run is not evidence at
   small data scales.
3. **Keep the training-scene diagnostic** as a cheap pre-test (10 rollouts) before spending a full benchmark on a checkpoint, while
   remembering it is not yet validated.
4. **Capacity scaling still has no support.** The 2M model is perfect and perfectly stable on dense interpolation across five seeds; the wall
   is coverage.

**Stopping here, as pre-registered. No density sweep, no capacity scaling, no new demonstrations.**
