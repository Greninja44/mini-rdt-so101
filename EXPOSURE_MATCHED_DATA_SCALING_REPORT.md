# Physics-v2 exposure-matched data scaling: is it the data, or the coverage?

**Answer: almost entirely the coverage.** With optimisation exposure held constant, held-out success rises from 15/56 to 28/56 between 10 and
80 demonstrations (p = 0.001) — but once the distance to the nearest training cube is accounted for, the residual effect of demonstration
count is **zero within error** (logistic coefficient −0.05 per doubling, 95% CI [−0.35, +0.21]). Adding demonstrations helps because it puts
demonstrations closer to where the cube actually is, not because the model benefits from a larger or more diverse dataset. This is **CASE B**,
with **CASE E** (extrapolation never improves) and a partial **CASE D** (matched exposure did not equalise closed-loop trainability).

Pre-registration: `docs/research/exposure_matched_data_scaling_spec.md` (commit `5836929`, before any model here was trained).
Frozen prior benchmark: `docs/research/physics_v2_generalization_freeze.json` (commit `db17d4f`).
Raw numbers: `artifacts/exposure_matched_scaling/summary.json`, `RESULTS_DIGEST.txt`.

---

## 1. Research question
**Primary:** with optimisation exposure controlled, how does demonstration count affect spatial generalization of the same ~2M TinyRDT?
**Secondary:** does more data improve dense interpolation, fill sparse regions, or enable extrapolation?
This phase concerns data scale only. No model capacity was changed, and none follows.

## 2. Pre-registration
The spec fixed the hypotheses, the exposure equation, the derived budgets, the nested subsets, the training-scene diagnostic, the frozen
benchmark, primary K = 8, the statistics (Wilson, exact McNemar, cluster-bootstrapped logistic), the **distance bins**, and the decision gate
— all before training. Its predictions were H1 monotone scaling, H2 gains concentrated where coverage improves, H3 little residual effect
after controlling distance, H4 offline MAE remains a poor predictor. **H1 holds overall but not per category; H2, H3 and H4 hold.**

## 3. Frozen previous benchmark
Commit `db17d4f`; benchmark = the same **56 expert-valid held-out positions** (A 20 dense interpolation, B 20 sparse interpolation inside a
30 mm hole cut out of training, C 16 extrapolation 3–15 mm outside the workspace), with the dataset, split and checkpoint hashes recorded.
Positions were not regenerated, moved or filtered, and no test failure entered training.

## 4. Exposure definition
A training window is one observation → 16-action chunk, one per frame. Windows are sampled uniformly with replacement, so exposure is
**passes over the subset's own windows**:

```
E = optimizer_steps × batch_size / training_windows        (batch_size = 8 everywhere)
```

Anchor: the existing 80-demo / 80,000-step model, reused unchanged → **E = 80,000 × 8 / 4,328 = 147.874 passes**.

## 5. Actual training-window counts
Measured from the datasets, not assumed — episode lengths differ:

| subset | episodes | windows | mean episode length |
|---|---|---|---|
| TRAIN10 (= CLEAN10 ids) | 10 | 544 | 54.4 |
| TRAIN20 | 20 | 1,084 | 54.2 |
| TRAIN40 | 40 | 2,165 | 54.1 |
| TRAIN80 | 80 | 4,328 | 54.1 |

## 6. Derived optimizer budgets

| model | steps = round(80,000 × W / 4,328) | resulting E | total sampled windows | action targets |
|---|---|---|---|---|
| TRAIN10_e | **10,055** | 147.868 | 80,440 | 1.29M |
| TRAIN20_e | **20,037** | 147.875 | 160,296 | 2.56M |
| TRAIN40_e | **40,018** | 147.873 | 320,144 | 5.12M |
| TRAIN80_e | **80,000** (anchor, reused) | 147.874 | 640,000 | 10.2M |

All exposures agree to within 0.01 passes (0.005%). The naive 10k/20k/40k/80k schedule happens to be close **only because episode lengths
are near-uniform here**; the budgets were still derived from measured counts, as pre-registered. Note the old TRAIN20 checkpoint (20,000
steps, E = 147.60) was 0.19% short of target and was **retrained** rather than reused.

## 7. Nested dataset construction
`TRAIN10 ⊂ TRAIN20 ⊂ TRAIN40 ⊂ TRAIN80`, reused unchanged from the generalization split (farthest-point sampling from CLEAN10), so each step
up in scale **adds** spatial coverage instead of replacing samples. No held-out position appears in any subset.

## 8. Spatial coverage (measured before training)
`docs/assets/v2_scaling_coverage.png`, `artifacts/exposure_matched_scaling/coverage.json`. Distance from each of the 56 test positions to the
nearest training cube:

| subset | median | mean | min | max | within 2.5 / 5 / 7.5 / 10 / 15 mm | A / B / C median |
|---|---|---|---|---|---|---|
| TRAIN10 | 17.7 | 19.5 | 3.1 | 52.8 | 0 / 2 / 5 / 9 / 22 | 13.6 / 19.6 / 19.2 |
| TRAIN20 | 11.4 | 12.2 | 3.1 | 21.3 | 0 / 3 / 15 / 23 / 46 | 7.2 / 12.5 / 14.7 |
| TRAIN40 | 9.5 | 10.4 | 3.1 | 21.3 | 0 / 10 / 24 / 30 / 49 | 5.5 / 10.5 / 14.5 |
| TRAIN80 | 7.2 | 8.8 | 2.2 | 21.3 | 2 / 15 / 29 / 34 / 50 | 4.3 / 10.5 / 13.3 |

**Data count and local coverage are entangled by construction** — which is why §16–18 do the real work.

## 9. Training configuration
Identical for all four: TinyRDT hidden 192, 4 layers, 6 heads, H = 16, frozen pooled MobileNet encoder, cosine schedule, x0 prediction,
hold-last-action padding, AdamW lr 1e-3 / wd 1e-4, grad clip 1.0, batch 8, EMA 0.999, fp32, **training seed 17**, final EMA weights, no early
stopping, no checkpoint selection. Normalization is computed per subset from its own windows.

## 10. Training curves
Final EMA training MAE: 0.0077 (10), 0.0060 (20), 0.0050 (40), 0.0072 (80). All four curves decrease monotonically with no sign of
overfitting. Note the non-monotonicity: at matched exposure the 40-demo model reaches the *lowest* training error and the 80-demo model does
not — more windows at equal passes means fewer gradient updates per window pattern.

## 11. Training-scene closed-loop diagnostic
Closed loop on scenes **inside each model's own training data** (K = 8, seed 0). The primary set is the 10 CLEAN10 scenes, which belong to
every subset:

| model | CLEAN10 scenes (common) | own diagnostic scenes | offline train MAE |
|---|---|---|---|
| TRAIN10_e | **9/10** | 9/10 | 0.0077 |
| TRAIN20_e | **7/10** | 13/20 | 0.0060 |
| TRAIN40_e | **6/10** | 13/20 | 0.0050 |
| TRAIN80_e | **10/10** | 18/20 | 0.0072 |

**Matched exposure did not equalise closed-loop competence on training data** (partial CASE D). The 20- and 40-demo models cannot reliably
execute scenes they were trained on, while having *better* offline training error than the 80-demo model. Per the protocol this is reported
as a flag, not repaired: no budget was changed after seeing any held-out result.

## 12. Offline metrics

| model | train | validation | held-out test | test episode-start | test phase (DESCEND / CLOSE / LIFT) |
|---|---|---|---|---|---|
| TRAIN10_e | 0.0077 | 0.0291 | 0.0357 | 0.0345 | 0.048 / 0.042 / 0.041 |
| TRAIN20_e | 0.0060 | 0.0087 | 0.0137 | 0.0246 | 0.019 / 0.013 / 0.012 |
| TRAIN40_e | 0.0050 | 0.0064 | **0.0101** | 0.0171 | 0.014 / 0.010 / 0.009 |
| TRAIN80_e | 0.0072 | 0.0072 | **0.0101** | 0.0255 | 0.014 / 0.010 / 0.009 |

TRAIN40_e and TRAIN80_e are offline-identical to three decimals, yet differ sharply in closed loop, in opposite directions per category
(§13). Treat these as diagnostics only.

## 13. Held-out results by category (primary: K = 8, seed 0, 56 positions)

| model | A interpolation (20) | B sparse (20) | C extrapolation (16) |
|---|---|---|---|
| TRAIN10_e | 11 (55%) | 2 (10%) | 2 (12%) |
| TRAIN20_e | 12 (60%) | 2 (10%) | 3 (19%) |
| TRAIN40_e | 10 (50%) | **14 (70%)** | 2 (12%) |
| TRAIN80_e | **20 (100%)** | 4 (20%) | 4 (25%) |

**Per category the curve is not monotone.** The 40-demo model is the best on the sparse hole band by a wide margin (14/20 vs 4/20 for the
80-demo model) while being the *worst* on dense interpolation (10/20). Since both models have the same median nearest-demo distance in B
(10.5 mm), this difference is not explained by coverage — see §18 and §25.

## 14. Total held-out results

| model | demos | steps | success | Wilson 95% CI |
|---|---|---|---|---|
| TRAIN10_e | 10 | 10,055 | **15/56** (26.8%) | 17.0–39.6% |
| TRAIN20_e | 20 | 20,037 | **17/56** (30.4%) | 19.9–43.3% |
| TRAIN40_e | 40 | 40,018 | **26/56** (46.4%) | 34.0–59.3% |
| TRAIN80_e | 80 | 80,000 | **28/56** (50.0%) | 37.3–62.7% |

## 15. The exposure-matched scaling curve
`docs/assets/v2_scaling_curve.png` (left panel). Overall success rises monotonically, 15 → 17 → 26 → 28, but the CIs of adjacent scales
overlap heavily and only the 8× endpoint comparison is significant (§20). The curve is shown raw and unsmoothed.

For contrast, the previous fixed-step budget gave 13 → 17 → 13 → 9. **That curve was an artifact of shrinking exposure**, and this phase
replaces it.

## 16. Success vs nearest-demonstration distance
Pooled over all 4 × 56 = 224 model-position rollouts, logistic regression with a position-cluster bootstrap:

| model | coefficient | 95% CI |
|---|---|---|
| distance only | **−1.93 per 10 mm** | [−2.55, −1.38] |
| demos only | **+0.371 per doubling** | [+0.21, +0.56] |
| **both** — distance | **−1.97 per 10 mm** | [−2.85, −1.31] |
| **both** — log2(demos) | **−0.05 per doubling** | [−0.35, +0.21] |

Taken alone, doubling the dataset looks beneficial. **Conditioned on the nearest-demo distance, that effect vanishes** (the point estimate is
even slightly negative, with the CI spanning zero). Distance keeps essentially its full magnitude.

## 17. Distance-bin analysis (bins fixed in the pre-registration)

| nearest demo | TRAIN10_e | TRAIN20_e | TRAIN40_e | TRAIN80_e |
|---|---|---|---|---|
| 0–2.5 mm | — | — | — | 2/2 |
| 2.5–5 mm | 2/2 | 1/3 | 6/10 | 11/13 |
| 5–7.5 mm | 2/3 | 6/12 | 5/14 | 8/14 |
| 7.5–10 mm | 4/4 | 4/8 | 5/6 | 2/5 |
| 10–15 mm | 6/13 | 5/23 | 9/19 | 5/16 |
| > 15 mm | 1/34 | 1/10 | 1/7 | 0/6 |

Every model degrades with distance, and every model is near-hopeless beyond 15 mm (3/57 pooled). Several bins hold few positions per model,
so these are raw counts and no bin-level inference is drawn.

## 18. Matched-distance comparison
Reading §17 **across rows** answers the key question: at comparable local coverage, does more data help?

- 2.5–5 mm: 100% (2/2), 33% (1/3), 60% (6/10), 85% (11/13) — no consistent ordering, two cells tiny.
- 5–7.5 mm: 67%, 50%, 36%, 57% — the 40-demo model is worst.
- 7.5–10 mm: 100% (4/4), 50%, 83%, 40% — the 80-demo model is worst.
- 10–15 mm: 46%, 22%, 47%, 31% — non-monotone.

**There is no systematic advantage for larger datasets at matched distance**, which is exactly what the regression in §16 reports. What
variation remains looks like per-run training variability (§25), not a data-scale effect.

## 19. Interpolation vs extrapolation (kept separate)
- **Dense interpolation (A):** 11 → 12 → 10 → **20**. Only the largest, best-converged model solves it completely.
- **Sparse interpolation (B):** 2 → 2 → **14** → 4. Non-monotone and dominated by which model happened to train well, not by data count.
- **Extrapolation (C):** 2 → 3 → 2 → 4. **Flat.** Across all four models and 64 extrapolation rollouts there are 11 successes, every one of them
  with a training cube within 14.8 mm (median 12.8 mm) — i.e. only at the inner edge of the extrapolation band. Nothing here suggests more demonstrations teach extrapolation beyond the training support (**CASE E**).

No combined "generalizes better everywhere" claim is made.

## 20. Statistical comparisons (exact McNemar, identical positions)

| comparison | successes | gained by larger | lost by larger | p (two-sided) |
|---|---|---|---|---|
| 20 vs 10 demos | 15 → 17 | 11 | 9 | 0.82 |
| 40 vs 20 demos | 17 → 26 | 18 | 9 | 0.12 |
| 80 vs 40 demos | 26 → 28 | 12 | 10 | 0.83 |
| **80 vs 10 demos** | 15 → 28 | **14** | **1** | **0.00098** |

Only the 8× endpoint difference is significant. Note the churn: even between models differing by 2 successes, 20+ positions change outcome,
so small differences between adjacent scales carry no information.

**Sampler-seed control** (seeds 1 and 2 on the 38 discordant positions): per-seed successes were 12/8/12 (TRAIN10_e), 14/19/16 (TRAIN20_e),
23/23/26 (TRAIN40_e), 25/27/24 (TRAIN80_e). Positions flipping across seeds: 7, 16, 11, **3**. Sampling noise is modest and *shrinks* with
better-converged models, so the category effects in §13 are properties of the trained models, not of diffusion sampling.

## 21. Offline MAE vs closed-loop success
`docs/assets/v2_scaling_curve.png` (right panel). Across the four exposure-matched models the relationship is weak and, in the crucial
region, absent:
- TRAIN40_e and TRAIN80_e have **identical** held-out offline MAE (0.0101) and differ by 10 percentage points overall, in opposite
  directions per category (A 10/20 vs 20/20, B 14/20 vs 4/20).
- TRAIN40_e has the **best** training-set offline error (0.0050) and the **worst** training-scene closed-loop success (6/10).
- Only the 10-demo model, whose offline error is 3.5× larger, is separated by this metric.

Combined with the previous phase (identical offline error, 7/20 vs 18/20 on training scenes), **offline action MAE should not be used for
model selection in this project** (H4 supported).

## 22. Failure breakdown (primary K = 8, seed 0)

| outcome | TRAIN10_e | TRAIN20_e | TRAIN40_e | TRAIN80_e |
|---|---|---|---|---|
| success | 15 | 17 | 26 | 28 |
| wrong_lateral_alignment | 39 | 35 | 29 | 24 |
| early_close | 0 | 0 | 1 | 3 |
| wrong_height | 0 | 2 | 0 | 0 |
| cube_nudge_displacement | 1 | 1 | 0 | 1 |
| cube_tilt | 1 | 1 | 0 | 0 |

One failure mode dominates at every data scale: **the gripper closes in the wrong place**, and more data simply reduces how often that
happens (39 → 24). Median lateral offset at the first close, successes vs failures: 10.3/30.1 mm (10 demos), 9.0/20.5 (20), 10.2/17.8 (40),
6.5/20.0 (80), against an expert range of 0.9–2.0 mm. The failure *mechanism* never changes with data scale; only its frequency does.

## 23. Physical validity
All 224 primary rollouts plus the secondary-seed rollouts (228 in this phase) enforce physics-v2.

| model | successes | with invalid reason | invalid-physics rollouts | max robot–table | max cube–table | max pad–cube |
|---|---|---|---|---|---|---|
| TRAIN10_e | 15 | 0 | 0 | 0.002 mm | 0.67 mm | 1.32 mm |
| TRAIN20_e | 17 | 0 | 0 | 0.00 mm | 0.74 mm | 1.28 mm |
| TRAIN40_e | 26 | 0 | 0 | 0.00 mm | 0.73 mm | 1.20 mm |
| TRAIN80_e | 28 | 0 | 0 | 0.00 mm | 0.73 mm | 1.43 mm |

Zero invalid successes; every counted success is a valid opposing side pinch held 5 steps with the cube lifted, inside the 1.0/1.0/1.5 mm
tolerances. Two successes (one each for the 10- and 20-demo models) had a transient pre-grasp vertical pad contact, as in earlier phases.

## 24. Deviations from protocol
- **None affecting the design.** Budgets, subsets, benchmark, primary K, bins and statistics are exactly as pre-registered.
- The secondary sampler seeds were run on the 38 discordant positions, which is the pre-registered conditional branch.
- The analysis ran once before the secondary seeds existed and once after; the primary numbers use seed 0 only, and the seed files are kept
  separate (`_s1`, `_s2` suffixes) so they cannot leak into the primary metric.

## 25. Limitations
- **One training seed per data scale.** This is the biggest limitation. The non-monotone category results (B: 14/20 at 40 demos vs 4/20 at
  80) cannot be separated from run-to-run training variance, and the sampler-seed control (§20) shows the variance is *not* in sampling, so
  it must be in training. A per-scale seed replication is the missing control.
- **Matched exposure did not produce matched trainability** (§11), so part of the remaining differences reflect optimisation, not data.
- 56 positions, 20/20/16 per category; several distance bins hold under 5 positions per model.
- Coverage and count cannot be fully disentangled in a nested design: the regression conditions on distance, but distance is estimated from
  the same 56 positions.
- One task, one cube, fixed yaw, one camera, no distractors; "generalization" means cube position only.
- Extrapolation spans only 3–15 mm beyond the workspace.
- Rendering is CPU-bound (~1 s per simulated step), which is why this phase is 452 rollouts rather than thousands.

## 26. Exact reproduction
```bash
# 0. coverage of the frozen benchmark by each nested subset (before training)
.venv/bin/python -m evaluation.coverage_analysis

# 1. exposure-matched training (budgets derived from measured window counts)
.venv/bin/python -m training.audit_overfit --dataset artifacts/pickcube_physics_v2_rgb160 \
    --output artifacts/exposure_matched_scaling/models/TRAIN10_e --train-subset CLEAN10 \
    --split-file docs/research/physics_v2_generalization_split.json \
    --schedule cosine --prediction x0 --padding hold --steps 10055 --seed 17 --batch-size 8 \
    --learning-rate 0.001 --device cuda --eval-interval 2000 --ema-decay 0.999 --snapshot-interval 4000
#   TRAIN20_e: --train-subset TRAIN20 --steps 20037 ; TRAIN40_e: --train-subset TRAIN40 --steps 40018
#   TRAIN80_e: reused unchanged from artifacts/physics_v2_generalization/models/TRAIN80_80k (80,000 steps)

# 2. training-scene diagnostic and held-out benchmark (K=8, seed 0)
.venv/bin/python -m evaluation.closed_loop --policy tinyrdt --checkpoint <ckpt> \
    --dataset artifacts/pickcube_physics_v2_rgb160 --episodes 0 2 3 4 5 6 7 8 9 11 --k 8 --max-steps 150 \
    --skip-replay --no-media --output artifacts/exposure_matched_scaling/train_scenes/<model>
.venv/bin/python -m evaluation.closed_loop --policy tinyrdt --checkpoint <ckpt> \
    --dataset artifacts/pickcube_physics_v2_gen_test_rgb160 --episodes <56 ids> --k 8 --max-steps 150 \
    --workspace-margin 0.015 --skip-replay --no-media --output artifacts/exposure_matched_scaling/closed_loop/<model>

# everything end to end (resumable, <= 2 simulation workers), then analysis:
scripts/experiments/run_exposure_scaling.sh
scripts/experiments/run_exposure_finish.sh     # secondary seeds on discordant positions + analysis + digest
.venv/bin/python -m evaluation.scaling_analysis
env -u PYTHONPATH .venv/bin/python -m pytest -q tests/
```

## 27. Artifacts and hashes
- Report commit: see `git log` for this file; pre-registration `5836929`; frozen prior benchmark `db17d4f`.
- Checkpoints (`ema_last.pt`, sha256 prefix): TRAIN10_e `6a3bf87cd5387bfe`, TRAIN20_e `de9bfdbf0dde44dc`,
  TRAIN40_e `5b9ea7584ce49dc5`, TRAIN80_e (anchor, reused) `e26f2f9ae8151f18`.
- Benchmark, dataset and split hashes: `docs/research/physics_v2_generalization_freeze.json`.
- Results: `artifacts/exposure_matched_scaling/{summary.json,RESULTS_DIGEST.txt,coverage.json}`; rollouts under `closed_loop/`,
  `train_scenes/`; models and logs under `models/`. Manifest: `docs/artifact_manifest.json`.

## 28. Decision gate and recommendation
**CASE B (dominant):** the data-scale benefit is explained by reduced nearest-demo distance; the residual coefficient for doubling the
dataset is −0.05 [−0.35, +0.21].
**CASE E:** interpolation improves, extrapolation does not (2 → 3 → 2 → 4 of 16).
**CASE D (partial):** equal exposure did not equalise closed-loop trainability — the 20- and 40-demo models fail scenes they were trained on
(7/10 and 6/10), so their held-out numbers are partly an optimisation result.
**Not CASE A**, and not CASE C either: data does help, but only through coverage.

Recommended next, in order:
1. **Seed replication before anything else.** Three training seeds per data scale at the same exposure-matched budgets, evaluated at K = 8 on
   the same 56 positions. §25 makes this the highest-value missing control: without it the B-category inversion (14/20 vs 4/20) cannot be
   attributed. Cost ≈ 8 training runs plus 8 × 56 rollouts.
2. **Then the density sweep** (explicitly deferred by this phase's protocol): hold demonstration count fixed and vary *spacing* directly, to
   test the ~5 mm threshold that §17 implies, independently of dataset size.
3. **Fix the trainability confound:** since matched exposure did not give matched training-scene competence, either report a convergence
   criterion per run (e.g. train to a fixed training-scene success) or match total updates as a second arm.
4. **Only afterwards** ask about capacity. The 2M model reaches 20/20 on dense interpolation and 0/6 beyond 15 mm — a coverage wall, not an
   obvious capacity wall.

**Stopping here, as pre-registered. The density sweep was not started and no capacity scaling was run.**
