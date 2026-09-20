# Physics-v2 exposure-matched data scaling: pre-registered protocol

**Registered 2026-09-20, before any model in this phase was trained and before any new closed-loop result was seen.** Nothing below is
changed after results appear. Deviations are logged in the report.

## Question
**Primary:** with optimisation exposure held constant, how does the number of demonstrations affect the spatial generalization of the same
~2M TinyRDT?
**Secondary:** does adding demonstrations mainly improve dense interpolation, fill sparse regions, or enable extrapolation?

This phase is about **data scale**. It is not about model scale. No capacity scaling follows it.

## What is fixed
physics-v2, the SO-101 model, camera, success definition, side-grasp validity, penetration tolerances, TinyRDT architecture, the frozen
vision encoder, the diffusion formulation, H = 16, AdamW (lr 1e-3, wd 1e-4, grad clip 1.0), batch size 8, EMA 0.999, fp32, the normalization
methodology (per-subset statistics over that subset's own windows), the 56-position held-out benchmark, primary K = 8, and the success
criteria. No architecture change, no vision fine-tuning, no spatial tokens, no DAgger, no corrective data, no world model, no K tuning, no
change to held-out positions, and no test failure ever enters training.

## Frozen prior result
`docs/research/physics_v2_generalization_freeze.json` records commit `db17d4f`, the benchmark and dataset aggregate hashes, the split hash,
all five existing checkpoint hashes, model/diffusion/optimizer configs, and the PR #5 results. Those artifacts are read-only; this phase
writes to `artifacts/exposure_matched_scaling/`.

## Hypotheses (predictions; reported whatever happens)
- **H1:** with exposure matched, held-out success increases monotonically with demonstration count (10 → 20 → 40 → 80).
- **H2:** the gain is concentrated in categories whose nearest-demo distance shrinks most, i.e. mostly A and partly B, with C roughly flat.
- **H3:** after controlling for nearest-demo distance, the residual effect of demonstration count is small (i.e. CASE B rather than CASE A).
- **H4:** offline MAE stays a poor predictor of closed-loop success across data scales.

## Exposure definition (the core of this phase)
A training window is one observation → 16-action chunk, one per recorded frame, hold-padded at the end of an episode. Windows are sampled
uniformly with replacement, so the natural measure of optimisation exposure is **passes over the subset's own windows**:

```
E = optimizer_steps × batch_size / training_windows
```

Batch size is fixed at 8 for every run, so E is the only quantity equalised. **Episode lengths differ between subsets, so the budgets are
derived from the measured window counts, not assumed.**

Anchor: the existing 80-demo / 80,000-step model, which is reused unchanged. E = 80,000 × 8 / 4,328 = **147.874 passes**.

| subset | episodes | windows (measured) | required steps = round(80,000 × W / 4,328) | resulting E | total sampled windows |
|---|---|---|---|---|---|
| TRAIN10 (= CLEAN10 ids) | 10 | 544 | **10,055** | 147.868 | 80,440 |
| TRAIN20 | 20 | 1,084 | **20,037** | 147.875 | 160,296 |
| TRAIN40 | 40 | 2,165 | **40,018** | 147.873 | 320,144 |
| TRAIN80 | 80 | 4,328 | **80,000** (anchor, reused) | 147.874 | 640,000 |

All four exposures agree to within 0.01 passes (0.005%). Action-target exposure is identical in form: each sampled window contributes
16 action tokens, so total action-target exposure = 16 × 8 × steps, in the same ratio as E.

**Checkpoint reuse rule:** a checkpoint is reused only if its recipe and budget exactly satisfy this definition. Only `TRAIN80_80k`
qualifies. The existing 20k-step models are **retrained**, including TRAIN20 whose old budget (E = 147.60) is 0.19% below target; equality
is not approximated.

## Datasets
The nested subsets from the previous phase are reused unchanged (`docs/research/physics_v2_generalization_split.json`):
**TRAIN10 ⊂ TRAIN20 ⊂ TRAIN40 ⊂ TRAIN80**, built by farthest-point sampling from CLEAN10, so larger sets add spatial coverage instead of
replacing samples. No held-out position is in any of them.

Coverage of the 56 benchmark positions, measured before training (`evaluation/coverage_analysis.py`,
`docs/assets/v2_scaling_coverage.png`):

| subset | median | mean | min | max | within 2.5 / 5 / 7.5 / 10 / 15 mm | A / B / C median |
|---|---|---|---|---|---|---|
| TRAIN10 | 17.7 | 19.5 | 3.1 | 52.8 | 0 / 2 / 5 / 9 / 22 | 13.6 / 19.6 / 19.2 |
| TRAIN20 | 11.4 | 12.2 | 3.1 | 21.3 | 0 / 3 / 15 / 23 / 46 | 7.2 / 12.5 / 14.7 |
| TRAIN40 | 9.5 | 10.4 | 3.1 | 21.3 | 0 / 10 / 24 / 30 / 49 | 5.5 / 10.5 / 14.5 |
| TRAIN80 | 7.2 | 8.8 | 2.2 | 21.3 | 2 / 15 / 29 / 34 / 50 | 4.3 / 10.5 / 13.3 |

Demonstration count and local coverage move together by construction, which is exactly why §"coverage-aware analysis" below is the decisive
part of this phase.

## Training
Identical recipe for all four runs: cosine schedule, x0 prediction, hold-last-action padding, frozen pooled MobileNet encoder, hidden 192,
4 layers, 6 heads, H = 16, AdamW lr 1e-3, batch 8, EMA 0.999, **training seed 17** for every run (the anchor's seed), final EMA weights, no
early stopping and no checkpoint selection. Saved per run: config, exact subset ids, normalization statistics, loss/metric curves,
`last.pt`, `ema_last.pt`, and EMA snapshots every 4,000 steps.

## Training-scene closed-loop diagnostic (before held-out evaluation)
The previous phase showed offline error can look converged while closed-loop skill is under-trained. Each model is therefore first run in
closed loop on scenes **inside its own training data**.
- **Primary diagnostic set: the 10 CLEAN10 scenes**, which belong to every subset, so all four models are directly comparable. K = 8,
  policy seed 0, max 150 steps.
- **Secondary:** the 20 TRAIN20 scenes for the three models that contain them (the existing TRAIN80_80k number, 18/20, is reused).
- A model with good offline MAE that cannot solve its own training scenes is flagged as closed-loop under-trained, and that flag is reported
  with its held-out numbers rather than used to grant it extra budget. **No budget is ever changed after seeing held-out results**; the
  budgets above are fixed by the exposure equation.

## Held-out evaluation
The identical frozen 56 expert-valid positions (A 20, B 20, C 16), never regenerated, moved or filtered.
- **Primary: K = 8, policy seed 0, max 150 steps, DDIM-10**, for all four models.
- Success requires physics-v2 validity: a valid opposing side pinch held 5 steps, contact normals within tolerance, cube lifted to
  z ≥ 0.10 m, and no penetration tolerance exceeded. Invalid rollouts cannot count as successes; the existing rejection logic is untouched.
- Rollouts are deterministic given (checkpoint, position, policy seed, sampler), as verified previously, so identical conditions are never
  re-run.
- **Secondary sampler seeds:** after the primary comparison, seeds 1 and 2 are run for the three retrained models on **discordant
  positions** — those where the four models do not all agree at seed 0. This measures sampling variability where it can matter; it never
  replaces the primary number. Seeds are never re-run until a desired outcome appears.

## Metrics
- **Primary metric:** held-out success at K = 8, seed 0, per model, as raw counts out of 56, with Wilson 95% CIs, reported overall and split
  A / B / C.
- **Secondary:** per-category curves, training-scene diagnostic success, offline MAE (train / validation / held-out, per-joint, episode
  start, phase-wise), failure taxonomy, steps to success, and physical-validity counters.

## Statistics
- Wilson 95% CIs for every proportion; raw counts always shown.
- Paired exact McNemar on identical positions for the adjacent pairs (10 vs 20, 20 vs 40, 40 vs 80) and for 10 vs 80, with discordant counts
  and effect sizes.
- **Distance bins fixed in advance: 0–2.5, 2.5–5, 5–7.5, 7.5–10, 10–15, > 15 mm.** Bin edges are never adjusted after seeing results; bins
  with few positions are reported as raw counts without inference.
- **Coverage-aware (the CASE A/B discriminator):** logistic regression of success on (a) nearest-demo distance alone, (b) log2(demos) alone,
  and (c) both, pooled over the 4 × 56 model-position rollouts, with a position-cluster bootstrap 95% CI on each coefficient. The
  coefficient on log2(demos) **after** conditioning on distance is the residual data-scale effect.
- **Matched-distance comparison:** within each pre-defined distance bin, compare success across data scales. Reported as raw counts; no
  claim is made where matched samples are too few.
- Interpolation, sparse interpolation and extrapolation are reported separately and never merged into a single "generalizes better" claim.

## Decision gate
Classified at the end, possibly multiple cases:
- **CASE A:** success rises with demonstration count even after controlling exposure and distance → dataset diversity helps beyond local coverage.
- **CASE B:** the rise is explained almost entirely by reduced nearest-demo distance → local density dominates.
- **CASE C:** little benefit despite matched exposure → investigate representation/control before more data.
- **CASE D:** larger datasets still fail their own training scenes → the exposure definition or recipe is insufficient, and held-out numbers
  are not a data-scaling result.
- **CASE E:** interpolation improves while extrapolation stays poor → local interpolation without broad spatial extrapolation.

## Stop condition
Stop after: four exposure-matched models trained; training-scene diagnostics complete; all four evaluated on the identical 56-position
benchmark; coverage-aware analysis and statistics complete; `EXPOSURE_MATCHED_DATA_SCALING_REPORT.md` written; artifacts and manifest
updated.

**Do not** run the density sweep, scale model capacity, add world models, or start new ablations.
