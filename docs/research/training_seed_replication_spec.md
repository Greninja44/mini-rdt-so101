# Physics-v2 training-seed replication and variance study: pre-registered protocol

**Registered 2026-09-21, before any new model in this phase was trained and before any new closed-loop result was seen.** Nothing below is
changed after results appear; deviations are logged in the report.

## Questions
- **Primary:** how much closed-loop variance comes from TinyRDT training initialisation / optimisation randomness at each dataset scale?
- **Secondary:** after averaging over independent training seeds, does the previous conclusion still hold — that local demonstration distance,
  not dataset size, explains the apparent benefit of more data?
- **Tertiary:** is TRAIN40's sparse-region result (B = 14/20) reproducible, or was it a seed-specific optimisation outcome?

Also: whether larger datasets have higher or lower optimisation variance; whether training-scene execution predicts held-out success;
whether offline MAE predicts held-out success across training seeds; and whether particular held-out positions are consistently solved or
unsolved by independently trained policies.

## The independent variable is the training seed. Everything else is fixed.
physics-v2, the dataset, the TRAIN10/20/40/80 subsets, the 56-position held-out benchmark, the exposure definition and the derived budgets,
optimizer, learning rate, batch size, normalization methodology, frozen vision encoder, architecture, diffusion formulation, H = 16, K = 8,
success definition, penetration thresholds, camera, action representation, and the evaluation sampler seed. No hyperparameter tuning, no
added or removed demonstrations, no split regeneration, no DAgger, no corrective data, no vision fine-tuning, no spatial tokens, no
language, no world models, no sampler changes, and no capacity scaling.

## Frozen prior result
`docs/research/exposure_matched_scaling_freeze.json`: commit `c2526f5`, benchmark/dataset/split hashes, budgets, model, optimizer,
normalization and evaluation configs, the four existing seed-17 checkpoints with hashes, and the single-seed results
(15 / 17 / 26 / 28 of 56; training scenes 9 / 7 / 6 / 10 of 10). Those artifacts are read-only; this phase writes to
`artifacts/seed_replication/`.

## Hypotheses (predictions, reported whatever happens)
- **H1:** training-seed SD of held-out success is ≥ 3/56 at every scale, i.e. non-trivial relative to the 13/56 gap previously attributed to data scale.
- **H2:** TRAIN40 B = 14/20 does **not** replicate; the per-scale mean for B is well below 14/20.
- **H3:** conditioned on nearest-demo distance, the dataset-size coefficient remains near zero (CASE A of the gate below).
- **H4:** training-scene closed-loop success correlates with held-out success more strongly than offline held-out MAE does.
- **H5:** extrapolation stays absent: no replicate succeeds substantially farther outside the demonstrated region than the current 14.8 mm.

## Seeds
**Five independent training seeds per scale: 17, 1, 2, 3, 4** (20 models). Seed 17 is the existing exposure-matched run and is counted as one
replicate, because its procedure, budget and configuration are exactly those pre-registered here; only the 16 missing models are trained.

Each seed sets, inside `training/audit_overfit.py`: Python `random`, NumPy, and the torch global RNG (parameter initialisation) via
`set_seed(seed)`; the batch-order stream (`seed + 100`); the diffusion-noise stream (`seed + 200`); and the state-dropout stream
(`seed + 300`, unused here). TF32 is disabled and cuDNN benchmarking/autotuning is off. Data loading is deterministic (features are
pre-extracted once, no shuffling dataloader). The evaluation sampler seed stays 0 for every model, so training variance is never mixed with
inference sampling variance.

## Reproducibility check (before the matrix)
One scale and one seed are trained twice for a short run under identical settings, comparing initial loss, the logged metric rows, and the
checkpoint metadata. Remaining nondeterminism is documented rather than eliminated: no effort is spent forcing bit-exact CUDA kernels.

## Training matrix
Every cell uses the pre-registered exposure-matched budget, unchanged:

| scale | windows | steps | exposure |
|---|---|---|---|
| TRAIN10 | 544 | 10,055 | 147.868 |
| TRAIN20 | 1,084 | 20,037 | 147.875 |
| TRAIN40 | 2,165 | 40,018 | 147.873 |
| TRAIN80 | 4,328 | 80,000 | 147.874 |

- No early stopping, no run extended, no seed retrained, no seed dropped. **Every pre-registered seed stays in the analysis regardless of its
  result.** A poor model is a result, not a failed run; only genuine infrastructure failures (crash, OOM, power loss) may be restarted from
  the same definition, and they are logged separately.
- Execution order is seed-major (all four scales of one seed, then the next), so that if the machine is interrupted, complete replicates
  exist rather than partial rows. If fewer than five seeds are complete when the report is written, the report states exactly which cells
  exist; three seeds per scale is the documented minimum fallback.
- Checkpoints are named `train{10,20,40,80}_seed{S}` and store the raw and EMA weights, config, subset ids, dataset and split hashes, seed,
  step count, exposure, logs and loss history, plus a machine-readable manifest.

## Evaluation
1. **Training-scene diagnostic (every model):** closed loop on the 10 CLEAN10 scenes, which lie inside every subset's training data.
   K = 8, policy seed 0, max 150 steps.
2. **Held-out (every model):** the identical frozen 56 expert-valid positions (A 20, B 20, C 16). **Primary: K = 8, policy seed 0**, max 150
   steps, DDIM-10.
3. **Offline (every model):** train, validation and held-out MAE, per joint, episode start and phase-wise.
4. Physics-v2 validity is enforced unchanged; invalid rollouts cannot count as successes and are reported separately.
5. **Sampler replication is not repeated** for all 20 models (the previous phase covered it). Extra sampler seeds, if used at all, are
   limited to boundary positions or models with extreme results, and are labelled secondary.

## Metrics and statistics
- **Primary:** held-out successes / 56 per model; per scale the mean, SD, median, min, max and a bootstrap 95% CI. **Individual seeds are
  always plotted, never only means.**
- A/B/C reported separately at every scale, with the same per-seed spread.
- **Position stability:** for each of the 56 positions and each scale, the success frequency across seeds (0/5 … 5/5), compared against the
  nearest-demo distance. No "easy/hard" labels beyond those frequencies.
- **TRAIN40 anomaly:** a 20 × 5 position × seed matrix for category B, with per-position frequencies.
- **Repeated measures are respected.** Rollouts sharing a position are not treated as independent. The coverage model is fitted on pooled
  rollouts with a **two-way cluster bootstrap** that resamples held-out positions and training seeds; success is modelled as a function of
  log2(dataset size) and nearest-demo distance, fitted with and without the distance term. Raw counts accompany every model-based number.
- **Scale comparisons** use per-position success frequencies (out of 5 seeds) paired across scales, tested by a permutation test on the paired
  differences.
- **Variance decomposition:** the share of outcome variance attributable to position versus training seed, per scale.
- **Distance response:** the pre-registered bins 0–2.5, 2.5–5, 5–7.5, 7.5–10, 10–15, > 15 mm, unchanged, reporting trials, successes, rate and
  CI per bin, aggregated over seeds. A logistic fit of success against distance gives the distances at model-predicted 90 / 75 / 50 / 25 / 10%
  success, **with uncertainty and without calling any distance a hard generalization radius**.
- **Predictor comparison:** across the 20 models, correlations (with bootstrap CIs) between held-out success and each of training MAE,
  validation MAE, held-out MAE, and training-scene success. No seed is selected or removed on the basis of these.

## Decision gate
Classified at the end; several may apply.
- **CASE A:** the coverage result replicates — distance dominates, dataset size adds little after conditioning → next, the density sweep.
- **CASE B:** dataset size stays independently predictive after conditioning on distance → investigate what diversity adds.
- **CASE C:** training variance is large and scale conclusions are unstable → investigate optimisation stability first.
- **CASE D:** the training-scene diagnostic predicts held-out success much better than offline MAE → adopt it as a pre-test diagnostic, but
  **do not** retroactively remove poor seeds here.
- **CASE E:** extrapolation remains absent.

## Stop condition
Stop after: the planned seeds are trained; training-scene diagnostics complete; all models evaluated on the identical 56 positions; training
variance quantified; the coverage-conditioned analysis repeated; the TRAIN40 anomaly analysed; extrapolation analysed;
`TRAINING_SEED_REPLICATION_REPORT.md` written; manifests, checkpoints and figures committed.

**Do not** run the density sweep, scale capacity, collect demonstrations, add world models, change the architecture, add language, or start
new task families.
