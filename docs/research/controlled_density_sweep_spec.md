# Physics-v2 controlled demonstration-density sweep: pre-registered protocol

**Registered 2026-09-21, before any density dataset was collected, any policy trained, and any density-sweep closed-loop outcome inspected.**
Nothing below changes after results appear; deviations are logged in the report.

## Question
**Primary:** when model capacity, physics, optimisation exposure and evaluation procedure are fixed, does **deliberately manipulated** local
demonstration geometry determine closed-loop success?

Until now the nearest-demonstration distance effect has been *observational*: growing a dataset changes count, coverage, density, nearest
distance and diversity together. Here local geometry is manipulated directly while demonstration count is held constant.

**Secondary:** does the controlled curve reproduce the observational one; is degradation smooth or sharp; what distances correspond to 90 /
75 / 50 / 25 / 10% success; does geometry beyond the single nearest demonstration matter; do surrounded configurations beat one-sided ones at
matched distance; which failure modes grow as coverage thins; is the effect stable across training seeds?

## Frozen prior state
`docs/research/seed_replication_freeze.json`: commit `1b618f0`, environment and expert source hashes, benchmark and dataset hashes, model /
diffusion / optimizer / normalization / evaluation configs, all 5-seed checkpoints, and the observational curve
(**d75 ≈ 4.5 mm, d50 ≈ 8.8 mm [7.2, 10.3], d10 ≈ 17.4 mm**). Those artifacts are read-only; this phase writes to `artifacts/density_sweep/`.

## Hypotheses
- **H1:** success falls monotonically as the controlled nearest-demonstration distance grows.
- **H2:** the controlled curve agrees with the observational one within its uncertainty (controlled d50 inside roughly 7–11 mm).
- **H3:** at matched nearest distance (7.5 mm), surrounded support beats one-sided support.
- **H4:** the density effect is larger than training-seed variance.
- **H5:** as coverage thins, lateral-alignment failures grow fastest (the mode that already dominates).

## Density manipulation (the core design)
Seven **separate 80-demonstration training sets**, one per condition. Each evaluation position sits at the centre of a hole of radius r that
contains no demonstration; a ring of demonstrations sits at exactly r; the remaining demonstrations fill the rest of the workspace by
farthest-point sampling and never enter a hole. **Demonstration count is identical (80) in every condition**, so this cannot become a
dataset-size sweep in disguise. Construction: `data/density_design.py` → `docs/research/controlled_density_design.json`, figure
`docs/assets/v2_density_design.png`.

| condition | nominal r | evaluation positions | demos | measured d1 | median largest angular gap | own NN median | workspace coverage within 10 mm |
|---|---|---|---|---|---|---|---|
| r2.5 | 2.5 mm | 12 | 80 | 2.500 mm | 60° | 2.5 mm | 0.89 |
| r5.0 | 5.0 mm | 12 | 80 | 5.000 mm | 60° | 5.0 mm | 0.98 |
| r7.5 | 7.5 mm | 10 | 80 | 7.500 mm | 60° | 7.5 mm | 1.00 |
| r10.0 | 10.0 mm | 8 | 80 | 10.000 mm | 36° | 8.3 mm | 1.00 |
| r15.0 | 15.0 mm | 4 | 80 | 15.000 mm | 31° | 8.0 mm | 0.94 |
| r20.0 | 20.0 mm | 3 | 80 | 20.000 mm | 46° | 6.1 mm | 0.82 |
| **r7.5_onesided** (secondary) | 7.5 mm | 10 | 80 | 7.500 mm | **220°** | 4.9 mm | 0.89 |

**Feasibility adjustments, fixed before training:** the workspace is only 45 × 140 mm, so a hole of radius r costs πr² of it. The
pre-registered targets of 12/12/10/8/5/3 evaluation positions are met except at r = 15 mm, where packing (centres ≥ 2r + 0.5 mm apart, rings
inside the workspace) permits **4, not 5**. All six nominal radii are retained. 59 evaluation positions in total.

For the one-sided condition the ring spans only ±45°, and filler is excluded from the unsupported side out to 2.5 r, so the manipulation is
not undone by filler. Its measured angular gap is 220° versus 60° for the surrounded conditions at the same 7.5 mm distance.

## Measured geometry (labels are never used in the analysis)
Per evaluation position: d1, d2, d3, mean of the six nearest, counts within 5 / 10 / 15 mm, local support points within 2.5 r, the largest
angular gap and a surrounded flag. All analyses use these measurements, not nominal labels.

## Expert validation first
The unchanged physics-v2 expert collects every training position and every evaluation position. **Evaluation positions the expert cannot
solve are excluded before any policy evaluation** and reported; they never count as policy failures. Policy-hard positions are never removed
afterwards. If the expert fails a *training* position, that condition simply has fewer than 80 demonstrations, and the shortfall is reported.

## Training
The same ~2M TinyRDT, unchanged in every respect (hidden 192, 4 layers, 6 heads, frozen pooled MobileNet encoder, H = 16, cosine schedule,
x0 prediction, hold padding, DDIM-10 sampler, action representation, AdamW lr 1e-3 / wd 1e-4, batch 8, EMA 0.999, fp32).

- **Three training seeds per condition: 0, 1, 2** (21 models). No seed is retrained or dropped, whatever its result.
- **Exposure matched** at the established 147.874 passes: `steps = round(147.874 × windows / 8)`, with windows **measured** from each
  collected dataset, never assumed from demonstration count.
- **Normalization** follows the established protocol: statistics computed from each training set's own windows. Evaluation positions never
  influence normalization.
- Training may run two jobs concurrently on the GPU; training is bit-deterministic given a seed (verified in the previous phase), so
  concurrency cannot alter results.

## Evaluation
- **Training-scene diagnostic** for every model: closed loop on 10 scenes from its own training set (the first 10 episode indices),
  K = 8, policy seed 0. Diagnostic only — a bad run is flagged, never discarded.
- **Primary closed-loop evaluation:** each model on its own condition's evaluation positions, **K = 8, policy seed 0**, max 150 steps,
  DDIM-10. K is not tuned.
- **Physics-v2 validity is enforced unchanged**: valid opposing side pinch, contact-normal constraints, robot–table / cube–table / pad–cube
  penetration tolerances, lift and hold. Invalid rollouts cannot count as successes and are reported separately. Thresholds are never loosened.
- Offline metrics (train / eval-position MAE) are recorded as secondary diagnostics only.

## Analysis
- **Primary figure:** success probability against *measured* nearest-demonstration distance, showing individual positions, every training seed,
  the fitted curve and its uncertainty. Individual observations are never hidden behind the curve.
- **Controlled curve:** logistic fit of success on measured d1, with a position-cluster bootstrap, giving **d90 / d75 / d50 / d25 / d10 with
  CIs**. These are "distances associated with X% estimated success", never thresholds.
- **Comparison with the observational curve** from the seed-replication study, plotted together and compared numerically. Agreement and
  disagreement are both reported as informative.
- **Beyond the nearest neighbour:** compare simple, interpretable predictors — d1, d2, d3, mean of six nearest, counts within 5/10/15 mm,
  local support points, largest angular gap — by single-predictor logistic fit and by adding each to a d1-only model. No large model is fitted
  to this small dataset.
- **Support geometry:** surrounded (r7.5) versus one-sided (r7.5_onesided) at matched d1, paired by nothing (different positions) and
  compared with exact tests plus per-position counts.
- **Repeated measures respected:** rollouts share positions and seeds; the statistical model is a logistic regression with a two-way cluster
  bootstrap over evaluation positions and training seeds, as in the previous phase. Raw counts accompany every model-based number.
- **Variance decomposition:** the share of outcome variance associated with position, condition, and training seed.
- **Failure taxonomy** (existing, unchanged) by condition; **trajectory analysis** on representative positions using already-recorded
  quantities (grasp-centre lateral and vertical error at the close, cube displacement, close timing). Analysis only: nothing changes policy
  behaviour.

## Causal language
Density is manipulated here, so this design supports stronger statements than the previous regression. Claims stay within the design:
conditions differ in local geometry **and** in the global arrangement forced by holding the count fixed (e.g. clustered rings at small r).
Those side effects are quantified and disclosed rather than assumed away.

## Exclusions
- Evaluation positions the expert cannot solve (decided before policy evaluation).
- Nothing else. No position, seed or condition is removed after seeing policy results.

## Decision gate
**CASE A** controlled density effect; **CASE B** observational curve replicates; **CASE C** support geometry matters at matched distance;
**CASE D** density does not explain performance; **CASE E** training variance dominates; **CASE F** density explains failure and 2M capacity
has not yet been tested as a bottleneck. Several may apply.

## Stop condition
Stop after: datasets constructed and collected; expert validation complete; all 21 models trained; closed-loop evaluation complete; the
controlled density response estimated; the observational comparison done; support-geometry, failure and trajectory analyses complete;
`CONTROLLED_DENSITY_SWEEP_REPORT.md` written; artifacts committed.

**Do not** scale capacity (not even if CASE A/B/F holds — the capacity experiment must be designed from the measured curve), add
demonstrations after seeing results, add DAgger, add a world model, fine-tune vision, add language, or start new task families.
