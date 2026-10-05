# Optimization scaling study: pre-registered protocol

**Registered 2026-10-05, before any optimization experiment was run.** The candidate recipes below are fixed now and may not be extended
because an early result disappoints. Deviations are logged in the report.

## Question
The corrected capacity matrix (PR #13, commit `ced1488`) left the capacity comparison **optimization-confounded**: 9.1M and 19.5M do not fit
their own training data under the single recipe inherited from the 2M baseline. This study asks:

- **Question A — can the model train?** Can 9.1M and 19.5M be trained to healthy training-set behaviour by changing *only* optimization?
- **Question B — does better training help?** If health is restored, does held-out spatial generalization improve?

These are kept separate throughout. A recipe is selected on **A**; **B** is the outcome and is never used for selection.

## Frozen baseline
Commit `ced1488`. Corrected results, reused unchanged as the comparison:

| capacity | r10 | r15 | r20 | one-sided | r7.5 control | mean train MAE | training scenes |
|---|---|---|---|---|---|---|---|
| 2.0M | 18/24 | 7/12 | 3/9 | 17/30 | 28/30 | 0.00791 | 139/150 (93%) |
| 4.3M | 21/24 | 8/12 | 4/9 | 20/30 | 30/30 | 0.00868 | 146/150 (97%) |
| 9.1M | 18/24 | 3/12 | 3/9 | 13/30 | 18/30 | 0.01020 | 120/150 (80%) |
| 19.5M | 12/24 | 3/12 | 3/9 | 11/30 | 9/30 | 0.01238 | 70/150 (47%) |

## Baseline recipe (read from the repository, authoritative)
`training/audit_overfit.py`: **AdamW**, lr **1e-3**, betas (0.9, 0.999), eps 1e-8, weight decay **1e-4**, gradient-norm clip **1.0**,
**no scheduler, no warmup** (constant lr), EMA decay 0.999 evaluated from `ema_last.pt`, batch **8**, no gradient accumulation,
**80,222 updates** (147.87 passes over each condition's measured windows), checkpoints every 2,000 updates carrying model, EMA, optimizer
and all RNG streams. Gradient norms were **not** logged before this study.

## What is held constant
Task, physics-v2, cube geometry, demonstration datasets and their hashes, density geometry, RGB resolution, frozen vision encoder,
proprioception, action representation, diffusion objective and schedule, x0 prediction, hold-last padding, EMA decay and EMA evaluation,
H = 16, receding-horizon execution, DDIM-10, **K = 8**, success and physics-validity criteria, evaluation positions, evaluation sampler seed,
training seed IDs (0, 1, 2), training exposure (80,222 updates), checkpoint semantics, renderer, simulator, normalization, and the policy
architecture. **Only optimizer settings vary.**

## Capacities
Primary: **9.1M** (d320/L7/h10, 9,137,286) and **19.5M** (d416/L9/h13, 19,517,062) — the capacities that fail to optimize.
Controls: **2.0M** (2,009,670) and **4.3M** (4,304,902), reused from the frozen baseline and re-run only in Stage A instrumentation.
**40M is not trained in this study under any outcome.**

## Instrumentation (added before Stage A, defaults preserve history)
New knobs in the existing trainer: `--weight-decay`, `--grad-clip`, `--warmup-steps`, `--lr-schedule {constant,cosine}`,
`--lr-final-fraction`, `--grad-log-interval`, `--stop-after`. **Their defaults reproduce the historical recipe exactly**, which
`tests/test_optimization_recipe.py` asserts by training two runs — one with no flags, one with the baseline values written out — and
comparing weights, EMA and optimizer state for bit-equality. The learning-rate multiplier is a pure function of the update index, so no
scheduler state needs checkpointing; the same test confirms that an interruption resumed under a **non-default** recipe is bit-identical to
an uninterrupted run. **An interruption must keep `--steps` fixed**, because the schedule horizon is part of the recipe; `--stop-after`
emulates an interruption faithfully.

## Candidate recipes (fixed now, derived a priori — not from Stage A)
Learning rates are expressed as multiples of the baseline 1e-3 and are the **same for both capacities**, so no capacity is tuned
independently. Justification: under width scaling (lr ∝ 1/width) the baseline width 192 implies ≈0.60× for 9.1M (width 320) and ≈0.46× for
19.5M (width 416); **0.5× and 0.25× bracket those values**, and **2×** tests the opposite hypothesis that the larger models are simply
under-trained rather than unstable.

| stage | variable | candidates | runs |
|---|---|---|---|
| **A** | none — instrumented diagnosis of the baseline recipe | 2.0M, 4.3M, 9.1M, 19.5M, seed 0, condition r10, 10,000 updates, `--grad-log-interval 50` | 4 |
| **B** | learning rate | **0.25× (2.5e-4), 0.5× (5e-4), 2× (2e-3)** vs the existing 1× baseline | 2 capacities × 3 = 6 |
| **C** | warmup | **2% (1,604) and 5% (4,011)** of 80,222 updates, linear, at that capacity's best Stage-B learning rate | ≤ 4 |
| **D** | weight decay | **0 and 1e-2** vs the 1e-4 baseline, at the best recipe so far | ≤ 4 |
| **E** | gradient clipping | **0.5 and 0.25** (tighter only; clipping is never removed) | ≤ 4 |
| **F** | confirmation | the selected recipe per passing capacity × seeds 0, 1, 2 × all five conditions | ≤ 30 |

Stages B–E screen on **condition r10, seed 0, full 80,222 updates**. Stage F is the only stage that touches held-out positions.

## Stopping logic (fixed in advance)
1. Run Stage A, then Stage B.
2. If a Stage-B recipe passes the training-health gate for a capacity, **stop screening that capacity** and take it to Stage F.
3. Otherwise run Stage C for that capacity; then D; then E — each only if the previous stage left the capacity failing. Stage E additionally
   requires Stage-A evidence of gradient pathology (spikes or norms persistently above the 2.0M reference); without that evidence Stage E is
   skipped and recorded as skipped.
4. If no pre-registered recipe passes the gate for a capacity, that capacity goes **no further** and the study reports CASE D for it.
5. Ties: if two recipes pass, both go to Stage F and both are reported; no cherry-picking.

## Training-health gate (Question A) — the selection criterion
A recipe passes for a capacity when **all** of the following hold on the screening condition (and, in Stage F, across its three seeds):

1. **training-scene closed-loop success ≥ 90%** (≥ 9/10 for one run; ≥ 27/30 over three seeds);
2. **no NaN or Inf** in loss or gradient norm at any logged update;
3. **final training MAE ≤ 1.25 ×** the 2.0M baseline's mean on the same condition — on r10 the 2.0M mean is 0.00809 (0.00675, 0.00862, 0.00891), so the threshold is
   **≤ 0.01012**;
4. **gradient stability**: the median logged gradient norm over the final 10% of updates ≤ 3× the 2.0M reference median on the same
   condition, and no logged norm exceeds 100× that reference median;
5. **EMA sanity**: final EMA sampled MAE ≤ final raw sampled MAE;
6. **checkpoint integrity**: `last.pt` carries an embedded EMA with matching decay, and the evaluated `ema_last.pt` hash matches the hash
   recorded in every rollout record.

Held-out success is **never** an input to this gate. If a recipe wins only on held-out positions it is rejected as a selection criterion
(CASE E) and reported as such.

## Closed-loop evaluation (Question B)
Stage F only. Each selected model is evaluated on its condition's frozen evaluation positions — r10, r15, r20, r7.5 one-sided and the r7.5
surrounded control — at **K = 8**, policy seed 0, DDIM-10, 150-step limit, with physics-v2 validity enforced unchanged. No tuning of K, the
sampler, success thresholds or evaluation positions.

## Analysis
- Raw counts, rates and Wilson intervals; per-seed values always shown.
- Paired comparisons at **matched capacity, seed and position**, baseline recipe versus optimized recipe, using the cluster structure
  established in the recovery study (`evaluation/recovery_bootstrap.py`: positions resampled condition-stratified and shared across
  recipes, seeds resampled independently within recipe, 4,000 replicates, fixed RNG seed).
- Mechanism: training loss trajectory, gradient norms, learning-rate trajectory, EMA/raw divergence, lateral and vertical error at the first
  close, closest approach, close timing, pre-pinch cube displacement.
- Offline train and held-out MAE as secondary diagnostics only.
- Figures: loss and gradient norm versus update per capacity, learning-rate trajectory, baseline versus optimized training-scene success,
  training MAE by capacity and recipe, held-out success by recipe and capacity, per-condition results, one-sided versus surrounded, lateral
  aiming error by recipe, capacity × optimization response, offline MAE versus closed-loop success, and d50 where enough controlled points
  support a fit. Individual seeds are shown wherever they exist; axes are shared across panels that are compared.

## Decision gates
**A** optimization problem confirmed (a recipe restores ≥ 90% training-scene success) · **B** capacity becomes interpretable (health restored
*and* held-out materially improves) · **C** training recovers but generalization does not (health restored, held-out stays near 4.3M) ·
**D** optimization does not restore training · **E** a recipe wins only on held-out success (rejected as a criterion) · **F** support geometry
remains the bottleneck. Several may apply.

## Compute and infrastructure
At most two workers, the existing trainer and evaluator, the established atomic-checkpoint and user-service mechanism for long runs. Every
checkpoint restores model, EMA, optimizer and all RNG streams; the schedule is recomputed from the step. Interrupted runs resume from their
atomic checkpoint and are never rebuilt by hand; if checkpoint integrity fails, that run **stops** and is reported. New artifacts live under
`artifacts/optimization_scaling_v1/`; no previous experiment, report, dataset or checkpoint is modified.

## Stop condition
Stop after the matrix completes under its own stopping logic, diagnostics and any Stage-F evaluation finish, the report and figures are
written, tests pass and results are committed. **Do not** train 40M, run another density sweep or capacity sweep, add demonstrations, or
start DAgger, world models, language, cloth, real-robot or sim-to-real work.

## Amendment 1 (2026-10-05, before any Stage-B run; strengthens the design, loosens nothing)
Stage A was registered as 10,000 instrumented updates. That cannot supply the gradient reference the training-health gate needs, because the
gate compares the **final 10% of updates** of a full 80,222-update run against a 2.0M reference on the same condition, and a 10,000-update
run has no comparable window. The existing capacity runs carry no gradient logs at all, so the reference does not exist elsewhere.

Stage A therefore runs the **full 80,222 updates** at the **unchanged baseline recipe** for all four capacities (2.0M, 4.3M, 9.1M, 19.5M) on
condition r10, seed 0, with `--grad-log-interval 50`. The first 10,000 updates remain the early-dynamics window for diagnosis.

Because `--grad-log-interval` consumes no RNG and the recipe is otherwise untouched, each Stage-A run must reproduce its frozen counterpart
**bit-exactly**. The checkpoint hashes are compared against the frozen baseline runs and reported; a mismatch would indicate a defect in the
instrumentation and would stop the study. No candidate recipe, gate threshold, or stopping rule is changed by this amendment.
