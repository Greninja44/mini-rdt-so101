# Physics-v2 targeted capacity × density scaling: pre-registered protocol

**Registered 2026-09-22, before any model larger than the ~2M baseline was trained.** Nothing below changes after results appear; deviations
are logged in the report.

## Question
**Primary:** does increasing TinyRDT **policy capacity** expand its spatial generalization range when demonstration geometry is held fixed?
**Secondary:** does capacity help sparse *surrounded* interpolation and weak *one-sided* support equally?

This is the first capacity experiment in the project. It was deferred until now on purpose: the 2M policy was already perfect wherever
demonstrations were dense, so capacity had nothing to show. The controlled density sweep produced conditions where the 2M policy is neither
saturated nor hopeless, and this study is confined to exactly those.

## Frozen prior state
`docs/research/density_sweep_freeze.json`: commit `2c86475`, environment sources, the density design and dataset hashes, the frozen vision
encoder checkpoint, the 2M model/diffusion/optimizer configs, all 21 density checkpoints, the evaluation config, the success criterion, and
the 2M results. Those artifacts are read-only; this phase writes to `artifacts/capacity_scaling/`.

## Hypotheses (predictions, reported whatever happens)
- **H1** — if 2M capacity limits interpolation, larger models shift the curve outward: d75, d50 and d25 grow with capacity.
- **H2** — if one-sided support is a harder distribution shift, capacity helps surrounded conditions more than r7.5_one-sided.
- **H3** — if the limit is representation or data geometry, larger models show no systematic improvement.
- **H4** — if larger models overfit local demonstrations, capacity *reduces* generalization.

Non-monotone, saturating and negative outcomes are all reportable results.

## Model family (one scaling rule, no hand-picked architectures)
Everything except transformer size is fixed: tokenization (pooled vision token + state token + time token + 16 action tokens), the frozen
MobileNet encoder and its preprocessing, the proprioceptive and action representations, H = 16, the learned positional embedding, the x0
diffusion objective with a cosine schedule, the DDIM-10 sampler, and the output head.

**Scaling rule:** head dimension fixed at 32 (heads = d / 32), MLP width fixed at 4 d, pre-LN GELU blocks, and depth tied to width at the
baseline's aspect ratio, **layers = round(d / 48)**. Stepping d by 64-ish along this rule gives a near-geometric ×2.1 ladder:

| label | hidden d | layers | heads | MLP | **exact trainable policy parameters** |
|---|---|---|---|---|---|
| 2.0M (existing baseline) | 192 | 4 | 6 | 768 | **2,009,670** |
| 4.3M | 256 | 5 | 8 | 1,024 | **4,304,902** |
| 9.1M | 320 | 7 | 10 | 1,280 | **9,137,286** |
| 19.5M | 416 | 9 | 13 | 1,664 | **19,517,062** |

(Counts exclude the frozen 0.94M MobileNet, which is identical in every model.) The nearest family members to the requested "5M / 10M / 20M"
are 4.3M, 9.1M and 19.5M; they are reported under their real sizes. **40M is not trained in this phase.**

## Conditions (informative range only)
The exact controlled-density datasets and evaluation positions from PR #9, reused byte-for-byte:

| condition | measured d1 | evaluation positions | 2M result (seeds 0/1/2) |
|---|---|---|---|
| r10.0 surrounded | 10 mm | 8 | 6, 6, 6 → 18/24 (75%) |
| r15.0 surrounded | 15 mm | 4 | 3, 2, 2 → 7/12 (58%) |
| r20.0 surrounded | 20 mm | 3 | 0, 1, 2 → 3/9 (33%) |
| r7.5_one-sided | 7.5 mm | 10 | 6, 5, 6 → 17/30 (57%) |
| **r7.5 surrounded (easy control)** | 7.5 mm | 10 | 10, 10, 8 → 28/30 (93%) |

The easy control is included so the surrounded-vs-one-sided gap can be measured at every capacity and to check that capacity does not damage
a condition 2M already solves. r2.5 and r5.0 are saturated for 2M and are not rerun.

## Training
- **Seeds 0, 1, 2** for every size × condition, the same seeds as the density sweep. The existing 2M runs are reused. New runs:
  3 sizes × 5 conditions × 3 seeds = **45** (36 primary + 9 easy control). No seed is replaced or retrained, whatever its result.
- **Matched data exposure:** every model sees 147.874 passes over its 4,339 measured windows = **80,222 steps at batch 8**, regardless of
  size. Larger models therefore consume more FLOPs; **this is a parameter-capacity study at matched data exposure, not a compute-matched or
  compute-optimal study.**
- **Optimizer recipe frozen** exactly as for every prior phase: AdamW lr 1e-3, wd 1e-4, grad clip 1.0, EMA 0.999, fp32, batch 8, final EMA
  weights, no early stopping. **No per-size learning-rate tuning** — tuning without touching the test set would need its own protocol. This is
  a stated limitation: the recipe was chosen for 2M, so a larger model underperforming could reflect the recipe rather than capacity. Any
  divergence (NaN or loss blow-up) is reported as a result, not rerun.
- Batch size is never changed. If a model ever failed to fit in memory, gradient accumulation would preserve the effective batch of 8 and be
  documented. Measured peak VRAM is at most 2.41 GB (19.5M), so this is not expected.
- Two training jobs run concurrently on the GPU. Training is bit-deterministic given a seed (verified earlier), so concurrency cannot change
  results. Wall time, peak VRAM and throughput are recorded per run.

## Evaluation
- **Training-scene diagnostic** for every model: the first 10 episodes of its own training set, K = 8, policy seed 0. Diagnostic only.
- **Primary closed loop:** each model on its condition's evaluation positions, **K = 8** (not tuned per size), policy seed 0, max 150 steps,
  DDIM-10 — identical for all sizes. Sampler steps are never changed by capacity.
- **Physics-v2 validity enforced unchanged**; invalid rollouts cannot count as successes and are reported separately.
- **Offline metrics** (train and evaluation-position MAE) recorded for every model, secondary only.
- **Inference latency:** `TinyRDTPolicy.predict` (10 DDIM steps, batch 1) on the same GPU and on CPU, per size, measured on a fixed input.

## Primary metric
Success per size × condition: raw counts, every seed visible, with Wilson intervals.

## Analysis
- **Capacity curves:** success vs parameter count, separately per condition.
- **Distance curves by capacity:** logistic fit of success on d1 over the surrounded conditions (r7.5, r10, r15, r20 — the conditions present
  for every size), with d90 / d75 / d50 / d25 / d10 and position-cluster bootstrap CIs, per size. The central question is whether **d50
  moves outward**.
- **Interaction model:** success ~ distance + log2(params) + distance × log2(params) over the surrounded conditions, with a two-way cluster
  bootstrap over evaluation positions and training seeds (seeds nested in size). The interaction term answers whether capacity reduces the
  penalty of sparse demonstrations.
- **Support geometry:** gap = success(r7.5 surrounded) − success(r7.5 one-sided), per size.
- **Failure mechanism:** the existing taxonomy, plus lateral and vertical error at the first close, closest approach, close timing, and
  pre-pinch cube displacement, per size and condition.
- **Position level:** per position, successes out of 3 seeds for each size; count positions **rescued, unchanged and regressed** relative to 2M.
- **Offline vs closed loop:** offline MAE vs success across all models, and whether capacity improves one without the other.
- **Parameter efficiency:** Δ success and Δ d50 for 2.0→4.3, 4.3→9.1 and 9.1→19.5M. No invented efficiency score.
- **Cost:** training wall time, peak VRAM, and inference latency per size.
- Raw counts accompany every model-based number. No p-value is reported alone.

## 40M gate (reported, never launched in this phase)
A future 40M run is justified only if 19.5M materially beats 9.1M, the gain exceeds seed noise, the capacity × distance interaction remains
favourable, and d50 has not clearly saturated. Even if all four hold, this phase stops and reports.

## Decision gate
**A** capacity expands the generalization range · **B** capacity helps hard interpolation but not one-sided support · **C** capacity also
closes the one-sided gap · **D** capacity saturates early · **E** capacity does not help · **F** capacity hurts · **G** offline improvement
without closed-loop improvement. Several may apply.

## Stop condition
Stop after: the 4.3M / 9.1M / 19.5M models are trained on all three seeds; the selected conditions are evaluated; the capacity-distance
curves, support-geometry analysis, failure mechanism and compute/latency measurements are complete; `CAPACITY_DENSITY_SCALING_REPORT.md` is
written; artifacts are committed.

**Do not** train 40M, add a world model, DAgger, spatial tokens, vision fine-tuning, language or new demonstrations, or start PickPlace,
cloth or real-robot transfer.
