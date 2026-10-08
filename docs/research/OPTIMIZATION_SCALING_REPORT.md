# Optimization scaling study — interim report (Stages A and B)

**Status: interim. Stages A and B are complete; Stage F is running.** This report covers the screening
result and states plainly what it does and does not establish. The held-out evaluation that decides
whether restored training health converts into better generalization is Stage F, which was still in
progress when this was written. No held-out number influenced any selection reported here.

Pre-registration: [`optimization_scaling_spec.md`](optimization_scaling_spec.md), committed as `5073d57`
before any run. Deviations and interruptions: [`optimization_deviation_log.md`](optimization_deviation_log.md).

## 1. Question

The capacity × density study found that models above ~4.3M parameters did not improve, and the recovery
study closed the EMA-resume defect as an explanation, leaving the capacity comparison **optimization-
confounded**: every capacity had been trained with one recipe, tuned at 2.0M. This study asks whether
the poor training of the larger models is caused by that fixed recipe rather than by capacity. Only
optimizer variables may change — not architecture, data, horizon, sampler, evaluation or success criteria.

## 2. What was held constant

Dataset and splits, 80,222 optimizer updates, batch size, H = 16, x0 prediction, cosine noise schedule,
hold-last-action padding, DDIM-10, EMA 0.999, K = 8, physics-v2 success criteria, evaluation positions,
the frozen pooled MobileNet encoder, and all seeds. The screening cell is condition r10, seed 0.

## 3. The training-health gate

Selection used training health only. A run passes when all six hold:

| # | criterion | bound |
|---|---|---|
| 1 | training-scene success | ≥ 90% |
| 2 | no NaN/Inf | — |
| 3 | final sampled training MAE | ≤ 0.01012 |
| 4a | gradient-norm median over the final 10% of updates | ≤ 0.159 |
| 4b | spike rate (norms above 10× that run's own median) | ≤ 3.4% |
| 5 | EMA MAE ≤ raw MAE | — |
| 6 | checkpoint integrity | — |

Criterion 4 was rewritten by Amendment 2 **before any candidate ran**, because the version registered
first ("no norm above 100× the reference median") was not satisfiable by the healthy 2.0M baseline, which
peaks at 77× its own median. Both replacements are scale-free and were derived from 2.0M/4.3M baseline
data only. The MAE threshold was never changed.

## 4. Instrumentation, and the evidence it changed nothing

Seven knobs were added to the trainer (`--weight-decay`, `--grad-clip`, `--warmup-steps`, `--lr-schedule`,
`--lr-final-fraction`, `--grad-log-interval`, `--stop-after`) with defaults that reproduce the historical
recipe. `tests/test_optimization_recipe.py` asserts bit-equality between a no-flag run and one with the
baseline values written out, and that an interruption resumed under a **non-default** recipe is bit-identical
to an uninterrupted run. Empirically, Stage A's 4.3M run is bit-identical to its frozen counterpart
(max abs EMA difference 0.0), and 9.1M reproduces its frozen training MAE exactly (0.00745).

The learning-rate multiplier is a pure function of the update index, so no scheduler state is checkpointed.
An interruption must keep `--steps` fixed, since the schedule horizon is part of the recipe; `--stop-after`
emulates an interruption faithfully, and finding this was necessary is recorded as a fix, not a result.

## 5. Stage A — the baseline recipe at four capacities

Full 80,222 updates, unchanged recipe, condition r10 seed 0, gradients logged every 50 updates.

| capacity | train MAE | EMA MAE | final loss | scenes | grad median | tail median | spike rate | max/median | lateral at close | wall | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.0M | 0.00778 | 0.00462 | 0.00187 | 9/10 | 0.097 | 0.053 | 0.81% | 77× | 5.50 mm | 36 min | pass |
| 4.3M | 0.00973 | 0.00525 | 0.00831 | 10/10 | 0.103 | 0.058 | 0.87% | 87× | 4.79 mm | 53 min | pass |
| 9.1M | 0.00745 | 0.00555 | 0.00191 | 10/10 | 0.074 | 0.049 | 2.74% | 145× | 4.34 mm | 61 min | pass |
| 19.5M | **0.01310** | 0.00935 | 0.00933 | **4/10** | 0.093 | 0.059 | 2.62% | 157× | **14.11 mm** | 89 min | **fail (1, 3)** |

Only 19.5M fails, and it fails plainly: it misses the MAE threshold and solves fewer than half the scenes
it was fit on, while missing the cube laterally by 14.1 mm against the 4–5 mm of the healthy models.
Gradients are **not** exploding — its tail median and spike rate are both inside the bounds. This is
underfitting, not instability.

**9.1M passes the baseline gate at the screening cell**, which weakens its screen. Amendment 3(a) recorded
this in advance: the frozen 9.1M r10 per-seed training-scene results are 10, 7 and 7 of 10, and the
pre-registered screen uses seed 0 — the passing one. The screening cell was not moved after seeing this,
because choosing a seed by which ones fail would be selection by outcome.

## 6. Stage B — learning rate

Candidates were fixed a priori from width scaling (lr ∝ 1/width): baseline width 192 implies ≈0.60× for
9.1M (width 320) and ≈0.46× for 19.5M (width 416); 0.25× and 0.5× bracket those, and 2× tests the opposite
hypothesis that the large models are under-trained.

### 9.1M (width 320)

| recipe | train MAE | EMA | loss | scenes | tail | spike | lateral | gate |
|---|---|---|---|---|---|---|---|---|
| 2× (2e-3) | 0.01202 | 0.01070 | 0.00541 | 6/10 | 0.052 | 1.99% | 13.05 mm | fail |
| 1× baseline | 0.00745 | 0.00555 | 0.00191 | 10/10 | 0.049 | 2.74% | 4.34 mm | pass |
| 0.5× | 0.00753 | 0.00445 | 0.00155 | 9/10 | 0.067 | 0.87% | 4.38 mm | pass |
| **0.25×** | **0.00697** | 0.00443 | 0.00232 | 10/10 | 0.116 | 0.31% | 4.16 mm | **pass, selected** |

### 19.5M (width 416)

| recipe | train MAE | EMA | loss | scenes | tail | spike | lateral | gate |
|---|---|---|---|---|---|---|---|---|
| 2× (2e-3) | 0.01623 | 0.01330 | 0.01323 | 3/10 | 0.069 | 2.68% | 16.79 mm | fail |
| 1× baseline | 0.01310 | 0.00935 | 0.00933 | 4/10 | 0.059 | 2.62% | 14.11 mm | fail |
| 0.25× | 0.00732 | 0.00444 | 0.00116 | 8/10 | 0.140 | 0.55% | 4.58 mm | fail (criterion 1) |
| **0.5× (5e-4)** | 0.00789 | 0.00497 | 0.00117 | **9/10** | 0.062 | 1.50% | 4.41 mm | **pass, selected** |

## 7. What the learning rate actually did

Three independent measures move together at 19.5M, and the optimum is interior:

- **Fit.** Training MAE falls 0.01310 → 0.00789 (−40%) and final loss 0.00933 → 0.00117 (−87%). The
  largest model, properly stepped, fits better than the 4.3M baseline (0.00973) — it was never short of
  capacity at this cell.
- **Aim.** Lateral error at first gripper close falls 14.11 mm → 4.41 mm, into the band every healthy run
  occupies. The failure mode was reaching for the wrong place, not mistiming the grasp.
- **Trajectory.** At matched updates the baseline stops converging in its second half (raw MAE oscillating
  0.0129–0.0164, EMA pinned near 0.011) while 0.5× and 0.25× descend smoothly.

The selected 0.5× sits almost exactly on the a-priori width-scaling prediction of ≈0.46× for width 416,
which was written into the spec before any of these runs. The same manipulation run *downward* in capacity
corroborates it: at 9.1M, 2× reproduces the 19.5M failure signature closely (MAE 0.01202, 6/10 scenes,
13.05 mm lateral against 19.5M's 0.01310, 4/10, 14.11 mm). Too large a step for the width produces this
failure at either capacity.

Gradient behaviour explains why the gate needed criterion 4 rewritten. Lowering the rate *raises* the
gradient tail median (19.5M: 0.059 at 1×, 0.062 at 0.5×, 0.140 at 0.25×) because the weights move less per
step, while sharply lowering the spike rate (2.62% → 1.50% → 0.55%). Nothing explodes in any failing run;
the failures are all underfitting.

## 8. What this does not establish

- **The screening evidence is one cell, one seed, ten scenes.** Wilson intervals: 4/10 = 40% [17, 69];
  9/10 = 90% [60, 98]. Fisher's exact test on 19.5M baseline 4/10 versus 0.5× 9/10 gives **p = 0.057** —
  by the proportions alone this comparison is underpowered and does not clear 0.05. The weight of the
  evidence is carried by the continuous measures (MAE, loss, lateral error), where the separation is large
  and consistent, not by the scene counts.
- **The 9.1M result is nearly a null.** Its baseline already passed, and 0.25× improves training MAE only
  0.00745 → 0.00697 — well inside the frozen per-seed spread at that cell (0.00745, 0.01071, 0.01158).
  The honest reading is that 0.25× does not *break* a healthy cell, not that it helps.
- **Nothing here is a generalization result.** Every number above is training health. Whether a better-
  trained 19.5M generalizes better to held-out positions is Stage F, and it remains open.

## 9. Decision gates

- **A — optimization problem confirmed: YES at 19.5M.** A pre-registered recipe restores ≥90% training-scene
  success at the screening cell, with the caveat in §8.
- **B / C — undetermined.** Whether restored health converts into held-out improvement (B) or leaves
  generalization where it was (C) requires Stage F.
- **D — optimization does not restore training: rejected at 19.5M**, on the screening cell.
- **E — not triggered.** No recipe was selected on held-out success; no held-out evaluation had been run at
  selection time.
- **F — open.** The support-geometry bottleneck from the density sweep is untouched by this study and is
  re-measured in Stage F via the one-sided versus surrounded contrast at r7.5.

The practical consequence for the earlier work is concrete: the capacity comparison **was** optimization-
confounded at 19.5M, exactly as the recovery study suspected but could not demonstrate. Its 19.5M column
was measuring a mis-stepped optimizer, not a capacity limit.

## 10. Stage F (in progress)

30 runs: 9.1M at 0.25× and 19.5M at 0.5×, each across seeds 0/1/2 and all five conditions (r7.5,
r7.5 one-sided, r10, r15, r20), compared against the frozen baseline matrix from the recovery study.
This is the only stage that touches held-out positions. It matters more than a confirmation usually would,
because at 9.1M r10 the *baseline* recipe fails the health gate on two of three seeds (train MAE 0.0107 and
0.0116, 7/10 scenes) and passes only on seed 0 — the cell the pre-registration happened to screen.

Stages C (warmup), D (weight decay) and E (clipping) were **not run**: the stopping rule retires a capacity
from screening as soon as one recipe passes, and both capacities passed in Stage B.

## 11. Selection rule, as applied

Amendment 3(b): among passing recipes, one advances per capacity, ranked by training-scene success, then
training MAE, then the lower learning rate. 9.1M: 0.25× (10/10, 0.00697) over baseline (10/10, 0.00745)
over 0.5× (9/10). 19.5M: 0.5× was the only recipe that passed. Held-out results were not consulted and did
not exist. Every screened recipe is reported above whether or not it advanced.

## 12. Reproduction

```
.venv/bin/python -m pytest tests/test_optimization_recipe.py -q      # recipe equivalence and resume
bash scripts/experiments/run_optimization_stage.sh docs/research/optimization_stage_a_manifest.json
.venv/bin/python evaluation/optimization_analysis.py                 # gate + summary.json
.venv/bin/python evaluation/optimization_figures.py                  # figure set
.venv/bin/python scripts/experiments/build_next_stage.py             # staged logic, from training health only
```

Figures: `docs/assets/v2_optimization_diagnostics.png`, `_health.png`, `_mae_by_recipe.png`,
`_capacity_response.png`, `_lateral.png`, `_heldout.png`, `_offline_vs_loop.png`. Panels requiring Stage F
render placeholders until it completes.

## 13. Interruptions

Three, none affecting a result: a reboot during Stage A's 19.5M run (resumed from its atomic checkpoint);
a Stage B trainer that outlived a service stop and completed as an orphan, whose evaluation was truncated
to 2 of 10 rollouts and later completed; and a ~4-hour host sleep that froze the WSL VM during Stage F.
Details in the deviation log.
