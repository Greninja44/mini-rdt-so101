# Physics-v2 controlled demonstration-density sweep: distance matters, but *where* the demonstrations sit matters as much

**Headline: manipulating local demonstration geometry causally changes closed-loop success — and the previous observational curve was too
pessimistic by about 7 mm, because it confounded distance with one-sided support.** With demonstrations *surrounding* an unseen cube
position, the same ~2M TinyRDT holds 100% at 2.5 mm, 93% at 7.5 mm and 75% at 10 mm, with an estimated 50% point at **16.0 mm
[11.9, 21.8]** versus the observational **8.8 mm [7.2, 10.3]**. The slope of the two curves is identical (−2.56 vs −2.55 logit per 10 mm);
only the offset differs. At an *identical* 7.5 mm nearest distance, surrounded support gives **28/30** and one-sided support **17/30**
(Fisher p = 0.0021) — and 17/30 = 57% is almost exactly what the observational curve predicts at 7.5 mm (58%).

Pre-registration: `docs/research/controlled_density_sweep_spec.md` (commit `99b4b5b`, before any density dataset was collected).
Frozen prior state: `docs/research/seed_replication_freeze.json` (commit `1b618f0`).
Raw numbers: `artifacts/density_sweep/{summary.json,RESULTS_DIGEST.txt}`.

---

## 1. Research question
When capacity, physics, optimisation exposure and evaluation are fixed, does **deliberately manipulated** local demonstration geometry
determine closed-loop success?

## 2. Motivation
Every distance effect so far was observational: growing a dataset changes count, coverage, density, nearest distance and diversity together.
Regression said local coverage explained the benefit of more data, but regression cannot establish that *manipulating* density moves
performance. This phase manipulates it.

## 3. Pre-registration
Fixed in advance: hypotheses, the manipulation, evaluation positions, three seeds (0, 1, 2), the exposure rule, primary K = 8, the statistics,
the exclusion rule, and the decision gate. Predictions: **H1** monotone decline (**holds**), **H2** the controlled curve agrees with the
observational one (**fails — it is shifted right by ~7 mm**), **H3** surrounded beats one-sided at matched distance (**holds**), **H4** the
density effect exceeds training-seed variance (**holds, overwhelmingly**), **H5** lateral-alignment failures grow fastest (**holds**).

## 4. Density manipulation
Seven separate **80-demonstration** training sets. Each evaluation position sits at the centre of a hole of radius r containing no
demonstration; a ring of demonstrations sits at exactly r; farthest-point filler covers the rest and never enters a hole. **Demonstration
count is identical in every condition**, so this is not a dataset-size sweep in disguise. Construction: `data/density_design.py`;
figure `docs/assets/v2_density_design.png`.

## 5. Dataset construction and measured geometry

| condition | nominal r | **measured d1** | eval positions | demos | median angular gap | own-NN median |
|---|---|---|---|---|---|---|
| r2.5 | 2.5 mm | 2.500 mm | 12 | 80 | 60° | 2.5 mm |
| r5.0 | 5.0 mm | 5.000 mm | 12 | 80 | 60° | 5.0 mm |
| r7.5 | 7.5 mm | 7.500 mm | 10 | 80 | 60° | 7.5 mm |
| r10.0 | 10.0 mm | 10.000 mm | 8 | 80 | 36° | 8.3 mm |
| r15.0 | 15.0 mm | 15.000 mm | 4 | 80 | 31° | 8.0 mm |
| r20.0 | 20.0 mm | 20.000 mm | 3 | 80 | 46° | 6.1 mm |
| **r7.5_onesided** | 7.5 mm | 7.500 mm | 10 | 80 | **220°** | 4.9 mm |

Every nominal radius is realised exactly. The pre-registered target of 5 evaluation positions at r = 15 mm became 4 (a hole costs πr² and the
workspace is only 45 × 140 mm); this was recorded before training.

## 6. Expert validation
The unchanged physics-v2 expert collected all 560 training demonstrations and all 59 evaluation positions: **59/59 solvable, zero
exclusions, and all seven conditions kept their full 80 demonstrations.** Every condition therefore enters the analysis at full strength, and
no policy failure is contaminated by an unreachable position.

## 7. Actual geometry used in analysis
Per position: d1, d2, d3, mean of the six nearest, counts within 5/10/15 mm, support points within 2.5 r, largest angular gap, surrounded
flag. All analyses use these measurements; nominal labels are used only as condition names.

## 8–10. Training, exposure and seeds
The same ~2M TinyRDT, unchanged (hidden 192, 4 layers, 6 heads, frozen pooled MobileNet, H = 16, cosine, x0, hold padding, DDIM-10, AdamW
lr 1e-3, batch 8, EMA 0.999, fp32). Windows were **measured** at 4,339 per condition, giving **80,222 optimizer steps** at the established
147.874 passes. Three seeds per condition (0, 1, 2) = **21 models**; none retrained or dropped. Two training jobs ran concurrently; training
is bit-deterministic given a seed, so concurrency cannot change results.

## 11. Training-scene diagnostic (10 own-training scenes per model)
Per seed: r2.5 10/10/10 · r5.0 10/10/10 · r7.5 8/10/8 · r10.0 9/10/10 · r15.0 10/10/8 · r20.0 10/7/9 · one-sided 10/10/10. Every model
executes most of its own training data, so no condition's held-out result is explained by gross under-training.

## 12. Offline metrics (secondary)
Mean offline MAE on the evaluation positions rises with sparsity: 0.0049 (2.5 mm), 0.0055 (5), 0.0057 (7.5), 0.0054 (10), 0.0062 (15),
0.0074 (20), **0.0074 (one-sided)**. Training MAE is flat at 0.0047–0.0052 everywhere. Unlike the within-scale comparison of the previous
phase, offline error does track condition difficulty here — including the one-sided condition, whose offline error matches the 20 mm
condition. That is consistent with the geometry effect being visible in the learned mapping, not only in closed-loop execution.

## 13. Closed-loop results (primary: K = 8, policy seed 0, 3 training seeds)

| condition | measured d1 | per seed | aggregate | rate | Wilson 95% |
|---|---|---|---|---|---|
| r2.5 | 2.500 mm | 12, 12, 12 | **36/36** | **100%** | 90.4–100% |
| r5.0 | 5.000 mm | 12, 10, 12 | 34/36 | 94.4% | 81.9–98.5% |
| r7.5 | 7.500 mm | 10, 10, 8 | 28/30 | 93.3% | 78.7–98.2% |
| r10.0 | 10.000 mm | 6, 6, 6 | 18/24 | 75.0% | 55.1–88.0% |
| r15.0 | 15.000 mm | 3, 2, 2 | 7/12 | 58.3% | 32.0–80.7% |
| r20.0 | 20.000 mm | 0, 1, 2 | 3/9 | 33.3% | 12.1–64.6% |
| **r7.5_onesided** | 7.500 mm | 6, 5, 6 | **17/30** | **56.7%** | 39.2–72.6% |

**Manipulating density changes success monotonically: 100% → 94% → 93% → 75% → 58% → 33%.** Figure:
`docs/assets/v2_density_response.png`, with every position and every seed plotted.

## 14. Controlled density-response curve
Logistic fit on measured d1 over the six surrounded conditions (153 rollouts, 51 positions), position-cluster bootstrap:
intercept 4.09, **slope −2.56 logit per 10 mm** (two-way cluster bootstrap [−5.32, −1.16]).

## 15. Characteristic distances (estimates, **not thresholds**)

| estimated success | controlled | 95% CI | observational (previous) |
|---|---|---|---|
| 90% | **7.4 mm** | [4.9, 11.1] | 0.2 mm |
| 75% | **11.7 mm** | [9.1, 15.7] | 4.5 mm |
| **50%** | **16.0 mm** | [11.9, 21.8] | 8.8 mm [7.2, 10.3] |
| 25% | 20.3 mm | [14.5, 28.3] | 13.1 mm |
| 10% | 24.5 mm | [17.0, 34.7] | 17.4 mm |

## 16. Comparison with the observational curve — the informative disagreement
The two curves have **the same slope** (−2.56 vs −2.55 logit per 10 mm) and **different offsets**: the controlled curve sits ~7 mm to the
right. The observational estimate of "how far the policy can generalize" was pessimistic by roughly a factor of two in distance.

The reason is visible in this experiment. In the observational study, positions far from demonstrations were also positions whose
demonstrations lay mostly on one side (hole-band edges, workspace boundaries). Here, at a matched 7.5 mm:
- **surrounded: 93.3%**, well above the observational prediction of 58% at that distance;
- **one-sided: 56.7%**, almost exactly the observational prediction.

So the observational curve was not wrong about the data it summarised — it was measuring a mixture dominated by one-sided support. **Nearest
distance alone does not capture coverage.**

## 17. Beyond the nearest neighbour
Among surrounded conditions, d1, d2, d3 and the mean of the six nearest are statistically indistinguishable (standardised coefficients all
≈ −1.2), and counts within 5/10/15 mm are their positive mirror image. That is expected: the design makes them near-collinear (within a
condition d1 is constant by construction). No local metric beats d1 in this design, and the honest reading is that this experiment cannot
separate them.

The angular-gap predictor is **confounded inside the surrounded set** (the gap measure shrinks with r because it is computed in a 2.5 r
window), so its apparent positive coefficient there is an artifact and is not interpreted. The clean geometry test is §18.

## 18. Interpolation vs one-sided support (the strongest single result)
Same model, same exposure, same demonstration count, same measured d1 = 7.500 mm, different surrounding geometry:

| geometry | angular gap | per seed | aggregate | rate |
|---|---|---|---|---|
| surrounded | 60° | 10, 10, 8 | **28/30** | 93.3% |
| one-sided | 220° | 6, 5, 6 | **17/30** | 56.7% |

Fisher exact on rollouts p = 0.0021 (anti-conservative because rollouts cluster by position and seed — but the effect is consistent in all
three seeds, and seed variance in this study is negligible, §19).

Pooled over all seven conditions, with distance and a one-sided indicator (two-way cluster bootstrap):
- distance **−2.56 per 10 mm** [−5.20, −1.27];
- **one-sided support −2.00 logit** [−4.06, −0.20].

In distance units, losing the surround costs about **7.8 mm** of nearest-demonstration distance. The policy interpolates *between*
demonstrations much better than it reaches *away* from them.

## 19. Training-seed variance
Negligible here. Variance decomposition of the explained outcome variance: **position 65.9%, density condition 33.8%, training seed 0.26%.**
Per-seed condition totals differ by at most 2 rollouts except r20.0 (0, 1, 2 of 3). The manipulated variable dominates optimisation noise,
which is the opposite of the 20-demonstration regime in the previous phase.

## 20. Trajectory analysis
Using already-recorded quantities, the lateral error of the grasp centre at the first close grows smoothly with sparsity — for **successful**
rollouts: 3.0 mm (2.5), 2.8 (5), 3.7 (7.5), 6.4 (10), 8.1 (15), 9.9 mm (20). For failures it is 8.5–16.6 mm. The one-sided condition is the
outlier: its successes close at 7.0 mm and its failures at 14.8 mm, i.e. it behaves like a much sparser surrounded condition.

Failure therefore appears as a *progressive aiming error*, not a sudden collapse: the policy keeps executing a plausible grasp and misses by
more and more as support thins.

## 21. Failure analysis
`wrong_lateral_alignment` dominates at every level and scales with sparsity (share of all rollouts): 0% at 2.5 mm, 2.8% at 5, 3.3% at 7.5,
16.7% at 10, 41.7% at 15, 55.6% at 20, and **43.3% one-sided**. `wrong_height` appears occasionally (1–2 rollouts per condition). There were
no drops, no cube-tilt failures, and no approach-positioning failures. H5 holds.

## 22. Physical validity
387 rollouts (177 held-out + 210 training-scene), all enforcing physics-v2. **Zero successes with an invalid reason and zero invalid-physics
rollouts.** Max robot–table penetration 0.00 mm; max cube–table 0.77 mm; max pad–cube 1.21 mm (limits 1.0 / 1.0 / 1.5 mm). Every counted
success is a valid opposing side pinch held 5 steps with the cube lifted.

## 23. Statistical summary
- Raw counts everywhere, Wilson intervals for per-condition rates (naive: rollouts cluster by position and seed).
- The curve and all coefficients use a **two-way cluster bootstrap** over evaluation positions and training seeds.
- 42 of 59 positions are solved by all three seeds and 6 by none, so most positions are decided by geometry rather than by the run.

## 24. Deviations from pre-registration
- r = 15 mm yields 4 evaluation positions instead of 5 (geometric, recorded before training).
- The one-sided condition excludes filler from the unsupported side out to 2.5 r. This was in the registered design; without it filler
  silently restores the surround (a first draft measured a 156° gap instead of 220°).
- Nothing else. Seeds, budgets, K, positions, exclusion rule and statistics are exactly as registered; no condition, seed or position was
  dropped after seeing results.

## 25. Limitations
- **3 seeds and few positions at large radii** (4 at 15 mm, 3 at 20 mm): the wide ends of the curve rest on 12 and 9 rollouts, and the CIs
  say so (d50 [11.9, 21.8]).
- **Holding count fixed changes global structure**: at 2.5 mm the 80 demonstrations form 12 tight clusters, at 20 mm they are spread. Local
  geometry and global arrangement therefore move together to some degree; the conditions are matched on count, not on global layout.
- Within a condition d1 is constant by construction, so local metrics are collinear and this design cannot rank them (§17).
- The one-sided test is at a single distance (7.5 mm); the −2.00 logit estimate should not be extrapolated to all distances.
- One task, one cube, fixed yaw, one camera, K = 8 only.
- The controlled curve is fitted over 2.5–20 mm; d10 = 24.5 mm is an extrapolation beyond the data and its CI is correspondingly wide.

## 26. Reproduction
```bash
.venv/bin/python -m data.density_design          # design + docs/research/controlled_density_design.json
.venv/bin/python -m data.density_positions       # per-condition position files
scripts/experiments/run_density_sweep.sh         # collect, train 21 models, evaluate (<= 2 sim workers)
scripts/experiments/run_density_finish.sh        # analysis, figures, digest
.venv/bin/python -m evaluation.density_analysis
env -u PYTHONPATH .venv/bin/python -m pytest -q tests/
```

## 27. Artifacts
Datasets `artifacts/density_sweep/data/{condition}_{train,eval}`; models `artifacts/density_sweep/models/{condition}_seed{0,1,2}` (config,
seed, steps, loss history, `last.pt`, `ema_last.pt`); rollouts under `closed_loop/` and `train_scenes/`; results in `summary.json` and
`RESULTS_DIGEST.txt`; expert control in `expert_control.json`; hashes in `docs/research/seed_replication_freeze.json` and
`docs/artifact_manifest.json`.

## 28. Decision gate and recommendation
- **CASE A — confirmed.** Manipulated density changes success monotonically (100% → 33%) with the seed effect at 0.26% of explained variance.
  Local demonstration coverage is now supported causally, not just by regression.
- **CASE C — confirmed, and it is the new result.** At matched nearest distance, surrounded support beats one-sided by 93% vs 57%
  (−2.00 logit, ≈ 7.8 mm equivalent). **Nearest-demo distance alone is insufficient.**
- **CASE B — not supported.** The controlled curve has the same slope but sits ~7 mm to the right of the observational one; the observational
  curve matches *one-sided* geometry. The earlier d50 of 8.8 mm understated what the policy can do when it is surrounded.
- **CASE E — rejected.** Training-seed variance is negligible in this regime.
- **CASE F — applies.** Density explains failure, and 2M capacity has still not been tested as the bottleneck.

Recommended next experiment: **capacity × density at the informative distances**, now measurable rather than guessed. The 2M policy is
saturated at ≤ 7.5 mm surrounded (93–100%, nothing to gain) and near-floor at 20 mm (33%), so the informative band is **10–20 mm surrounded
plus the 7.5 mm one-sided condition**, where it sits at 33–75%. A capacity comparison should be run there, with the density conditions of this
study reused unchanged so that capacity is the only variable.

**Stopping here, as pre-registered. No capacity scaling was started.**
