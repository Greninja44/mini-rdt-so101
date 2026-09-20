# Physics-v2 spatial generalization: can the same ~2M TinyRDT pick cubes it never saw?

**Short answer: yes, but only inside the region its demonstrations actually cover, and only if the optimisation budget grows with the
dataset.** With 80 demonstrations and a matched budget, the same 2.01M-parameter TinyRDT picks the cube at **20/20 unseen positions inside
dense training coverage**, while dropping to 4/20 inside a 30 mm hole deliberately cut out of the training set and 4/16 just outside the
training workspace. The pre-registered primary configuration under-trained the model, and **no offline metric detected it**.

Pre-registration: `docs/research/physics_v2_generalization_spec.md` (commit `c4b9559`, before any held-out demonstration existed).
Split: `docs/research/physics_v2_generalization_split.json`. Baseline freeze: `docs/research/physics_v2_clean10_baseline_freeze.json`.
Raw numbers: `artifacts/physics_v2_generalization/summary.json` and `RESULTS_DIGEST.txt`.

---

## 1. Frozen physics-v2 baseline
The starting point is the physics-v2 CLEAN10 **memorization** baseline: this same TinyRDT (2.01M policy parameters plus a 0.94M frozen
MobileNet encoder) scored 49/50 closed-loop on the ten scenes it was trained on. Its artifacts were never modified. The freeze file records
the git commit, the effective MuJoCo collision-model hash `ce879457…`, environment and expert source hashes, the dataset aggregate hash,
checkpoint hashes, normalization, camera and training seed (17).

## 2. Pre-registered split
Fixed before any validation or test episode existed:

| set | n | construction |
|---|---|---|
| TRAIN80 | 80 | the 100 physics-v2 demos minus an interior hole band |
| TRAIN40 ⊃ TRAIN20 ⊃ CLEAN10 | 40 / 20 / 10 | nested, farthest-point sampling from CLEAN10 |
| VAL10 | 10 | fresh positions in the workspace, outside the band |
| TEST A (interpolation) | 20 | fresh positions in the workspace, outside the band |
| TEST B (sparse interpolation) | 20 | the hole-band dataset positions, y ∈ [16.3, 46.6] mm, spanning the full x range |
| TEST C (extrapolation) | 20 | fresh positions 3–15 mm outside the sampling rectangle |

The hole is centred on CLEAN10's largest interior y-gap, so no CLEAN10 position is removed and every B position is bracketed in y by
training data. B is genuine *interpolation* geometrically, but its nearest training cube is 4.7–16.6 mm away instead of the usual 4.3 mm.

## 3. Workspace
Cubes are uniform in x ∈ [235, 280] mm, y ∈ [−70, 70] mm, yaw 0. The 100 demos have a median nearest-neighbour distance of 4.3 mm
(max 11.9), and 99% of the workspace lies within 10 mm of a demo. CLEAN10 covers only y ∈ [−31, 69] mm with a 44 mm interior gap, a median
nearest-neighbour distance of 17 mm, and 34% coverage within 10 mm.

Figures: `docs/assets/v2_workspace_positions.png`, `docs/assets/v2_generalization_split.png`.

## 4. TRAIN80 statistics
80 episodes, 4,328 windows (one per frame), median episode length 54 steps (2.7 s). Nested subsets: CLEAN10 544 windows, TRAIN20 1,084,
TRAIN40 2,165.

## 5. Validation and test statistics
- VAL10: 10/10 expert successes.
- TEST: 60 positions proposed, **56 in the benchmark**.
- The 4 excluded positions are all category C at x ≥ 289 mm, where the expert's IK returns `PLAN_INFEASIBLE` (an arm reach limit). Under the
  pre-registered rule they are **outside the benchmark**, not policy failures. They were never replaced or moved.
- Final benchmark: **A 20, B 20, C 16**.

## 6. Nearest-training distances (mm, min / median / max)

| set | to TRAIN80 | to CLEAN10 |
|---|---|---|
| VAL10 | 2.0 / 6.2 / 8.8 | — |
| TEST A | 2.2 / 4.3 / 8.7 | 3.1 / 13.6 / 42.7 |
| TEST B | 4.7 / 10.5 / 16.6 | 13.3 / 19.6 / 27.0 |
| TEST C | 6.1 / 14.1 / 22.0 | 10.6 / 21.2 / 52.8 |

## 7. Expert held-out control
The unchanged expert (`expert-v2-sidepinch-3`) solves **A 20/20, B 20/20, C 16/20**, and VAL 10/10.
- Every success holds a valid side pinch through the hold; zero vertical pad contacts.
- Max penetrations over all held-out episodes: robot–table 0.00 mm, cube–table 0.28 mm, pad–cube 0.95 mm, all inside the physics-v2
  tolerances (1.0 / 1.0 / 1.5 mm).
- Median time to grasp 2.0 s, median time to success 2.7 s.
- `data.validate_v2`: 0 errors on both the validation and test datasets.
- **Re-collection determinism:** the 20 B positions were re-collected from their recorded cube XY. Actions, joint states, gripper and RGB are
  bit-identical to the original dataset episodes in all 20. In 4, the recorded cube pose differs by ≤ 3e-11 m (float noise in the
  near-identity quaternion); nothing else differs.
- **Scene control:** open-loop expert replay succeeds **56/56** at the test positions, and every reset image is bit-identical to the recorded
  first frame (max abs difference 0).
- **Rollout determinism:** re-running a rollout with the same checkpoint, position, policy seed and sampler reproduces the executed actions
  bit-exactly, as pre-registered. Identical conditions were therefore never re-run; repeats use different sampler seeds.

## 8. Training configuration
Identical to the frozen baseline except for which episodes are used: TinyRDT hidden 192, 4 layers, 6 heads, H = 16, frozen pooled MobileNet
encoder, cosine schedule, x0 prediction, hold-last-action padding, EMA 0.999, AdamW lr 1e-3, batch 8, seed 17, fp32, final EMA weights, no
early stopping, no checkpoint selection. No architecture change, no DAgger, no spatial tokens, no vision fine-tuning, no language.

## 9. Training budget and sample exposure

| model | demos | windows | steps | exposures | final EMA train MAE |
|---|---|---|---|---|---|
| CLEAN10 (frozen baseline) | 10 | 544 | 20,000 | 294 | 0.0049 |
| TRAIN20 | 20 | 1,084 | 20,000 | 148 | 0.0060 |
| TRAIN40 | 40 | 2,165 | 20,000 | 74 | 0.0070 |
| TRAIN80 (**pre-registered primary**) | 80 | 4,328 | 20,000 | 37 | 0.0073 |
| TRAIN80_80k (**pre-registered secondary arm**) | 80 | 4,328 | 80,000 | 148 | 0.0072 |

The primary rule was a fixed 20k-step budget at every dataset size, equalising updates and compute. The 80k arm was registered in advance to
test whether 37 epochs under-trains TRAIN80. **It does — and §10–12 show the offline metrics cannot tell.**

## 10. Offline: training-set error
DDIM-10, seed 4100, every window, unpadded tokens only.

| model | train MAE | arm (rad) | gripper | episode-start MAE |
|---|---|---|---|---|
| CLEAN10 | 0.0049 | 0.0054 | 0.0025 | 0.0051 |
| TRAIN20 | 0.0060 | 0.0066 | 0.0035 | 0.0147 |
| TRAIN40 | 0.0070 | 0.0076 | 0.0043 | 0.0159 |
| TRAIN80 | 0.0073 | 0.0078 | 0.0049 | 0.0194 |
| TRAIN80_80k | 0.0072 | 0.0076 | 0.0051 | 0.0172 |

## 11. Offline: validation error

| model | val MAE | val ÷ train |
|---|---|---|
| CLEAN10 | 0.0304 | 6.2× |
| TRAIN20 | 0.0088 | 1.5× |
| TRAIN40 | 0.0079 | 1.1× |
| TRAIN80 | 0.0077 | 1.05× |
| TRAIN80_80k | 0.0072 | 1.00× |

TRAIN80's validation curve falls monotonically (4k 0.0142, 8k 0.0111, 12k 0.0095, 16k 0.0085; 20k 0.0077 and 24k 0.0072 in the 80k run).
There is no overfitting: the model is still improving when the primary budget ends.

## 12. Offline: held-out test error

| model | test MAE | A | B | C | test start MAE | arm (rad) | Spearman ρ vs own nearest-training distance |
|---|---|---|---|---|---|---|---|
| CLEAN10 | 0.0378 | 0.0295 | 0.0186 | 0.0722 | 0.0315 | 0.0431 | 0.63 (p < 1e-4) |
| TRAIN20 | 0.0136 | 0.0099 | 0.0097 | 0.0233 | 0.0245 | 0.0151 | 0.49 (p < 1e-4) |
| TRAIN40 | 0.0112 | 0.0083 | 0.0084 | 0.0184 | 0.0248 | 0.0123 | 0.51 (p = 1e-4) |
| TRAIN80 | 0.0104 | 0.0081 | 0.0083 | 0.0158 | 0.0283 | 0.0113 | 0.47 (p = 2e-4) |
| TRAIN80_80k | 0.0101 | 0.0077 | 0.0084 | 0.0154 | 0.0255 | 0.0110 | 0.54 (p < 1e-4) |

**Offline prediction generalizes spatially.** With 80 demos the held-out error is 1.4× the training error, against 7.7× for CLEAN10. Error
rises with novelty in every model (**H4 supported**), and extrapolation is always worst.

Two cautions that §13–14 turn into the main result:
- **TRAIN80 at 20k and 80k are offline-indistinguishable** (test 0.0104 vs 0.0101) yet behave completely differently in closed loop.
- Episode-start error is ~3× the window-averaged error for every 80-demo model. At t = 0 only the image can say where the cube is; once the
  arm is moving, the state itself predicts the continuation, so the averaged metric flatters the policy.

## 13. Held-out closed loop, by K
All 56 benchmark positions, seed 0, physics-v2 validity enforced. Counts are successes / 56.

| model | K=1 | K=2 | K=4 | **K=8 (primary)** | K=16 |
|---|---|---|---|---|---|
| CLEAN10 (10 demos, 20k) | 13 | 15 | 9 | **13** | 11 |
| TRAIN20 (20k) | — | — | — | **17** | — |
| TRAIN40 (20k) | — | — | — | **13** | — |
| TRAIN80 (20k, **primary**) | 0 | 26 | 13 | **9** | 8 |
| TRAIN80_80k (80k) | 29 | 33 | 29 | **28** | 24 |

The whole sweep is reported, and K = 8 was fixed in advance. Note how differently K behaves here than in the memorization benchmark, where
every K ≥ 2 was perfect:
- the under-trained TRAIN80 is wildly K-sensitive (0 at K=1, 26 at K=2);
- the converged 80k model works at every K (24–33 of 56), peaking at K=2;
- CLEAN10 is flat at ~20–27% regardless of K, the signature of a model whose *target* is wrong: replanning more often cannot fix it.

Per-category for the 80k model (successes / n):

| K | A (20) | B (20) | C (16) |
|---|---|---|---|
| 1 | 16 | **13** | 0 |
| 2 | **20** | 8 | 5 |
| 4 | **20** | 3 | 6 |
| 8 | **20** | 4 | 4 |
| 16 | **20** | 1 | 3 |

Dense interpolation is solved at every K ≥ 2. The hole band is the opposite: it is best at K=1 (13/20) and degrades as the policy commits to
longer chunks — an open-loop chunk aimed at the wrong place cannot be corrected, and in the hole the initial aim is wrong.

## 14. Primary-K result and what it means
**Pre-registered primary metric — TRAIN80, K=8, seed 0: 9/56 = 16.1%** (Wilson 95% CI 8.7–27.8), by category A 7/20, B 1/20, C 1/16.
**H1 (A ≥ 80%) is rejected for the primary configuration.**

That number is not a statement about spatial generalization, because of two findings that arrived after it:

1. **A post-hoc diagnostic on training scenes only** (`artifacts/physics_v2_generalization/diagnostic_train_scenes/`): TRAIN80 at 20k solves
   just **7/20 of the scenes it was trained on** (4/10 of the CLEAN10 scenes that the CLEAN10 model solves 10/10). The same recipe at 80k
   solves **18/20** (10/10 on CLEAN10 scenes). The primary model cannot reliably execute its own training data.
2. **Sampler-seed sensitivity.** TRAIN80 at 20k scores 9, 17, 18 of 56 over sampler seeds 0/1/2 — pooled 44/168 = 26.2%
   (cluster-bootstrap 95% CI 18.5–34.5). Seed 0 was an unlucky draw. CLEAN10 is stable (13, 11, 13; pooled 22.0%).

**With the exposure-matched budget the pre-registered secondary arm gives the real answer: TRAIN80_80k, K=8, seed 0: 28/56 = 50%**
(CI 37.3–62.7), by category **A 20/20 (100%, CI 83.9–100)**, B 4/20 (20%), C 4/16 (25%). Across seeds: 28, 30, 27 — pooled 85/168 = 50.6%
(CI 38.1–63.1), with A 20/20 in every seed.

**Position difficulty is structural, not stochastic.** Over the three sampler seeds, 27 positions succeed every time, 26 fail every time, and
only 3 are inconsistent. The same partition of the workspace is reproduced by an independent sampler; the failures are geometric.

## 15. Physical validity
Every counted success satisfies physics-v2: a valid opposing side pinch held 5 steps, contact normals within the pre-registered tolerance,
the cube lifted to z ≥ 0.10 m, and no penetration tolerance exceeded at any step. The environment enforces this, so no v1-style sandwich
grasp can re-enter the benchmark.

| model | rollouts | successes | successes with invalid reason | max robot–table | max cube–table | max pad–cube | failures where the cube still lifted ≥ 30 mm |
|---|---|---|---|---|---|---|---|
| CLEAN10 | 392 | 85 | 0 | 0.00 mm | 1.12 mm | 1.49 mm | 4 |
| TRAIN20 | 56 | 17 | 0 | 0.00 mm | 0.73 mm | 1.28 mm | 1 |
| TRAIN40 | 56 | 13 | 0 | 0.00 mm | 0.73 mm | 1.27 mm | 0 |
| TRAIN80 | 392 | 91 | 0 | 0.00 mm | 0.77 mm | 1.49 mm | 11 |
| TRAIN80_80k | 392 | 200 | 0 | 0.013 mm | 0.77 mm | 1.43 mm | 3 |

- **One invalid-physics rollout in the whole study** (CLEAN10, test position 26, K=16): cube–table penetration 1.12 mm exceeded the 1.0 mm
  tolerance, the environment flagged `invalid_reason` and the rollout was **not** counted as a success. That is the tolerance working.
- Vertical pad contact occurred in 7 of the 80k model's 200 successes (transient, pre-grasp, as in the memorization baseline); the hold
  itself is always a valid side pinch.
- Robot–table penetration is essentially zero everywhere, so the v1 defect has not reappeared.

## 16. Failure breakdown
Pre-registered taxonomy, first matching rule. TRAIN80_80k at K=8, seed 0 (28 failures):

| category | A | B | C |
|---|---|---|---|
| success | 20 | 4 | 4 |
| wrong_lateral_alignment | 0 | 14 | 10 |
| early_close | 0 | 2 | 1 |
| cube_nudge_displacement | 0 | 0 | 1 |

**Essentially one failure mode: the gripper closes in the wrong place.** Median lateral offset at the first close is 6.5 mm for successes and
**20.0 mm** for failures, against an expert range of 0.9–2.0 mm. The cube's half-width is 10 mm, so a 20 mm miss is not a marginal grasp — the
policy aimed somewhere else. Trajectory deviation from the expert demo at the same position agrees: 0.025 rad for successes, 0.057 for
failures.

At K=1 the 80k model swaps failure modes: 13 of its failures become `cube_tilt` (the gripper contacts and tips the cube) rather than clean
misses, which is why K=1 helps in the hole band but is worst in extrapolation.

## 17. Performance vs spatial novelty
TRAIN80_80k at K=8, by nearest-TRAIN80 distance:

| nearest training cube | < 5 mm | 5–10 mm | 10–15 mm | ≥ 15 mm |
|---|---|---|---|---|
| success | 13/15 (87%) | 10/19 (53%) | 5/16 (31%) | 0/6 (0%) |

Logistic slope **−2.87 per 10 mm** (bootstrap 95% CI −5.04 to −1.63): success falls steeply and monotonically with geometric novelty.
CLEAN10 shows the same pattern against its own distances (slope −4.76, CI −19.97 to −2.41; 7/7 within 5–10 mm, 0/34 beyond 15 mm).
Offline error rises with the same variable (§12, ρ = 0.47–0.63). **H2 is supported as a distance effect**; by category, B (20%) and C (25%)
are statistically indistinguishable from each other, so the ordering A ≫ B ≈ C holds rather than A ≥ B ≥ C.

The practical threshold is sharp: this policy needs a demonstration within roughly **5 mm** for reliable success, and is at chance beyond
15 mm. The training set's median spacing is 4.3 mm, which is exactly why A is solved.

**Why the hole is hard (post-hoc diagnostic).** A ridge readout of the *frozen* pooled visual features, fit on TRAIN80 first frames only,
locates the cube to a median of 4.3 mm on A, 7.9 mm on B and 11.6 mm on C (p90 17.4 mm). So the visual features do carry position information
into the hole with a ~5 mm bias, while the policy's own closes are off by ~18 mm there and are biased consistently toward where training data
resumes. The bottleneck is mostly the learned action mapping, which does not interpolate across a 30 mm gap, not the frozen encoder alone —
though a p90 of 17 mm shows the pooled feature is itself coarse relative to the ~7 mm grasp tolerance.

## 18. CLEAN10 vs TRAIN80 on identical held-out scenes
All at K=8, seed 0, on the same 56 positions, exact McNemar:

| comparison | successes | discordant | p (two-sided) |
|---|---|---|---|
| TRAIN80 (20k) vs CLEAN10 (**pre-registered primary comparison**) | 9 vs 13 | 7 / 11 | 0.48 |
| TRAIN80_80k vs CLEAN10 | **28 vs 13** | 16 / 1 | **0.00027** |
| TRAIN80_80k vs TRAIN80 (20k) | **28 vs 9** | 20 / 1 | **2.1e-5** |
| TRAIN80_80k vs CLEAN10, category A only | 20 vs 11 | 9 / 0 | 0.0039 |

**H3 is not supported at the pre-registered budget and is strongly supported at matched exposure.** The honest statement is: *80
demonstrations beat 10 only when the optimisation budget grows with the dataset.* At a fixed 20k steps the data-scaling curve is flat or
declining — 13, 17, 13, 9 of 56 for 10, 20, 40, 80 demos — because each additional demonstration buys fewer passes over itself. This is a
budget artifact, not evidence against data.

It is also worth stating plainly what CLEAN10's 13/56 is: a memorizing model that happens to sit near 11 of the 20 A positions. Its offline
held-out error is 7.7× its training error, it never solves a single hole-band position in 3 seeds (0/60), and its failures close within
3.7 mm of one of its own 10 training cubes 98% of the time — it drives to a memorized location. The 80-demo model fails differently: it aims
at a plausible but wrong place in regions it never saw.

## 19. Representative rollouts
Selected by the pre-registered rules (largest / smallest nearest-training distance), rendered from the task camera plus a side camera so the
grasp type is visible. Re-simulating the recorded actions reproduces each outcome exactly.

| file | model | position | result |
|---|---|---|---|
| `docs/assets/v2_gen_80k_heldout_success.gif` | TRAIN80_80k | C, ep41, 13.8 mm from training | valid side pinch, lifted |
| `docs/assets/v2_gen_80k_heldout_failure.gif` | TRAIN80_80k | B, ep32, 4.7 mm from training | wrong lateral alignment |
| `docs/assets/v2_gen_heldout_success.gif` | TRAIN80 (20k) | C, ep45, 12.8 mm | valid side pinch, lifted |
| `docs/assets/v2_gen_heldout_failure.gif` | TRAIN80 (20k) | A, ep4, 2.9 mm | wrong lateral alignment |

Figures: `docs/assets/v2_gen_success_map.png` (nearest-distance map and success/failure maps for all three models),
`v2_gen_distance.png` (success and offline error vs novelty), `v2_gen_scaling.png` (K sweep and data scaling).

## 20. Limitations
- **n = 56 positions**, 20/20/16 per category. Differences of a few rollouts are not interpretable; all CIs are reported.
- **The primary configuration under-trained**, so the pre-registered primary number (9/56) mostly measures convergence, not generalization.
  The 80k arm was pre-registered, but it is a *secondary* arm; a reader should weight it as such.
- **Budget and data are confounded** in the 10/20/40/80 curve at 20k steps. An exposure-matched scaling curve was not run.
- **One environment, one task, one cube, fixed yaw, fixed camera, no distractors, no start-pose variation.** "Generalization" here means cube
  position only.
- **Extrapolation is only 3–15 mm outside the rectangle**, and 4 candidate positions were removed because the *expert* could not reach them.
  C therefore measures a narrow band just beyond the training support.
- The probe is a linear readout of a frozen encoder, so it lower-bounds what the features contain; a nonlinear readout might do better.
- Sampler-seed repeats cover K=8 only.
- Rendering is CPU-bound (~1 s per simulated step), which capped the study at ~2,100 rollouts; a GPU rendering path was rejected because it
  changes pixels (max abs difference 64/255) and would have broken comparability with the training data.

## 21. Exact reproduction
```bash
# 1. the frozen split (regenerates byte-identically; a test asserts this)
.venv/bin/python -m evaluation.workspace_analysis
.venv/bin/python -m data.generalization_split

# 2. expert control: collect the held-out positions (the workspace margin widens ONLY the reset bounds check)
.venv/bin/python -m data.collect_v2 --positions docs/research/physics_v2_generalization_split.json --split validation \
    --output artifacts/pickcube_physics_v2_gen_val_rgb160 --dataset-version pickcube-physics-v2-gen-val-1 --workspace-margin 0.015
.venv/bin/python -m data.collect_v2 --positions docs/research/physics_v2_generalization_split.json --split test \
    --output artifacts/pickcube_physics_v2_gen_test_rgb160 --dataset-version pickcube-physics-v2-gen-test-1 --workspace-margin 0.015
.venv/bin/python -m data.validate_v2 artifacts/pickcube_physics_v2_gen_test_rgb160

# 3. training (same recipe as the frozen baseline; --train-subset selects the spatial subset)
.venv/bin/python -m training.audit_overfit --dataset artifacts/pickcube_physics_v2_rgb160 \
    --output artifacts/physics_v2_generalization/models/TRAIN80 --train-subset TRAIN80 \
    --schedule cosine --prediction x0 --padding hold --steps 20000 --seed 17 --batch-size 8 --learning-rate 0.001 \
    --device cuda --eval-interval 2000 --ema-decay 0.999 --snapshot-interval 4000
#   the 80k arm: identical, with --steps 80000 --output .../TRAIN80_80k

# 4. offline and closed loop
.venv/bin/python -m evaluation.offline_generalization --checkpoint <ckpt> --dataset artifacts/pickcube_physics_v2_gen_test_rgb160 \
    --output artifacts/physics_v2_generalization/offline/<model>_test.json
.venv/bin/python -m evaluation.closed_loop --policy tinyrdt --checkpoint <ckpt> \
    --dataset artifacts/pickcube_physics_v2_gen_test_rgb160 --episodes <ids> --k 1 2 4 8 16 --max-steps 150 \
    --workspace-margin 0.015 --output artifacts/physics_v2_generalization/closed_loop/<model>

# everything end to end (resumable, <= 2 simulation workers), then analysis:
scripts/experiments/run_generalization.sh
scripts/experiments/run_generalization_posthoc.sh     # post-hoc 80k repeats and K sweep
scripts/experiments/run_generalization_finish.sh      # summary.json, figures, media, RESULTS_DIGEST.txt
.venv/bin/python -m evaluation.generalization_analysis --media
.venv/bin/python -m evaluation.generalization_probe
env -u PYTHONPATH .venv/bin/python -m pytest -q tests/
```

## 22. Recommendation for the next experiment
Mapping the outcome onto the pre-registered decision rule: offline held-out accuracy is good, closed-loop held-out performance is partly
good (100% inside dense coverage) and poor outside it, and the limiting variable is **distance to the nearest demonstration**, not model
capacity. Nothing here argues for a bigger model yet: the 2M policy executes perfectly wherever it has data within ~5 mm.

Recommended next, in order:
1. **Fix the budget confound before anything else.** Re-run the 10/20/40/80 scaling curve at *matched exposure* (≈294 epochs each, so 20k /
   40k / 80k / 160k steps). This is the single missing experiment needed before any capacity claim, and it is cheap — 4 training runs plus
   4 × 56 rollouts at K=8.
2. **Attack the coverage threshold directly.** The cleanest test of the "≈5 mm" hypothesis is a density sweep: train on 80 demos drawn at
   deliberately different spacings (or re-fill the hole band at known density) and measure where success collapses. That distinguishes "needs
   dense data" from "cannot interpolate".
3. **Then, and only then, ask about representation.** The frozen pooled feature localizes the cube to a median of 4.3 mm but a p90 of 17 mm.
   If (1) and (2) show the model saturates while the probe's p90 stays coarse, the encoder/token layout becomes the bottleneck worth
   attacking — which is where spatial tokens or a fine-tuned encoder would earn their place.
4. Report K = 2 alongside K = 8 in future held-out benchmarks. On memorized scenes K did not matter; off-distribution it does.

**Stopping here, as pre-registered. No model-capacity scaling was started.**
