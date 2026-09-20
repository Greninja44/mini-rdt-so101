# Physics-v2 spatial generalization: pre-registered protocol

**Registered 2026-09-19.** This spec was committed before any validation or test demonstration was collected and before any policy was
trained on or evaluated at a held-out position. Nothing below is changed after held-out results are seen. A genuine benchmark bug is handled
by versioning the benchmark and documenting the change, never by editing this file.

## Question
Can the **same ~2M TinyRDT** pick the cube at cube positions that were **not in its training demonstrations**?

What does not change: physics-v2, the expert, the camera and renderer, the architecture, the diffusion recipe, and the sampler. There is no
model scaling, no DAgger, no spatial tokens, no vision fine-tuning and no language.

## Frozen baseline
`docs/research/physics_v2_clean10_baseline_freeze.json` records the **physics-v2 CLEAN10 memorization baseline** (49/50 on its own training
scenes):
- git commit;
- effective-model hash, source hashes of the environment and expert, and dataset aggregate hash;
- TinyRDT config: 2.01M policy parameters plus a 0.94M frozen encoder;
- checkpoint hashes, normalization, camera, and training seed.

Its artifacts (`artifacts/physics_v2/tinyrdt_clean10`, `artifacts/physics_v2/closed_loop`) are read-only. All new work lives in
`artifacts/physics_v2_generalization/`.

## Hypotheses (predictions, reported whatever the outcome)
- **H1:** TRAIN80 reaches ≥ 80% closed-loop success on category-A (dense interpolation) test positions at the primary K.
- **H2:** For TRAIN80, success falls with geometric novelty: rate(A) ≥ rate(B) ≥ rate(C), and the logistic slope of success against
  nearest-TRAIN80 distance is negative.
- **H3:** On the identical held-out positions, TRAIN80 beats CLEAN10 (paired comparison at the primary K).
- **H4:** Offline held-out action error rises with nearest-training distance (positive Spearman ρ across test episodes).

## Workspace (inspected before the split; `evaluation/workspace_analysis.py`)
- Cubes are sampled uniformly in x ∈ [235, 280] mm, y ∈ [−70, 70] mm, with yaw fixed at 0.
- The 100 demos have a median nearest-neighbour distance of 4.3 mm (max 11.9 mm), and 99% of the workspace lies within 10 mm of a demo.
- CLEAN10 covers only y ∈ [−31, 69] mm, with an interior gap of 44 mm. Its median nearest-neighbour distance is 17 mm, and 34% of the
  workspace lies within 10 mm of a CLEAN10 demo.

## Split (`data/generalization_split.py` → `docs/research/physics_v2_generalization_split.json`, figure `docs/assets/v2_generalization_split.png`)
The split is deterministic, uses cube positions only, and has no duplicate coordinates (every fresh position is ≥ 2 mm from all others).

| set | n | construction | nearest-TRAIN80 distance (mm) min / median / max |
|---|---|---|---|
| **TRAIN80** | 80 | the 100 dataset positions minus the hole | — |
| TRAIN40 ⊃ TRAIN20 ⊃ CLEAN10 | 40 / 20 / 10 | nested; farthest-point sampling over TRAIN80 starting from CLEAN10 | — |
| **VAL10** | 10 | fresh positions, uniform in the workspace outside the hole band (rng 7100) | 2.0 / 6.2 / 8.8 |
| **TEST A: interpolation** | 20 | fresh positions, uniform in the workspace outside the hole band (rng 7200) | 2.2 / 4.3 / 8.7 |
| **TEST B: sparse / hard interpolation** | 20 | the 20 dataset positions nearest in y to the centre (y = 32.0 mm) of CLEAN10's largest interior y-gap; the band y ∈ [16.3, 46.6] mm spans the full x range, so each B position lies between training positions in y | 4.7 / 10.5 / 16.6 |
| **TEST C: extrapolation** | 20 | fresh positions 3–15 mm outside the sampling rectangle (rng 7300) | 6.1 / 14.1 / 22.0 |

Novelty descriptors are stored per position in the split file before any evaluation:
- nearest distance to CLEAN10, TRAIN20, TRAIN40 and TRAIN80;
- distance to the TRAIN80 centroid;
- number of TRAIN80 positions within 10 mm and 20 mm;
- distance outside the workspace.

B's demonstrations are re-collected from their recorded cube XY. They must reproduce the original dataset episodes bit-exactly, which is a
determinism check. B positions are never trained on by any model in this study.

## Expert control (before any policy evaluation)
- The unchanged physics-v2 expert (`expert-v2-sidepinch-3`) collects the demonstrations at every VAL and TEST position:
  `data/collect_v2.py --positions`.
- For C, `--workspace-margin 0.015` widens **only** the reset bounds check. Physics, camera and expert are untouched.
- The collector records success, side-pinch validity, penetration, time to grasp and time to success. The datasets are validated with
  `data.validate_v2`.
- **A position the expert cannot solve is marked "outside the benchmark"** and excluded from every policy metric. It is never replaced or moved.

## Training
- The same architecture and recipe as the CLEAN10 baseline:
  - TinyRDT: hidden 192, 4 layers, 6 heads, H = 16;
  - frozen MobileNet encoder with pooled tokens (the same encoder checkpoint);
  - cosine schedule, x0 prediction, hold-last-action padding, EMA 0.999;
  - AdamW lr 1e-3, batch 8, training seed 17, fp32.
- **Training-budget rule:**
  - **Primary: every training-set size gets 20,000 optimisation steps** (the validated recipe). That means identical updates, compute and
    batch size, which is the conventional fixed-budget data-scaling design.
  - Sample exposure differs and is reported: windows = frames, and exposure = 20,000 × 8 / windows.

| model | windows | epochs at 20k steps |
|---|---|---|
| CLEAN10 (existing baseline, not retrained) | 544 | 294 |
| TRAIN20 | 1,084 | 148 |
| TRAIN40 | 2,165 | 74 |
| TRAIN80 | 4,328 | 37 |

- **Pre-registered secondary budget arm: TRAIN80 at 80,000 steps** (148 epochs), with an identical recipe otherwise. It tests whether
  37 epochs under-trains TRAIN80. It is evaluated at the primary K only and reported alongside, never substituted for, the primary TRAIN80
  result.
- **Checkpoint:** the final EMA weights (`ema_last.pt`) are always used. There is no early stopping and no checkpoint selection.
  - EMA snapshots are saved every 4,000 steps for an offline validation-error curve. This is a diagnostic only.
  - The test set is never used during training or for any training decision.

## Offline evaluation (`evaluation/offline_generalization.py`)
- For each model, on TRAIN (its own training episodes), VAL10 and TEST (A and B have demos; C has demos wherever the expert succeeded).
- **Method:** DDIM-10 sampling with seed 4100 over every window, no clipping, error measured on valid (unpadded) action tokens only.
- **Reported:**
  - action MAE;
  - per-joint MAE;
  - MAE by episode quarter and by expert phase (from `expert_state.npy`);
  - episode-start (t = 0) MAE;
  - per-episode MAE, for the novelty analysis.
- The CLEAN10 memorization gate thresholds are reported for reference only (all-window 0.02; arm joint 0.03 rad; start 0.025). They are not
  a pass/fail criterion here.

## Closed-loop evaluation
- Protocol:
  - `evaluation/closed_loop.py`;
  - physics-v2 success (valid side pinch held for 5 steps, cube z ≥ 0.10 m, no tolerance ever exceeded);
  - max 150 steps, DDIM-10, H = 16;
  - bit-exact software renderer (the Mesa llvmpipe path used for all training data);
  - an open-loop expert replay control at every test position.
- **Primary K = 8, fixed in advance.** Justification, using only earlier physics-v2 CLEAN10 evidence:
  - K = 8 was 10/10 in the memorization benchmark;
  - it is the midpoint of the K ≥ 2 range that was all perfect;
  - it matches the K used for the README media.
- The whole K ∈ {1, 2, 4, 8, 16} sweep is reported. **No "best K" is selected after the fact.**
- **Determinism and repeats:**
  - For a fixed checkpoint, position, policy seed and sampler, a rollout is deterministic: MuJoCo is deterministic and the sampler noise is
    seeded by `policy_seed + call index`. This is verified once by re-running one rollout and comparing the executed actions.
  - Identical conditions are therefore **not** re-run. Repeats use different policy seeds (sampler noise) only.

**Evaluation order (priority tiers). All tiers are planned. If time runs out, the report states which tiers are complete.**

| tier | model | K | policy seeds | rollouts |
|---|---|---|---|---|
| 1 | TRAIN80 and CLEAN10 (+ expert replay) | 8 | 0 | 2 × 60 |
| 2 | TRAIN80 | 1, 2, 4, 16 | 0 | 240 |
| 3 | TRAIN80 and CLEAN10 | 8 | 1, 2 | 240 |
| 4 | TRAIN20, TRAIN40, TRAIN80-80k | 8 | 0 | 180 |
| 5 | CLEAN10 | 1, 2, 4, 16 | 0 | 240 |

At most two simulation workers run at once. Media (GIFs) is saved for tiers 1–2 only.

## Metrics and statistics
- **Primary metric:** TRAIN80 success at K = 8, policy seed 0, on all expert-valid test positions, with a Wilson 95% CI. It is also reported
  per category (A, B, C).
- **Primary comparison:** CLEAN10 vs TRAIN80 at K = 8, seed 0, on identical positions, using an exact McNemar test (two-sided binomial on
  discordant pairs).
- **Secondary metrics:**
  - the full K sweep;
  - 3-seed pooled success, with a position-cluster bootstrap CI (10,000 resamples);
  - the data-scaling curve (10 / 20 / 40 / 80), with Wilson CIs;
  - the 80k budget arm;
  - steps to success;
  - physical-validity flags;
  - trajectory deviation from the expert demo at the same position (mean |executed − expert| over the common prefix).
- **Success against novelty:**
  - binned success rates by nearest-TRAIN80 distance (fixed bins: < 5, 5–10, 10–15, ≥ 15 mm), with Wilson CIs;
  - a logistic-regression slope (per 10 mm) with a position-bootstrap 95% CI;
  - Spearman ρ between offline per-episode MAE and distance, with a permutation p-value (10,000 permutations).
  - For CLEAN10, the same analysis uses nearest-CLEAN10 distance.
- Raw counts are always shown. With n = 20 per category, a difference of a few rollouts is not interpreted.

## Physical validity
- Every counted success must satisfy the physics-v2 success definition, which is enforced by the environment.
- Reported separately, even for failures where the cube lifted:
  - any vertical pad contact;
  - max robot–table, cube–table and pad–cube penetration;
  - `invalid_reason`.
- A v1-style (sandwich) success cannot be counted: the environment requires a side pinch during the hold.

## Failure taxonomy
- **Thresholds.** They were measured beforehand on training-scene rollouts only (CLEAN10 expert replays and the 49 CLEAN10 memorization
  successes).
  - At the first close command, the expert's grasp centre is 0.9–2.0 mm lateral and 7.0–8.2 mm above the cube centre.
  - Successful policy rollouts are ≤ 7.2 mm lateral and 5.8–9.2 mm above the cube centre.
- The **first matching rule** assigns the primary category:
  1. `invalid_physics`: a penetration tolerance was exceeded.
  2. `cube_tilt`: before the first close (or at the end, if the gripper never closed), the cube tilted by > 20°.
  3. `cube_nudge_displacement`: before the first close (or at the end), the cube moved by > 5 mm in xy.
  4. `approach_positioning`: never closed, and never at a closable pose (lateral ≤ 8 mm and height 2–14 mm above the cube centre).
  5. `never_closed_at_closable_pose`: reached a closable pose but never commanded a close (the hover mode of the CLEAN10 ep0 K=1 failure).
  6. `early_close`: grasp centre > 20 mm above the cube centre at the first close.
  7. `wrong_lateral_alignment`: > 8 mm lateral at the first close.
  8. `wrong_height`: outside 2–14 mm (and ≤ 20 mm) above the cube centre at the first close.
  9. `failed_side_pinch`: closed at a closable pose but never formed a valid side pinch.
  10. `drop_after_grasp`: had a side pinch, lifted the cube ≥ 30 mm, then it fell below 20 mm.
  11. `lift_or_hold_failure`: had a side pinch but never met the success hold.
  12. `other`.
- "IK/workspace issue" applies only to the expert, whose failures are outside the benchmark.
- Non-exclusive flags are also reported: vertical pad contact, robot–table contact > 0.1 mm, cube displaced, and cube tilted.

## Visualizations (fixed selection rules)
- The split plot.
- A success/failure map at K = 8.
- A nearest-distance map.
- Success vs nearest-TRAIN80 distance.
- Offline error vs novelty.
- Representative GIFs, selected by rule:
  - **success:** the TRAIN80 K=8 seed-0 success with the largest nearest-TRAIN80 distance;
  - **failure:** the TRAIN80 K=8 seed-0 failure with the smallest nearest-TRAIN80 distance (the most surprising one).
  - Both are rendered from the task camera plus a side camera, so the side pinch is visible.

## Frozen after this commit
- Test positions, categories, the expert, the camera and renderer, the primary K, the model and budget, and the evaluation order.
- Failed test positions never move into training. Difficult positions are never removed.

## Stop condition
Stop after `PHYSICS_V2_GENERALIZATION_REPORT.md`. There is no model-capacity scaling.
