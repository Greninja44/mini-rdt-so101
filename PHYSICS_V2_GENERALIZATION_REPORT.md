# Physics-v2 spatial generalization: can the same ~2M TinyRDT pick cubes it never saw?

**Status: DRAFT — the pre-registered tiers 3–5 and the post-hoc runs are still executing. Numbers in the sections marked _(pending)_ are
incomplete and will be finalised before this report is considered done.**

Pre-registration: `docs/research/physics_v2_generalization_spec.md` (committed `c4b9559`, before any held-out demonstration existed).
Split: `docs/research/physics_v2_generalization_split.json`. Baseline freeze: `docs/research/physics_v2_clean10_baseline_freeze.json`.

## 1. Frozen physics-v2 baseline
The starting point is the physics-v2 CLEAN10 **memorization** baseline: the same 2.01M-parameter TinyRDT (plus a 0.94M frozen MobileNet
encoder) that scored 49/50 closed-loop on the ten scenes it was trained on. Its artifacts were not touched. The freeze file records the git
commit, the effective MuJoCo collision-model hash `ce879457…`, the environment and expert source hashes, the dataset aggregate hash, the
checkpoint hashes, the normalization, the camera and the training seed (17).

## 2. Pre-registered split
Written down before any validation or test episode was collected:

| set | n | construction |
|---|---|---|
| TRAIN80 | 80 | the 100 physics-v2 demos minus an interior hole band |
| TRAIN40 ⊃ TRAIN20 ⊃ CLEAN10 | 40 / 20 / 10 | nested, farthest-point sampling from CLEAN10 |
| VAL10 | 10 | fresh positions in the workspace, outside the band |
| TEST A (interpolation) | 20 | fresh positions in the workspace, outside the band |
| TEST B (sparse interpolation) | 20 | the hole-band dataset positions, y ∈ [16.3, 46.6] mm, spanning the full x range |
| TEST C (extrapolation) | 20 | fresh positions 3–15 mm outside the sampling rectangle |

The hole is centred on the largest interior y-gap of CLEAN10, so no CLEAN10 position is removed and every B position is bracketed in y by
training data.

## 3. Workspace
Cubes are sampled uniformly in x ∈ [235, 280] mm, y ∈ [−70, 70] mm with yaw 0. The 100 demos have a median nearest-neighbour distance of
4.3 mm (max 11.9 mm); 99% of the workspace is within 10 mm of a demo. CLEAN10 covers only y ∈ [−31, 69] mm with a 44 mm interior gap, a
median nearest-neighbour distance of 17 mm, and 34% coverage within 10 mm.

Figures: `docs/assets/v2_workspace_positions.png`, `docs/assets/v2_generalization_split.png`.

## 4. TRAIN80 statistics
80 episodes, 4,328 windows (= frames), median episode length 54 steps (2.7 s). Nested subsets: CLEAN10 544 windows, TRAIN20 1,084,
TRAIN40 2,165.

## 5. Validation and test statistics
- VAL10: 10/10 expert successes.
- TEST: 60 positions proposed, **56 in the benchmark**.
- The 4 excluded positions are all category C at x ≥ 289 mm: the expert's IK reports `PLAN_INFEASIBLE` (a reach limit), so under the
  pre-registered rule they are outside the benchmark rather than policy failures. They were not replaced or moved.
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
- Every success holds a valid side pinch through the hold, with zero vertical pad contacts.
- Max penetrations across all held-out episodes: robot–table 0.00 mm, cube–table 0.28 mm, pad–cube 0.95 mm — all inside the physics-v2
  tolerances (1.0 / 1.0 / 1.5 mm).
- Median time to grasp 2.0 s, median time to success 2.7 s.
- `data.validate_v2` reports 0 errors for both the validation and test datasets.
- **Re-collection determinism:** the 20 B positions were re-collected from their recorded cube XY. Actions, joint states, gripper and RGB
  are bit-identical to the original dataset episodes in all 20. In 4 of them the recorded cube pose differs by ≤ 3e-11 m (float noise in the
  near-identity quaternion); nothing else differs.
- **Closed-loop scene control:** the open-loop expert replay succeeds at 56/56 test positions, and every reset image is bit-identical to the
  recorded first frame (max abs difference 0).

## 8. Training configuration
Identical to the frozen baseline in every respect except the training episodes: TinyRDT hidden 192, 4 layers, 6 heads, H = 16, frozen pooled
MobileNet encoder, cosine schedule, x0 prediction, hold-last-action padding, EMA 0.999, AdamW lr 1e-3, batch 8, seed 17, fp32, final EMA
weights, no early stopping. No architecture change, no DAgger, no spatial tokens, no vision fine-tuning, no language.

## 9. Training budget and sample exposure

| model | demos | windows | steps | exposures (steps × batch / windows) | final EMA train MAE |
|---|---|---|---|---|---|
| CLEAN10 (frozen baseline) | 10 | 544 | 20,000 | 294 | 0.0049 |
| TRAIN20 | 20 | 1,084 | 20,000 | 148 | 0.0060 |
| TRAIN40 | 40 | 2,165 | 20,000 | 74 | 0.0070 |
| TRAIN80 (**pre-registered primary**) | 80 | 4,328 | 20,000 | 37 | 0.0073 |
| TRAIN80_80k (**pre-registered secondary budget arm**) | 80 | 4,328 | 80,000 | 148 | 0.0072 |

The primary rule was a fixed optimisation budget of 20k steps for every training-set size, which equalises updates and compute. The 80k arm
was registered in advance to test whether 37 epochs under-trains TRAIN80. **It does, and the offline metrics do not reveal it** (§10–12, §17).

## 10. Offline: training-set error
(DDIM-10, seed 4100, every window, unpadded tokens only.)

| model | train MAE | arm (rad) | gripper | episode-start MAE |
|---|---|---|---|---|
| CLEAN10 | 0.0049 | 0.0054 | 0.0025 | 0.0051 |
| TRAIN20 | 0.0060 | — | — | 0.0147 |
| TRAIN40 | 0.0070 | — | — | 0.0159 |
| TRAIN80 | 0.0073 | — | — | 0.0194 |
| TRAIN80_80k | 0.0072 | — | — | 0.0172 |

## 11. Offline: validation error
| model | val MAE | val ÷ train |
|---|---|---|
| CLEAN10 | 0.0304 | 6.2× |
| TRAIN20 | 0.0088 | 1.5× |
| TRAIN40 | 0.0079 | 1.1× |
| TRAIN80 | 0.0077 | 1.05× |
| TRAIN80_80k | 0.0072 | 1.00× |

TRAIN80's validation curve falls monotonically (4k 0.0142, 8k 0.0111, 12k 0.0095, 16k 0.0085, and at 20k/24k in the 80k run 0.0077/0.0072),
so there is no sign of overfitting; the model is still improving when the primary budget ends.

## 12. Offline: held-out test error
| model | test MAE | A | B | C | test start MAE | Spearman ρ (per-episode MAE vs own nearest-training distance) |
|---|---|---|---|---|---|---|
| CLEAN10 | 0.0378 | 0.0295 | 0.0186 | 0.0722 | 0.0315 | 0.63 (p < 1e-4) |
| TRAIN20 | 0.0136 | 0.0099 | 0.0097 | 0.0233 | 0.0245 | 0.49 (p < 1e-4) |
| TRAIN40 | 0.0112 | 0.0083 | 0.0084 | 0.0184 | 0.0248 | 0.51 (p = 1e-4) |
| TRAIN80 | 0.0104 | 0.0081 | 0.0083 | 0.0158 | 0.0283 | 0.47 (p = 2e-4) |
| TRAIN80_80k | 0.0101 | 0.0077 | 0.0084 | 0.0154 | 0.0255 | 0.54 (p < 1e-4) |

**Offline prediction generalizes spatially.** With 80 demos the held-out error is within 1.4× of the training error, versus 7.7× for
CLEAN10. Error grows with novelty in every model (H4 supported), and the extrapolation set is consistently the worst.

Note the episode-start error: at t = 0 only the image can say where the cube is, and the start error (0.025–0.028) is 3× the all-window
error for every 80-demo model. The window-averaged metric is partly carried by proprioceptive continuation once the arm is moving.

<!-- SECTIONS 13-22 PENDING: closed-loop K sweep, primary-K result, validity, failures, novelty, CLEAN10-vs-TRAIN80, media, limitations,
reproduction, recommendation. -->
