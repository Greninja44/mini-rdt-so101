# Closed-loop diagnostic report: why does offline-accurate TinyRDT fail closed-loop?

Date: 2026-09-18. Sweep: `artifacts/closed_loop_v2/` (130 rollouts). Analysis: `artifacts/closed_loop_v2/analysis/analysis.json`.
Policies: TinyRDT EMA (`research_audit/cosine_x0_hold_ema/ema_last.pt`, offline all-window MAE 0.0067),
BC image+state (`bc_rgb/best.pt`), BC privileged state + true cube pose (`bc_privileged/best.pt`).
All three use the same 10 memorised seeds (3000 + {0,2,3,4,5,6,7,8,9,11}), K ∈ {1,2,4,8}, H=16, DDIM-10, a 150-step limit,
and identical simulator settings. The analysis definitions were fixed before the results were read (docstring of `evaluation/closed_loop_analysis.py`).

## 1. TinyRDT K sweep (balanced: every K on the same 10 seeds)

| K | episodes | successes | failures | success rate |
|---|---|---|---|---|
| 1 | 10 | 5 | 5 | 0.50 |
| 2 | 10 | 6 | 4 | 0.60 |
| 4 | 10 | 8 | 2 | 0.80 |
| 8 | 10 | 9 | 1 | 0.90 |
| all | 40 | 28 | 12 | 0.70 |

The trend is monotone: less frequent replanning succeeds more often. With n=10 per K this is **not** a statistically established
ranking (two-sided Fisher exact: K=1 vs K=8 p=0.141; pooled K≤2 11/20 vs K≥4 17/20 p=0.082). Per seed (successes out of 4 K values): ep0 1, ep2 4, ep3 3, ep4 4, ep5 3, ep6 4,
**ep7 0**, ep8 3, ep9 4, ep11 2. The sweep is deterministic: ep6 K=1/2 reproduce the earlier interrupted sweep step for step.

## 2. Expert replay control
10/10 success. The reset RGB frame is bit-identical to the recorded frame 0 on all 10 seeds. The simulator, reset and evaluator are consistent.

## 3. Image+state BC closed-loop
**0/40** (0/10 at every K). 2/40 pinch the cube, 0/40 lift it. It deviates immediately (0.03 rad onset at step 1).

## 4. Privileged (true cube pose + state) BC closed-loop
**2/40** (K=1 1/10, K=8 1/10). 9/40 pinch, 11/40 lift, 9/40 drop.

## 5. Comparison

| | success | pinched | lifted ≥3 cm | dropped | offline worst-joint MAE (rad) |
|---|---|---|---|---|---|
| Expert replay | 10/10 | 10/10 | 10/10 | 0 | — |
| BC image+state | 0/40 | 2/40 | 0/40 | 0 | 0.034 (shoulder_lift) |
| BC privileged | 2/40 | 9/40 | 11/40 | 9 | 0.028 (elbow) |
| TinyRDT EMA | 28/40 | 29/40 | 29/40 | 1 | 0.016 |

**Caveat:** the BC baselines are not accuracy-matched controls. They were trained for 5k steps without EMA, and their offline
per-joint error is already at the measured 0.03 rad grasp tolerance. Their failure shows that small per-step errors are fatal in
closed loop, but it cannot separate "diffusion-specific" from "non-diffusion" behaviour.

## 6. Failure-phase distribution (rules fixed in advance)

| | approach | descent | grasp | lift | drop | other |
|---|---|---|---|---|---|---|
| TinyRDT (12 failures) | 0 | 0 | 11 | 0 | 1 | 0 |
| BC image+state (40) | 0 | 0 | 38 | 2 | 0 | 0 |
| BC privileged (38) | 0 | 0 | 31 | 2 | 5 | 0 |

"grasp" means the gripper closed but no two-pad pinch was achieved. Every policy reaches the cube region and closes; failures are
mis-aligned closes. Note, however, that **the divergence behind these failures starts in approach** (§7).

## 7. Divergence analysis (TinyRDT)
- Deviation beyond reconstruction noise (max joint error > 0.015 rad for 3 steps) appears by **step ~3–5 in every rollout**,
  successful or not. Grasp-center distance from the expert's path exceeds 5 mm by step ~3.5–4.
- The grasp tolerance of 0.03 rad is crossed at a median of **step 8 in failures, all 12 during approach** (steps 2–10), versus step 16
  in successes (2 successes never cross it).
- After that the two groups separate. Distance from the robot state to the nearest demo state is:

  | | step 5 | step 10 | step 30 |
  |---|---|---|---|
  | failures | 0.027 rad | 0.068 rad | 0.182 rad |
  | successes | 0.029 rad | 0.037 rad | 0.022 rad |

  Failures keep drifting. Successes stay about 0.03 rad off the demos and re-converge near the grasp.
- At K=1, time-aligned joint deviation reaches 0.4–0.8 rad during the fast approach, even in some successes. This is mostly *temporal
  lag*: the policy moves slower than the expert, whose IK steps saturate at 0.08 rad/step. The time-free path deviation is 26–37 mm.
- Per-rollout plots: `analysis/tinyrdt_ep*_k*_divergence.png`. Overlays: `analysis/*_overlay.png`.

## 8. Grasp sensitivity (physics-only replay with injected errors, `evaluation/grasp_sensitivity.py`)
Offsets are applied from DESCEND onward. Success out of 10 seeds:
- None: 10/10.
- ±0.03 rad on pan, elbow or wrist_flex: 10/10 (up to about 7 mm lateral). ±0.05 rad: 0–4/10.
- shoulder_lift −0.02 rad (+4.6 mm too high): 9/10. −0.03 rad (+6.6 mm): 2/10. +0.05 rad (−10 mm): 9/10.
- Gripper closed 1–5 steps late: 10/10. **1 step early: 4/10.**

TinyRDT at its first close command (median values; "early" means before the expert's close step):

| | lateral error | vertical error | joint deviation | early close | pinched |
|---|---|---|---|---|---|
| successes (n=28) | 4.6 mm (p90 7.8) | −0.6 mm | 0.034 rad | 2/28 | 28/28 |
| failures (n=12) | 21 mm (p90 33) | −1.9 mm | 0.150 rad | 5/12 | 1/12 |

Every failure closed 9.5–37 mm lateral of the cube, outside the ≈7 mm envelope. Failure is *lateral misalignment at close*, not height.
Even successes sit near the edge of the envelope (p90 7.8 mm).

## 9. Consecutive action-chunk disagreement
Measured as the jump in the next action between consecutive replans (max over arm joints, median). "On-distribution" means
replanning every step on the *recorded* expert observations.

| TinyRDT | approach | descent | grasp | lift |
|---|---|---|---|---|
| closed-loop | 0.019 (p90 0.051) | 0.019 | 0.013 | 0.033 |
| on-distribution | 0.010 (p90 0.018) | 0.015 | 0.008 | 0.016 |

Closed-loop disagreement is about 1.3–2× the on-distribution value (2.8× at p90 in approach). The BC baselines are much less
consistent even on-distribution (0.026–0.043).

## 10. Observation-distribution drift and prediction error
- State drift: see §7.
- Vision (pooled MobileNet features, nearest frame of the same demo):

  | | step 5 | step 30 |
  |---|---|---|
  | failures | 2.3 | 3.4 |
  | successes | 2.2 | 1.7 |

  For scale, adjacent demo frames are 2.7 apart and the nearest frame from a *different scene* is 2.3 apart. Visual drift is real but small
  relative to normal frame-to-frame change. The pooled encoder is not very sensitive.
- **Error grows with off-demo distance.** At TinyRDT's pre-close descent states, the gap between its action and the expert's action from
  that state is:

  | distance to nearest demo state | median action gap |
  |---|---|
  | < 0.02 rad | 0.038 rad |
  | 0.02–0.05 rad | 0.061 rad |
  | 0.05–0.10 rad | 0.091 rad |

  On the expert's own states the gap is 0.0015 rad (calibration).

## 11. Evidence on the compounding-error / coverage hypothesis

**For:**
1. The ~0.006 rad offline error becomes >0.015 rad of state deviation within 3–5 steps in every rollout.
2. The policy's error relative to the expert grows monotonically with distance from the demonstrated states (§10). It has no corrective behaviour.
3. Failures and successes start with the same deviation. Failures amplify it (0.03 → 0.18 rad); successes stay bounded.
4. More feedback hurts (K=1 5/10 vs K=8 9/10). With corrective knowledge, extra feedback should help. Instead, each replan from an
   off-demo state injects a new error, and long open-loop chunks track the memorised motion better.
5. Closed-loop chunk disagreement is 1.3–2× the on-distribution value.
6. The privileged-pose BC fails too (2/40), so perception is not the primary failure driver. This is weak evidence: see the §5 caveat.

**Against / caveats:**
- The BC controls are much less accurate offline, so "all policies fail similarly" is only qualitatively true.
- Vision drift is modest. Pooled features may limit fine lateral correction. This is not tested here, and is not excluded as a
  contributing factor for lateral (pan) accuracy.

**Decision:** the evidence supports compounding error from missing off-trajectory coverage as the dominant mechanism for TinyRDT.
Neither "privileged succeeds, image fails" (perception) nor "image BC succeeds, TinyRDT fails" (diffusion) occurred.

## 12. Critical finding for corrective data: the data-generating expert
The current `simulation/expert.py` (approach = timed joint interpolation to 0.18 m, 0.04 rad joint margin) **did not generate
CLEAN10**. Running it live diverges from the recorded actions at t=1 and takes 56–57 steps against 46–51.

`simulation/legacy_expert.py` reconstructs the actual data-generating expert: state-feedback DLS IK in every phase, approach target
at cube + 0.10 m, joint margin 0.01. It reproduces **all 10 demos bit-exactly** (same actions, joint states and phase sequence; all succeed).

Consequence: the data-generating expert is a valid state-feedback corrective oracle **in approach too**, which is where TinyRDT's
divergence begins. Corrective labels must come from this legacy expert, or the labels would disagree with CLEAN10.

## 13. Recommended next experiment and perturbation magnitudes
Corrective data from the legacy expert on the same 10 seeds, with an ablation of CLEAN10 vs +PERTURB vs +DAGGER. Measured ranges
to cover:
- Successful rollouts carry about 0.03 rad of state deviation and about 5 mm of path deviation.
- Failures pass 0.03 rad by step 8 and reach 0.07 rad by step 10 and 0.15 rad at close; lateral error at close is 10–37 mm.
- Grasp envelope: 0.03 rad per joint, 7 mm lateral.

Perturbations should therefore be mostly 0.01–0.03 rad, with a boundary set at 0.04 rad. The larger, policy-specific states
(0.05–0.15 rad) are covered by DAgger, which visits them naturally, not by large injected noise.

## 14. Reproduction
```
scripts/run_closed_loop_diagnostics.sh                              # resumable sweep, 2 workers
.venv/bin/python -m evaluation.grasp_sensitivity --output artifacts/closed_loop_v2/grasp_sensitivity.json
.venv/bin/python -m evaluation.closed_loop_analysis                 # -> artifacts/closed_loop_v2/analysis/
```
Legacy-expert verification: the check in CLAUDE_PROGRESS.md, Phase 4.
