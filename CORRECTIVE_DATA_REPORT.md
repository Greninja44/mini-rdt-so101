# Corrective-data report: does corrective state coverage make the same ~2M TinyRDT robust closed-loop?

**Answer: No.** Expert-corrective labels from states the policy actually visits (DAgger), and from controlled perturbations, did not improve
closed-loop success or recovery on the 10 memorised scenes. The model stayed at 2,009,670 trainable parameters throughout.

The best corrective model (C40k: DAgger data with the same clean-sample exposure as the baseline) matched the baseline at 28/40.
It trended worse on recovery (51/80 vs 63/80, p=0.054). DAgger did teach one thing specifically: early gripper closes fell from 7/40 to 1/40.

The failures that remain are lateral misalignments at the close. They point toward the **nearest other memorised cube**, which is evidence of
scene blending: the model does not read the exact cube position from the image. Corrective labels cannot fix that. It is being tested directly
in Phase 6 (spatial visual tokens; see §22).

All numbers come from `artifacts/phase5_analysis/phase5.json` (produced by `evaluation/phase5_analysis.py`). The design was pre-registered in
`CLAUDE_PROGRESS.md` before any collection. Fisher exact tests are two-sided on identical seeds and conditions.

## 1. Previous closed-loop diagnostic (summary of `CLOSED_LOOP_DIAGNOSTIC_REPORT.md`)
- Expert replay: 10/10.
- Clean TinyRDT (A): 28/40 over 10 seeds × K ∈ {1,2,4,8} (K=1 5, K=2 6, K=4 8, K=8 9).
- BC image+state: 0/40. BC privileged: 2/40. Neither is accuracy-matched.
- Every failure was a mis-aligned close. Divergence past 0.03 rad began in approach (median step 8).
- Measured grasp envelope: about ±0.03 rad per joint, about 7 mm lateral, no more than about +5 mm too high or −10 mm too low. Early close is dangerous.

## 2. Evidence for / against compounding error (going in)
For: policy error grew with off-demo distance (0.038 → 0.061 → 0.091 rad against the expert label), failures amplified their deviation while
successes stayed bounded, and more replanning hurt. That motivated this phase. Against: the BC controls were weak. See §18 for what this
phase showed.

## 3. Corrective-data collection design
- **Oracle: the reconstructed data-generating expert** (`simulation/legacy_expert.py`). The current `expert.py` did NOT generate CLEAN10.
  The legacy expert reproduces all 10 demos bit-exactly and is state-feedback IK in every phase, **including approach**, so approach states
  have valid corrective labels.
- **Label contract** (`data/corrective.py`): observation of the ACTUAL state → the 16-step chunk the legacy expert executes FROM that exact
  state. It is a counterfactual computed from a full `mjData` snapshot, then restored; hold-padded, with a valid mask. Executed futures are never
  sliced. Along the expert's own trajectory the labels equal the recorded actions bit-exactly (regression test `tests/test_corrective.py`).
  Exactness requires copying the full `mjData`: the expert's IK reads kinematics that `mj_step` leaves one substep stale.
- **B (+PERTURB)** (`data/collect_corrective.py --mode perturb`): per seed, 12 episodes = t0 ∈ {approach step 10, DESCEND−2, DESCEND,
  LIFT+1} × joint ∈ {pan, lift, elbow}. An offset is added to the expert command for 3 steps, then the expert continues from the actual state.
- **C (+DAGGER)** (`--mode dagger`): TinyRDT A drives (10 seeds × K ∈ {1,2,4,8}). The legacy expert shadows it and labels every visited state.
  Takeover happens when time-free joint distance to the seed's expert trajectory exceeds 0.10 rad, or when the policy proposes to close while
  the expert would keep it open.
  - The 0.10 threshold was derived from the measured traces. Successes peak at 0.04–0.14 rad before the close (27/28 below 0.10); failures peak
    at 0.10–0.25 rad.
  - The suggested 0.02–0.03 rad would have fired at about step 4 in every episode, so DAgger would never have seen the states where the
    policy actually fails.
- **Bm**: B subsampled to C's frame count, which separates coverage type from volume.

## 4. Physical perturbation ranges used (B)
- Magnitudes: 0.01 (33 episodes), 0.02 (34), 0.03 (37), 0.04 (16) rad. One joint at a time, sign ±, for 3 steps, clipped 0.01 inside joint limits.
- Realised deviation: median peak time-free joint distance 0.050 rad and median peak grasp-center path deviation 9.1 mm. This matches the
  measured drift of successful rollouts (≈0.03–0.07 rad, 5–15 mm).
- The expert recovered from all 120 perturbations (0 discarded).

## 5–7. Dataset statistics

| | episodes | frames / action windows | corrective states | unique cube seeds | discarded |
|---|---|---|---|---|---|
| A CLEAN10 | 10 | 483 | 0 | 10 | — |
| B +PERTURB | 10 + 120 | 483 + 2972 | 2972 (360 during the perturbation, 2612 expert recovery) | 10 | 0 |
| C +DAGGER | 10 + 40 | 483 + 2100 | 2100 (**1064 policy-visited**, 1036 expert after takeover) | 10 | 0 |
| Bm | 10 + 87 | 483 + 2080 | 2080 | 10 | — |

DAgger takeovers: 25/40 because the policy tried to close while the expert would keep the gripper open, 13/40 on joint distance, and 2/40 none
(the policy succeeded). C has 4.3× A's frames and B has 6.2×. Bm controls for B's volume.

The same seeds as training are used for collection by design (a memorised-scene experiment). The collector refuses non-CLEAN10 seeds, so
held-out seeds cannot leak in.

## 8. Training configuration
Identical to A: TinyRDT 192/4/6, H=16, frozen pooled MobileNetV3-S, cosine schedule, x0 prediction, hold padding, EMA 0.999, AdamW lr 1e-3,
batch 8, seed 17, the CLEAN10 normalisation stats kept fixed. Each batch is 50% CLEAN10 windows and 50% corrective samples. 20k steps
(equal gradient budget).

Follow-up **C40k** (not pre-registered, added after seeing the offline regression and before any C40k result): C with 40k steps, giving the
same number of clean-window updates as A.

## 9. Offline results (on the CLEAN10 training windows, the pre-registered gate)

| | all-window MAE | worst joint (rad) | episode start | gate |
|---|---|---|---|---|
| A | 0.0067 | 0.016 | 0.006 | pass |
| B | 0.0109 | 0.023 | 0.019 | pass |
| C | 0.0139 | 0.027 | 0.015 | pass |
| Bm | 0.0097 | 0.021 | 0.020 | pass |
| C40k | 0.0106 | 0.022 | 0.0069 | pass |

Mixing corrective data costs precision on the demonstrated path. C40k largely recovers it.

## 10. Normal closed-loop success (10 memorised seeds × K ∈ {1,2,4,8}, max 150 steps)

| | K=1 | K=2 | K=4 | K=8 | total | vs A (Fisher) |
|---|---|---|---|---|---|---|
| A | 5 | 6 | 8 | 9 | **28/40** | — |
| B | 1 | 5 | 5 | 7 | 18/40 | **p=0.041 (worse)** |
| C | 7 | 3 | 8 | 8 | 26/40 | p=0.81 |
| Bm | 2 | 5 | 8 | 8 | 23/40 | p=0.35 |
| C40k | 7 | 5 | 7 | 9 | **28/40** | p=1.0 |

## 11. Recovery benchmark (the expert to step 6, a 3-step offset δ on pan or lift, then the policy at K=4; 80 rollouts each)
Realised handover deviation (A, median):

| δ (rad) | time-free joint distance | grasp-center path deviation |
|---|---|---|
| 0.01 | 0.016 rad | 4.6 mm |
| 0.02 | 0.034 rad | 10.1 mm |
| 0.03 | 0.052 rad | 15.0 mm |
| 0.04 | 0.068 rad | 20.0 mm |

Recovery successes per magnitude (out of 20 each):

| | 0.01 | 0.02 | 0.03 | 0.04 | total | vs A |
|---|---|---|---|---|---|---|
| A | 18 | 17 | 15 | 13 | **63/80** | — |
| B | 16 | 15 | 14 | 13 | 58/80 | p=0.46 |
| C | 13 | 14 | 12 | 12 | 51/80 | p=0.054 |
| Bm | 16 | 16 | 16 | 13 | 61/80 | p=0.85 |
| C40k | 14 | 13 | 13 | 11 | 51/80 | p=0.054 |

Pan (lateral) is harder than lift for every variant (A: 29/40 vs 34/40). Plot: `artifacts/phase5_analysis/phase5_success.png`.

## 12. Grasp success / failure breakdown (normal rollouts)

| | pinched | lifted | dropped | failure phases |
|---|---|---|---|---|
| A | 29 | 29 | 1 | grasp 11, drop 1 |
| B | 23 | 20 | 2 | grasp 17, lift 3, drop 2 |
| C | 27 | 26 | 0 | grasp 13, lift 1 |
| Bm | 28 | 23 | 0 | grasp 12, lift 5 |
| C40k | 30 | 29 | 1 | grasp 10, lift 1, drop 1 |

## 13–15. Geometry at the policy's first close command

| | lateral median / p90 (mm) | ≤7 mm | vertical median (mm) | max joint deviation median (rad) | early closes |
|---|---|---|---|---|---|
| A | 6.4 / 24.8 | 21/40 | −1.4 | 0.038 | 7/40 |
| B | 13.1 / 32.1 | 13/40 | +1.8 | 0.076 | 10/40 |
| C | 5.4 / 22.3 | 24/40 | −2.5 | 0.037 | 9/40 |
| Bm | 8.5 / 31.5 | 18/40 | −2.5 | 0.066 | 20/40 |
| C40k | 5.7 / 18.3 | 23/40 | −1.9 | 0.033 | **1/40** |

"Early" means a close command before the expert's close step. DAgger (C40k) nearly eliminates early closes, consistent with its takeover
labels ("keep open"), but lateral misalignment remains.

## 16. Representative failure analysis
- **Per-seed success is stable across all variants**, so the difficulty is scene-specific. Successes out of 20 (A, B, C, Bm, C40k × 4 K):
  ep0 6, ep2 17, ep3 16, ep4 14, ep5 14, ep6 19, **ep7 2**, ep8 12, ep9 12, ep11 11.
- **Scene blending.** For failing closes with more than 7 mm lateral error (n=70, all variants), the error points toward the nearest *other*
  CLEAN10 cube: median cosine 0.92, and 59% have cosine > 0.5 against 33% expected for a random direction. Rollouts share seeds, so they are
  not independent.
  - ep7's cube (255, 63) mm sits between ep5 (15.8 mm away) and ep4/ep8 (~18 mm), and fails almost always.
  - ep4 and ep8 are only 5.5 mm apart, inside the grasp envelope, and blend harmlessly.
- The policy tracks the expert's path shape but offset toward a neighbouring scene's trajectory. GIFs in `artifacts/phase5_analysis/gifs/`:
  `A_ep007_k1.gif`, `C40k_ep007_k8.gif` (failure), `C40k_ep006_k8.gif` (success), `dagger_ep007_k1.gif` and `dagger_ep000_k1.gif`
  (collection with takeover). Each shows the camera, the top-down path against the expert, deviation, gripper and events.

## 17. CLEAN vs PERTURB vs DAGGER
- Neither kind of corrective data improved normal success.
- Perturbation data at full volume hurt significantly (B). At matched volume (Bm) it was neutral.
- DAgger was neutral on normal rollouts, whether or not precision was restored (C, C40k). It trended worse on recovery.
- Of B and C at matched volume (Bm vs C): normal 23 vs 26, recovery 61 vs 51. Neither is clearly better.

## 18. Additional data vs better coverage
Neither improved robustness, so there is no improvement to attribute. Two confounds were controlled:
- volume, via Bm;
- clean-sample exposure, via C40k.

Conclusion: **compounding error is real** (divergence grows, and error grows off-demo), **but missing corrective coverage is not the
limiting factor.** The policy does not act on corrective labels because its lateral error comes from not resolving *which* cube position it
is in (scene blending). From the same observation, the corrective label for "cube at ep7" and the behaviour learned for its neighbours conflict.

## 19. Remaining limitations
- n=40 normal and 80 recovery rollouts per variant. Differences of about ±15% are not resolvable.
- One DAgger iteration only.
- BC controls are not accuracy-matched.
- The scene-blending evidence is correlational: 70 closes sharing 10 seeds.
- The C40k follow-up was not pre-registered.
- Rendering is software llvmpipe, about 0.75 s/frame.

## 20. Reproduction
```
scripts/run_closed_loop_diagnostics.sh                       # A/BC baseline sweep (resumable)
scripts/run_corrective_phase.sh                              # collect B, C; train B, C, Bm; evaluate (resumable)
scripts/run_phase5_remaining.sh                              # C40k evaluation + full recovery benchmark
# C40k training:
.venv/bin/python -m training.audit_overfit --output artifacts/corrective_train/C40k --schedule cosine --prediction x0 --padding hold \
  --steps 40000 --seed 17 --batch-size 8 --learning-rate 0.001 --device cuda --eval-interval 4000 --ema-decay 0.999 \
  --corrective artifacts/corrective/dagger --corrective-fraction 0.5
.venv/bin/python -m evaluation.phase5_analysis                 # -> artifacts/phase5_analysis/
.venv/bin/python -m evaluation.annotate <rollout.npz|episode_dir> --episode N --output out.gif
```

## 21. Critical artifacts to back up (gitignored; hashes in `ARTIFACT_MANIFEST.json`)
- `artifacts/pickcube_smoke100_rgb160/` (dataset)
- `artifacts/research_audit/cosine_x0_hold_ema/` (A)
- `artifacts/corrective/{perturb,dagger}/` (datasets B and C, with per-episode metadata)
- `artifacts/corrective_train/{B,C,Bm,C40k}/` (checkpoints and configs)
- `artifacts/closed_loop_v2`, `closed_loop_v3`, `recovery/` (results)
- `artifacts/phase5_analysis/`

## 22. Recommended next phase
**Do not scale. Do not add more clean or corrective data yet.** Test the scene-blending / visual-precision explanation. This is Phase 6,
pre-registered and running:
- **6a oracle split:** policy arm + expert gripper rule, and expert arm + policy gripper. Is the failure the arm or the close decision?
- **6b spatial visual tokens:** the 4×5 MobileNet map as 20 tokens (2,013,318 trainable, +3,648), CLEAN10 only. Does lateral precision improve?
- **6c expert-prefix control:** the expert drives steps 0–8, then the policy. Is the early approach the origin?

Results will be appended to `CLAUDE_PROGRESS.md` and a Phase 6 report. Only if memorised scenes become reliable should the 80-demo held-out
phase start.
