# CLAUDE_PROGRESS — MiniRDT-SO101 TinyRDT overfit gate

> ⚠️ **ALL PHYSICS-V1 MANIPULATION RESULTS ARE INVALID** (see TABLE_COLLISION_AUDIT.md). Every closed-loop success rate,
> grasp-tolerance and grasp-geometry conclusion recorded below, up to the audit entry, used an environment without robot-table
> collision, where every success was a sandwich grasp through the table. The entries are left unchanged as history. The
> physics-v2 section at the end supersedes them.


Running log for the next agent. Newest entries at the bottom of each section.
See `docs/research/handoff_audit.md` for the repository state at takeover.

## Context at takeover (2026-09-18)

- Handoff said the gate was blocked on linear-β / ε-prediction sampling error.
- The repo already had a full diagnostic audit (`evaluation/research_audit.py`,
  `training/audit_overfit.py`, `artifacts/research_audit/*`) and the fix
  (cosine schedule + x0 prediction + hold padding), but nothing written up. The
  last result was the fixed-budget 5000-step comparison. Its
  `gate_definition.json` says "final dense gate evaluated separately", and
  that final evaluation had never been run.

## Hypotheses → evidence (summary; details in the audit doc)

| # | Hypothesis | Evidence | Verdict |
|---|---|---|---|
| H1 | Reverse-process math is wrong | Exact-ε oracle reconstruction, point-mass oracle through production DDIM (1–100 steps) and DDPM, and posterior vs independent Gaussian conditioning all pass (`tests/test_diffusion_oracles.py`) | Rejected |
| H2 | Terminal-SNR mismatch (linear, T=100) | ᾱ₉₉ = 0.3636; √ᾱ = 0.603 of the signal survives at t=99, but sampling starts at N(0,I). An analytic Gaussian test shows even a perfect score biases the mean by ᾱ_T·μ | **Confirmed** as a real train/inference mismatch |
| H3 | Cosine fixes it with no other change | cosine+ε: sampled MSE 5164 (diverges). At ᾱ₉₉ = 2.4e-7, x0 = (x−√(1−ᾱ)ε̂)/√ᾱ amplifies ε error ×4.1e6 | Rejected: ε-param is ill-conditioned at zero SNR |
| H4 | Cosine + x0 prediction | Sampled MAE 0.027 vs 0.102 for linear+ε; DDIM 5…100 and DDPM agree | **Confirmed** as the main fix |
| H5 | End-of-episode padding targets hurt | cosine_x0: last-quarter MAE 0.097 vs 0.025 with hold padding | **Confirmed**. Hold padding is the second fix |
| H6 | Pooled vision features cause the episode-start floor | BC with the **same** pooled features reaches episode-start MAE 0.0206, below the diffusion model. Privileged-cube BC: 0.0266. State-only BC: 0.042 (pan 0.15) | Rejected for memorisation. Pooling does cost localisation precision (probe: 4.3 mm pooled vs 2.8 mm spatial), which matters for generalisation, not the 10-demo gate |
| H7 | Remaining 20k-step gate misses come from optimisation noise, not capacity | Constant LR 1e-3: sampled MAE swings 0.0117–0.0159 between evals. Gate misses are 0.0001–0.0008 | Testing with EMA (below) |

## Experiment log

### E1 — 20k-step final run of the winning config (2026-09-18)
`--schedule cosine --prediction x0 --padding hold --steps 20000 --seed 17 --batch-size 8 --lr 1e-3`
→ `artifacts/research_audit/cosine_x0_hold_final/`.
- Reproduces the 5000-step run exactly at steps 2000 and 4000 (same RNG streams).
- Gate (`evaluation/gate_check.py`, thresholds unchanged):
  - `last.pt` FAIL: worst joint 0.0301 (limit 0.03, seed 4101); worst quarter 0.0251/0.0258.
  - `best.pt` FAIL: episode start 0.0257 on all seeds (limit 0.025). Everything else passes.
  - `best.pt` is selected by train-set sampled MSE with seed 4100, so it is mildly optimistic.

### E2 — same run + weight EMA (decay 0.999). PRE-REGISTERED before running
- The only change is an EMA copy of the weights. The raw trajectory must match E1 bit-for-bit, and I will check that.
- **Primary gate result = final EMA weights (`ema_last.pt`, step 19999), no checkpoint selection.**
- Thresholds unchanged. If it fails, I stop and report. No further knobs.
- **Result: PASS on all 3 seeds.** All-window MAE 0.0067, worst joint 0.0158 rad (shoulder_lift), gripper 0.003,
  worst quarter 0.017 (last), episode start 0.006. The raw trajectory is bit-identical to E1 at every eval, which confirms H7.
- DDIM 5/10/20/50/100 and DDPM-100 are identical (MAE 0.0045 on reference windows). Starting from q(x_T|x0) instead of N(0,I)
  gives the same result, so there is no remaining train/inference mismatch.
- Conditioning at t=99: wrong image 0.024, zero image 0.061, wrong state 0.218. Shuffling the image only at the episode start
  raises sampling MAE from 0.006 to 0.039. The model uses the image, but relies on state far more.
- Timestep errors: raw x0 MSE is 6e-5 at t=0 and 1.8e-4 at t=99, with no high-noise collapse. ε-MSE is 0.67 at t=0, which is
  expected for x0-param because ε is ill-posed as t→0, and it is not used there.
- `artifacts/research_audit/cosine_x0_hold_ema/{ema_last.pt,diagnostics_ema_last/,gate_result_ema_last.json}`.

### E3 — closed-loop MuJoCo, memorised scenes, K ∈ {1,2,4,8} — INCOMPLETE
`evaluation/closed_loop.py`, `ema_last.pt`, DDIM-10, max 150 steps, same seeds as the demos. The workers were killed twice
when the session ended; there are 15 of 50 policy rollouts (episodes 0, 2, 6, 7), in `artifacts/closed_loop/memorized_ema_{a,b}.log`.
- Controls: replaying the recorded expert actions succeeds 4/4. Reset RGB equals the recorded frame 0 exactly.
  Policy-on-recorded-obs error equals the offline error, so the inference path is faithful.
- Policy: K=1 2/4, K=2 1/3, K=4 1/2 (plus an ep0 smoke FAIL), K=8 2/2. Episode 6 succeeds at every K; episode 0 succeeds
  only at K=8. The sample is too small to rank K.
- Failure mechanism (ep0 K=4, fully traced): error compounds. Joint deviation is 0.005–0.02 rad early, about 0.05 by t=8,
  and 0.2 at the start of descent (t≈26). The grasp pose is 0.11/0.16 rad off (shoulder_lift/elbow), the cube drops at t≈43,
  and the policy parks at a fixed pose it never saw in the demos. Most other failures never grasp (cube z ≈ 0.015).
- H8 (not yet tested): covariate shift. The demos are deterministic and noise-free (initial state std about 1e-7), so there
  is no coverage off-trajectory, and the model relies strongly on state.
- **Memorised closed-loop execution does NOT reliably work → do not proceed to 80-episode training.**

## Next steps
1. Finish E3 (2 workers max on this 7.8 GB host, about 1 s/step because of rendering):
   `.venv/bin/python -m evaluation.closed_loop --checkpoint artifacts/research_audit/cosine_x0_hold_ema/ema_last.pt --episodes 3 4 5 8 9 11 --k 1 2 4 8 --max-steps 150 --output artifacts/closed_loop/memorized_ema_c`
2. Test H8: collect DART-style data on the same 10 seeds (the expert under injected action noise, labelled by the expert's
   corrective action), retrain the identical 2M config, and repeat E3. Control: closed-loop `bc_rgb`/`bc_privileged` on the
   same scenes, to confirm the failure is not diffusion-specific.

## Phase 3 — why offline accuracy ≠ closed-loop success (started 2026-09-18)

Constraints from the user: no retraining, no 80-episode training, no scaling, no architecture change, no new randomisation.

### Tooling
- `evaluation/closed_loop.py`: `--policy tinyrdt|bc`. Every rollout is saved as JSON+NPZ on completion (resumable). Each step
  records the grasp flag, pad contacts and grasp-center position. The control loop is unchanged.
- `training/policy.py::BCPolicy`: image+state and privileged (state + true cube pose) BC, using the same clamp as TinyRDT.
  Faithfulness on recorded observations (arm joints) matches the offline error.
- Balanced sweep: `scripts/experiments/run_closed_loop_diagnostics.sh`, covering 3 policies × the same 10 seeds × K ∈ {1,2,4,8}, max 150 steps,
  2 workers, plus expert replay on all 10 seeds. Output: `artifacts/closed_loop_v2/`.
- Rendering is llvmpipe (software) with a 4096 shadow map, about 0.75 s/frame. It can't be changed without shifting the image distribution.
- Determinism check: ep6 K=1/K=2 reproduce the earlier sweep exactly (51 and 49 steps).

### Analysis definitions (fixed BEFORE seeing sweep results; see docstring of `evaluation/closed_loop_analysis.py`)
Policy phase, failure phase, and onset thresholds of 0.015 rad (reconstruction noise) and 0.03 rad (grasp tolerance, below).
The time-free onset is when the grasp center is more than 5 mm from the expert's path.

### E4 — grasp sensitivity (physics only, `evaluation/grasp_sensitivity.py`)
Recorded expert actions are replayed on all 10 seeds. From the first DESCEND step onward, one joint gets a constant offset or
the gripper-close time is shifted. Result: `artifacts/closed_loop_v2/grasp_sensitivity.json`.
- Unperturbed: 10/10. Grasp-center offset at close: xy 0.9 mm, z +0.6 mm.
- ±0.03 rad on pan, elbow and wrist_flex: 10/10 (up to about 7 mm xy). ±0.05 rad: 0–4/10 (pan +0.05 still pinches 10/10
  but lifts only 1/10, an off-centre grasp).
- shoulder_lift is asymmetric. −0.02 rad: 9/10. −0.03 rad (+6.6 mm too high): 2/10. +0.05 rad (−10 mm): 9/10. +0.08 rad: 0/10.
- Close timing: 1–5 steps late 10/10; **1 step early 4/10**, 2–5 steps early 3/10 (closing while still descending).
- **Tolerance: about ±0.03 rad per joint, about 7 mm xy, grasp center no more than about +5 mm above or 10 mm below its
  target. Never close early.**

### E5 — closed-loop sweep COMPLETE (2026-09-18) → `docs/reports/01_closed_loop_diagnostics.md`
- Expert replay 10/10. TinyRDT 28/40 (K=1 5, K=2 6, K=4 8, K=8 9 of 10; K1 vs K8 Fisher p=0.141). BC image 0/40.
  BC privileged 2/40. The BCs are not accuracy-matched (offline worst joint about 0.03 rad).
- All 12 TinyRDT failures are mis-aligned closes (lateral error at close 9.5–37 mm, envelope about 7 mm). The 0.03 rad tolerance is crossed in
  **approach** (median step 8). The gap between the policy's action and the expert label grows with off-demo distance (0.038 → 0.061 → 0.091 rad).
- **Decision gate: coverage / compounding supported → proceed.**

### E6 — the data-generating expert (CRITICAL)
- The current `simulation/expert.py` did NOT generate CLEAN10: run live, it differs at t=1.
- `simulation/legacy_expert.py` (DLS IK in every phase, approach target cube+0.10 m, joint margin 0.01) reproduces all 10 demos
  **bit-exactly**.
- Its IK reads MuJoCo kinematics left stale by mj_step. Exact snapshots therefore need a full `mj_copyData` with no mj_forward on restore.
  `data/corrective.py::expert_chunk` counterfactual chunks equal the recorded futures bit-exactly on all 10 seeds.
- Consequence: approach HAS a valid state-feedback corrective oracle (option A, verified exact), and approach is where divergence begins.

## Phase 5 — corrective state coverage (PRE-REGISTERED 2026-09-18, before any collection)

**Hypothesis:** TinyRDT fails closed-loop because it never saw expert corrections from off-trajectory states. Adding expert labels at such states
(B: expert under controlled perturbations; C: states TinyRDT itself visits) will raise memorised-scene success and recovery with the
same 2,009,670-parameter model. If B ≈ C ≈ A, the hypothesis is weakened. If C > B, the policy's own state distribution matters beyond
generic perturbation.

**Label contract:** every corrective sample is (RGB + state of the ACTUAL state s_t) → the 16-step chunk the legacy expert executes FROM s_t,
computed as a counterfactual from an exact snapshot. There is no slicing of executed futures during perturbed or policy-driven steps.

**Datasets** (CLEAN10 normalisation stats kept fixed for every variant):
- A CLEAN10: the existing 10 demos, unchanged (hashes in docs/artifact_manifest.json). Model: the existing `cosine_x0_hold_ema/ema_last.pt`.
- B +PERTURB: per seed, 12 episodes = t0 ∈ {approach step 10, DESCEND−2, DESCEND, LIFT+1} × joint ∈ {shoulder_pan, shoulder_lift,
  elbow_flex}. Magnitude is drawn from {0.01, 0.02, 0.03} (p=0.8 total) or 0.04 (p=0.2), sign ±, fixed RNG per seed. The offset is added to the
  expert command for 3 steps, clipped to joint range ±0.01, then the expert resumes from the actual state. Frames are recorded from t0.
  Discarded if the expert does not succeed.
- C +DAGGER: TinyRDT A drives (policy seed 0) on 10 seeds × K ∈ {1,2,4,8}. The legacy expert shadows it (FSM observes the real state) and
  labels every visited state. Takeover when time-free joint distance to the seed's nominal trajectory > 0.10 rad, or the policy proposes
  close while the expert would keep it open. After takeover the expert drives to the end. Approach is included (valid oracle, E6). One
  iteration only. Discarded if the expert fails after takeover.
- B_m: B subsampled (fixed-seed episode order) to C's corrective frame count, to separate coverage type from data volume.

**Training:** identical config to A (cosine, x0, hold, EMA 0.999, 20k steps, batch 8, lr 1e-3, seed 17, same frozen encoder). Each batch is
50% CLEAN10 windows and 50% corrective windows. Every variant gets the same 20k gradient steps, so the optimisation budget is equal.
No per-variant tuning.

**Evaluation:**
- (i) Normal closed-loop: 10 memorised seeds × K ∈ {1,2,4,8}, same evaluator and analysis as A.
- (ii) Recovery benchmark, all variants: legacy expert to step 6, then offset δ ∈ {0.01, 0.02, 0.03, 0.04} for 3 steps on shoulder_pan or
  shoulder_lift (sign + for even seed index, − for odd), then the policy takes over at K=4 (single K, identical for all variants). 80 rollouts per variant.
  The start point (step 6) is not in B's t0 set.
- Primary metric: success. Also grasp, lift, drop, onset, joint/EEF error at close, early close, replans.

**Interpretation criteria** (no manufactured threshold):
- Compare success counts on identical seeds and conditions, with Fisher exact tests.
- "Substantial" means a clear gain in both normal (all K) and recovery success, plus closes inside the measured grasp envelope.
- An offline CLEAN10 regression beyond the gate for B or C is reported as a cost.

### Phase 5 interim (2026-09-18 15:10 UTC)
- Collection: B has 120 kept / 0 discarded (2972 frames). C has 40 kept / 0 discarded (2100 frames, including policy-visited frames). Bm has 87 B episodes (2080 frames).
- Offline on CLEAN10 (all pass the gate, but are worse than A's 0.0067 / 0.016 / 0.006):
  B 0.0109 / 0.023 / 0.019, C 0.0139 / 0.027 / 0.015, Bm 0.0097 / 0.021 / 0.020 (all-window / worst joint / episode start).
- Normal closed-loop, first look: C 26/40 (K1 7, K2 3, K4 8, K8 8) vs A 28/40. B 18/36 so far.
- **Confound identified:** 50/50 batches with equal steps give each corrective model half of A's clean-window updates. On-demo precision
  dropped toward the 0.03 rad grasp tolerance (C worst joint 0.027).
- **Added follow-up C40k (NOT pre-registered; decided after seeing the offline regression, before any C40k result):**
  identical to C but 40k steps, so it gets the same number of clean-window updates as A. Evaluated with the same protocol.
  Interpretation rule: if C40k restores offline precision AND improves closed-loop over A, the combination helps. If precision is
  restored but closed-loop does not improve, corrective coverage is not the missing ingredient.

## Phase 6 — isolate the remaining closed-loop failure (PRE-REGISTERED 2026-09-18 16:50 UTC, before any Phase-6 result)
Context: corrective data (B/C/Bm) did not improve normal closed-loop success (A 28/40, B 18/40 p=0.041 worse, C 26/40, Bm 23/40).
Every variant fails the same way: lateral misalignment at the close (p90 22–32 mm vs a ≈7 mm envelope). In 25/40 DAgger episodes the
policy tried to close while the expert would keep the gripper open. Per the user's plan: no scaling. Investigate the remaining cause.

**6a Oracle split (no training; A = ema_last.pt; 10 memorised seeds × K ∈ {1,2,4,8}, 150 steps):**
- `gripper_oracle`: policy arm + the data-generating expert's gripper rule (close only once the grasp center is within 4 mm of its
  target, from the shadow FSM on the actual state).
- `arm_oracle`: expert arm (DLS IK from the actual state) + the policy's gripper command.
- Interpretation:
  - gripper_oracle ≫ A → the close decision is the main failure.
  - gripper_oracle fails by never closing → the arm never gets within 4 mm, so arm precision is the problem.
  - arm_oracle ≈ expert → the policy's gripper timing is acceptable given an accurate arm.

**6b Spatial vision tokens (single-factor change, CLEAN10 only, otherwise identical to A):**
- The frozen MobileNet 576×4×5 map is flattened to 20 visual tokens (the same 576→192 projection per token, a learned position embedding)
  instead of one global-average-pooled token. Trainable parameters go from 2,009,670 to 2,013,318 (+3,648 position embedding).
- The same 20k steps / EMA / cosine x0 hold config and seed. Then the offline gate, normal closed-loop and the recovery benchmark.
- Hypothesis: lateral error at close is limited by pooled features' spatial precision (probe: 4.3 mm pooled vs 2.8 mm spatial).
- Success = more successes than A on identical seeds/K, closes within the 7 mm envelope, and a better recovery curve.

**6c Expert-prefix control (pre-registered 17:55 UTC, prompted by A's recovery curve; before any 6c result):** the legacy expert drives steps
0–8 unperturbed (δ=0), then the policy runs at K=4 from step 9 (same code path as the recovery benchmark), on all 10 seeds, for A and S.
If A's success is ≫ its normal K=4 result (8/10) and ≫ its 28/40 overall, the early approach from the home pose is the dominant failure origin.

### Phase 5 RESULT (2026-09-18 20:30 UTC) → `docs/reports/02_corrective_data.md`
- Normal: A 28/40, B 18/40 (p=0.041 worse), C 26/40, Bm 23/40, C40k 28/40. Recovery (80 each): A 63, B 58, C 51, Bm 61, C40k 51.
- **Corrective coverage does not make the same 2M TinyRDT robust.** DAgger (C40k) cut early closes from 7/40 to 1/40, but lateral misalignment remains.
- The new mechanism is **scene blending**. Failing lateral close errors point toward the nearest other CLEAN10 cube (median cos 0.92; 59% vs 33% chance).
  ep7 (neighbours 16–18 mm away) succeeds 2/20 across all variants. ep4/ep8 (5.5 mm apart, inside the envelope) are fine.
- Bug fixed: recovery filenames collapsed magnitudes (with_suffix ate ".NN"). Orchestrator bug: pgrep on script names matched launcher shells.
  The fix is to wait on log markers and files.

### Phase 6 RESULT (2026-09-18 22:30 UTC)
- 6a oracle split (A, 10 seeds × 4 K):
  - **expert arm + policy gripper 40/40** (vs A 28/40, p=0.0002): the policy's gripper timing is fine given an accurate arm.
  - policy arm + expert 4 mm gripper rule: 19/40. 19 failures never close because the arm never gets within 4 mm.
  - **The failure is arm (lateral) placement, not the gripper decision.**
- 6b spatial tokens S (2,013,318 trainable, CLEAN10):
  - Offline equal to A (0.0068 / 0.0156 / 0.0053). Uses the image twice as much (wrong-image MAE 0.047 vs 0.024).
  - Closed-loop 29/40 (p=1.0 vs A); recovery 61/80 vs 63/80. **Spatial visual precision alone is not the fix.**
- 6c expert prefix (the expert drives steps 0–8, then the policy at K=4): **A 10/10, S 10/10**, including ep0/ep7 (A's normal K=4 is 8/10).
- **Mechanism (supported):** every episode starts from the identical HOME state (std ~1e-7), so early actions can come only from the image.
  The model is state-dominant (wrong-state MAE 0.21 vs wrong-image 0.02–0.05). Early motion blends neighbouring scenes. The state then
  identifies a scene, and the policy locks onto a trajectory offset toward the nearest other cube (the §16 direction statistic).
  Given a correct 9-step start, the policy completes every seed. Corrective data cannot fix a scene-identification error at the start.

## Phase 7 — early scene identification (PRE-REGISTERED 2026-09-18 22:40 UTC, before any Phase-7 result)
- **7a prefix-length sweep:** the expert drives the first N ∈ {1, 2, 3, 5} steps unperturbed, then A at K=4, 10 seeds each.
  This finds how many correct steps disambiguate the scene.
- **7b state dropout (training-only regulariser):**
  - During training, with p=0.3 per sample, the state token is replaced by a learned null embedding (+192 params → 2,009,862
    trainable). At inference the state is always given.
  - Labels are unchanged (true expert chunks): no perturbed-observation → nominal-action pairs.
  - Otherwise identical to A (CLEAN10, cosine / x0 / hold, EMA, 20k steps, seed 17).
  - Prediction if the mechanism is right: more successes than A, concentrated on ep0/ep7 and small K; higher image dependence in the
    conditioning ablation.
  - Evaluated with the offline gate, normal (40) and recovery (80).
- The same stopping rule applies. 80-demo work starts only if memorised scenes become reliable. No scaling.

### 7a RESULT (2026-09-18 ~23:00 UTC): expert-prefix length N, then A at K=4, 10 seeds
N=0 (normal): 8/10 (ep0, ep7 fail). N=1: 9/10 (ep7). N=2: 7/10 (ep0, ep7, ep8). N=3: 9/10 (ep7). N=5: 10/10. N=9: 10/10.
Not monotone at n=10 (N=2 dip). ep7 (the most crowded scene) needs ≥5 correct early steps (0.25 s). This supports early scene
identification as the failure origin, and the effect is concentrated in the first ~5 steps.

### 7b RESULT (2026-09-19 00:02 UTC): state dropout p=0.3
- Offline passes (0.0088 / 0.019 / 0.012). Image reliance ×5 (wrong-image MAE 0.13 vs A 0.024). Wrong-state MAE is still 0.17.
- Closed-loop 27/40 (p=1.0 vs A); recovery 60/80; early closes 15/40. **Falsified as a fix.** ep0/ep7 still fail.
- Mechanism refined:
  - For the first 6 steps the 10 demo trajectories are within 0.000–0.008 rad of each other (shared HOME start). TinyRDT's own tracking error
    there is 0.01–0.05 rad, 5–10× larger.
  - The closed-loop state is nearest to a *neighbour's* demo at t=4–6 (ep7 → ep3/ep5), and the policy follows the neighbour (ep5/ep3/ep4 by t=15–20).
  - The trajectories only separate by ≥0.04 rad at t≈8. The state input actively misleads during the shared start. Dropout that keeps state
    at inference cannot remove this.

## 7c — image-only TinyRDT (PRE-REGISTERED 2026-09-19 00:12 UTC, before any 7c result)
- `--state-dropout 1.0`: the state token is ALWAYS the learned null, in training and at inference. The arm pose is available only through
  the image. Parameters 2,009,862 (state_proj unused). Otherwise identical to A.
- Prediction if state-based lock-on is the mechanism: ep0/ep7 improve and the per-seed pattern changes. Arm precision may degrade
  (frozen pooled features).
- Evaluated with the offline gate, normal (40), recovery (80) and an expert prefix (N=9).

### 7c RESULT (2026-09-19 01:55 UTC): image-only TinyRDT
- Offline passes (0.0107 / 0.024 / 0.010). Closed-loop 18/40 (p=0.041 worse than A); recovery 39/80 (p=0.0001 worse); prefix 6/10.
- Per-seed pattern changed as predicted: ep0 3/4 (A 1/4), ep7 1/4 (A 0/4). The other seeds got worse, because without proprioception the frozen pooled
  image cannot place the arm precisely.
- Conclusion: the shared-start lock-on is real. No model-side change tried at 2M on 10 sparse demos removes it: corrective data,
  spatial tokens, state dropout and image-only all fail. A correct 5–9 step start does remove it.

## Phase 8 — density diagnostic (PRE-REGISTERED 2026-09-19 02:00 UTC; a DIAGNOSTIC, not the generalisation phase)
Hypothesis:
- CLEAN10 cubes are sparse (nearest neighbours 5.5–32 mm; most 10–19 mm, beyond the ≈7 mm grasp envelope). Blending toward a
  neighbour's trajectory is therefore a grasp miss.
- With the 80 training demos, neighbours are much closer, and the same blending becomes interpolation toward the correct position.

Change: the same TinyRDT (2,009,670) and A's config (cosine / x0 / hold / EMA), trained on all 80 training-split episodes (normalisation
from those 80), 40k steps (the 8× larger dataset needs more than A's 20k; documented). Split seed 17; validation and test untouched in training.

Evaluation:
- (i) The 10 memorised CLEAN10 seeds × K ∈ {1,2,4,8}: directly comparable to A's 28/40.
- (ii) The 10 validation seeds × K ∈ {1,2,4,8}: held-out, a first generalisation look, with expert replay as the control. The test split stays untouched.

Prediction if density is the issue: (i) ≫ 28/40, especially ep0/ep7. If (i) ≈ 28/40, sparsity is not the explanation.

### Phase 8 RESULT (2026-09-19 03:28 UTC): density diagnostic (2M TinyRDT trained on 80 training episodes, 40k steps)
- Offline on its training windows: 0.0083 / 0.0174 / 0.0088. Still state-dominant (wrong-state 0.23 vs wrong-image 0.029).
- Memorised CLEAN10 seeds: **22/40** (K=1 4, K=2 5, K=4 4, K=8 9) vs A 28/40. ep7 2/4 (A 0), ep0 2/4 (A 1); other seeds got worse.
- Held-out validation seeds [10,29,41,49,68,72,79,89,90,96]: **20/40** (K=1 3, K=2 5, K=4 5, K=8 7). Expert replay 10/10. The test split is untouched.
- Density helps the crowded seeds but does not fix closed-loop reliability. Held-out ≈ memorised, so the limit is closed-loop control, not memorisation.

## 8b — K=16 (full-chunk open-loop execution) (PRE-REGISTERED 03:30 UTC, before any K=16 result)
K=8 is the best K in every model: pooled over the 7 CLEAN10-trained variants, K=8 59/70 vs K=1 35/70 (same seeds, not independent).
K=16 is untested. Evaluated uniformly (not per-model selection): A, S, C40k and D80 on the 10 memorised seeds, plus D80 on the 10 validation seeds
(50 rollouts). If K=16 ≥ K=8 across models, open-loop chunk execution (fewer replans from a misleading state) is the practical lever in
this regime.

### 8b RESULT (2026-09-19 ~04:10 UTC): K=16
A 9/10 (ep7 fails), S 10/10, C40k 9/10 (ep7), D80 memorised 8/10, D80 validation 8/10 (49, 89 fail). This is a plateau with K=8.
Memorised K≥8 pooled over A/S/C40k/D80: 73/80. Held-out D80 at K≥8: 15/20.
→ See `docs/reports/00_summary.md`. STOPPED for the user's decision (80-demo held-out benchmark at fixed K=8, or a varied-start-pose experiment).

## Phase 9 — varied start poses (user chose "start" on option B; PRE-REGISTERED 2026-09-19 05:30 UTC, before any result)
Hypothesis: in CLEAN10 the identical HOME start makes the trajectory a deterministic function of the scene, so the robot state acts as a
scene identifier (causal confusion). Near the shared start that identifier is wrong, and small errors put the arm on a neighbour's path.
Decorrelating start state from scene should force image-based scene identification.

Data VS10 (`data/collect_varied_start.py`):
- CLEAN10 (the 10 originals, symlinked, unchanged) plus 4 demos per CLEAN10 seed from random start poses (home ± U(0.15) rad on
  pan / lift / elbow / wrist_flex; start seed 9000).
- Each is a complete legacy-expert rollout from its own start, starting in MOVE_ABOVE (true expert labels, no corrupted pairs).
- A physics check gave 38/40 expert successes; failures are logged and discarded.

Training: the same TinyRDT (2,009,670) and A's config; all VS10 episodes (normalisation from them); 20k steps (A's budget).

Evaluation:
- (i) Standard HOME start, 10 seeds × K ∈ {1,2,4,8}: compared with A's 28/40. Prediction: > A, with ep0/ep7 improved and K=1 improved.
- (ii) Unseen random starts (start seed 777) at K ∈ {4,8} for VS and A, with the legacy expert from the same start as the control.
  Prediction: VS ≫ A.

### Phase 9 RESULT (2026-09-19 07:16 UTC): INCONCLUSIVE (underfit confound)
- VS10: 48 episodes (38 new starts; 2 expert failures discarded), 2321 windows. 20k steps.
- Offline 0.0151 / worst joint 0.0288 / start 0.0167. That is 2× worse than A and at the 0.03 rad grasp tolerance.
- HOME start 14/40 (K 4/3/5/2) vs A 28/40. Unseen random starts (seed 777, K 4/8): VS 0/20, A 2/20; expert control 10/10.
- The equal-step budget underfits 5× more data, so the start-pose hypothesis is NOT yet tested.

### CORRECTION to earlier interpretation (applies to CLOSED_LOOP_DIAGNOSTIC_REPORT §10, PHASE6_REPORT, OVERNIGHT_SUMMARY)
- The "state-dominant" label rested on wrong-state ablation MAE (~0.2) vs wrong-image (~0.02–0.05). Actions are ABSOLUTE joint targets
  (≈ current state + a small delta), so a wrong state necessarily produces a large error. That ablation does not measure scene identification.
- What remains valid: the wrong-image effect is small, so image changes move predictions only modestly. The expert-prefix result, the
  oracle split, the shared-start trajectory overlap, and the neighbour-directed lateral errors are unaffected.
- The corrected mechanism statement: early scene identification from the image is imprecise. During the shared start the trajectories
  overlap, so the policy's early motion is only weakly scene-specific and drifts toward neighbours' paths. The "state identifies scene"
  framing is not supported by the ablation.

## 9b — VS trained to comparable precision (PRE-REGISTERED 07:20 UTC, before any 9b result)
- The same VS10 data and config, with 60k steps (3×; D80 needed 40k for 8× data).
- Proceed only if offline worst joint ≤ 0.02 rad. If not, report it and stop.
- Evaluation identical to Phase 9: HOME start 10 seeds × K ∈ {1,2,4,8}, and unseen starts (seed 777) at K ∈ {4,8}.

### 9b RESULT (2026-09-19 08:36 UTC): varied start poses at comparable precision
- VS60k offline: 0.0081 / worst joint 0.0177 / start 0.0115 (criterion ≤ 0.02 met).
- HOME start **24/40** (K 5/7/4/8) vs A 28/40. Unseen random starts (seed 777): **3/20** vs A 2/20; expert control 10/10.
- **Varied start poses (4 per seed) do not fix closed-loop reliability.** It also does not generalise to new starts on memorised cubes.
- Across every variant except the forced image-only one, swapping the image changes predictions little (wrong-image MAE 0.02–0.05 vs
  correct ~0.007). The start-decorrelation data did not raise image reliance (VS60k 0.020).
- Current best-supported limitation: **weak visual conditioning.** With frozen, pooled ImageNet features at 160×120, the policy's
  motion is only weakly scene-specific, which fits the neighbour-directed misses. Changing the vision encoder (fine-tuning or a trainable
  CNN) is an architecture/parameter-count decision, so it is left for the user.

## Phase 10 — stronger vision pathway (user approved "start"; PRE-REGISTERED 2026-09-19 08:55 UTC, before any result)
Hypothesis: weak visual conditioning (frozen, pooled ImageNet features) limits scene-specific motion. Swapping the image changes
predictions by only ~0.01–0.04 rad in every frozen variant.

Changes, otherwise A's exact config (CLEAN10, cosine / x0 / hold, EMA 0.999, 20k steps, batch 8, seed 17):
- **V1:** MobileNetV3-S weights fine-tuned (lr 1e-4; head lr 1e-3; BatchNorm running stats frozen). Pooled token.
  Trainable 2,936,678 (+927,008, the encoder; the total parameter count is unchanged).
- **V2:** the same, with spatial tokens (20). Trainable 2,940,326.

The frozen path is verified unchanged (step-2000 MSE 0.003564 reproduced).

Evaluation per model:
- Offline gate plus conditioning ablation (the key check: does wrong-image MAE rise substantially?).
- Normal HOME, 10 seeds × K ∈ {1,2,4,8} (A 28/40).
- Unseen random starts (seed 777), K ∈ {4,8} (A 2/20).
- Recovery benchmark (A 63/80).

Prediction if the hypothesis holds: higher image reliance, more successes than A (ep0/ep7), better random-start and recovery results.
If image reliance rises but success does not, visual conditioning is not the bottleneck either.

## TABLE-COLLISION AUDIT (2026-09-19) — physics validity issue, see TABLE_COLLISION_AUDIT.md
- The robot has NO collision with the table (every robot geom has contype/conaffinity 0; the pads collide with the cube only).
  Penetration of up to ~100 mm during approach: expert and policy alike.
- **Every success is a top-bottom sandwich grasp with one pad ~10 mm inside the table.** This covers 10/10 CLEAN10 demos, 28/28 TinyRDT
  successes, and 10/10 current-expert grasps. The cube is pushed 3–6 mm into the table, and pad-cube normals are 100% vertical.
- **All previous success rates require re-validation** as physical PickCube results. No physics, expert, criterion, dataset or policy
  was changed. Recommended fix order: robot-table collision → valid side-pinch expert → stricter success criterion → regenerate the
  data → re-run the benchmarks.
- Phase 10 (fine-tuned vision), still running, is subject to the same caveat.

## PHYSICS-V2 (2026-09-19) → PHYSICS_V2_REPORT.md
- **v1 frozen:** tag `physics-v1-invalid`, `docs/research/physics_v1_freeze.json`. Phase 10 was cancelled (manipulation metric invalid).
- **Spec pre-registered** (`docs/research/physics_v2_spec.md`) with tolerances measured before any expert evaluation. Amendment 1 (pad
  placement, grasp-contact stiffness, a pad-cube check) and amendment 2 (table stiffness) only added checks or stiffened contacts.
- **Bugs found:**
  - the moving pad sat ~10 mm inside the jaw;
  - the mass-normalised soft contacts allowed a ~9 mm squeeze into the cube;
  - the v1 reset silently disabled the v2 pads;
  - the descent overshoot landed a pad on the cube's top (seed 4079).
- **Expert v2 (side pinch):** 10/10 and 100/100, zero robot-table penetration. Dataset v2: 100/100, validator 0 errors.
- **Same 2M TinyRDT on CLEAN10 v2:**
  - offline gate PASS (0.0050 / 0.010 / 0.0051);
  - closed loop 49/50 (K=1 9, K=2/4/8/16 10 each); expert replay 10/10;
  - robot-table 0.00 mm, 0 early closes.
  - The one failure: ep0 K=1 hovered with a pad on the cube's top and never closed.
  - Two successes (ep7 K1/K2) tipped the cube before a valid side pinch.
- **Answer:** MiniRDT learns a genuinely physical side grasp on memorised scenes. **STOPPED** here, per the phase instructions.
  Next: v2 generalisation (80 demos → held-out cubes), then scaling.

## PHYSICS-V2 SPATIAL GENERALIZATION (2026-09-20) → PHYSICS_V2_GENERALIZATION_REPORT.md
- **Pre-registered** (`docs/research/physics_v2_generalization_spec.md`, commit c4b9559) before any held-out demo existed: split, budget rule,
  primary K=8, statistics, failure taxonomy, evaluation tiers.
- **Split:** deterministic and spatial. TRAIN80 (nested TRAIN40/20/CLEAN10) + VAL10 + TEST A/B/C, where B is a 30 mm hole band cut out of
  training and C is 3–15 mm outside the workspace. Expert control: A 20/20, B 20/20, C 16/20 → **56 benchmark positions** (4 C positions are
  an arm reach limit, marked outside the benchmark, never replaced). Expert replay 56/56; resets bit-exact; rollouts deterministic.
- **Answer:** the same 2M TinyRDT generalizes **inside dense data coverage and nowhere else**. With matched exposure (80k steps):
  A **20/20**, B 4/20, C 4/16 (28/56 at K=8; 28/30/27 across sampler seeds; 27 positions always succeed, 26 always fail).
  Success vs distance to the nearest demo: 87% within 5 mm, 0/6 beyond 15 mm (logistic slope −2.87 per 10 mm).
- **Two methodological findings:**
  - **Offline error is blind to closed-loop skill.** TRAIN80 at 20k vs 80k steps: offline test MAE 0.0104 vs 0.0101, but 7/20 vs 18/20 on
    its own *training* scenes. The pre-registered primary (20k) was therefore under-trained: 9/56, and highly sampler-seed sensitive (9/17/18).
  - **At a fixed step budget more data does not help** (13/17/13/9 of 56 for 10/20/40/80 demos); budget must scale with the dataset.
    So "80 demos beat 10" holds only at matched exposure (28 vs 13, McNemar p = 2.7e-4).
- **Failure mode:** one dominant category, `wrong_lateral_alignment` — median 20.0 mm lateral miss at close vs 6.5 mm for successes
  (expert 0.9–2.0 mm). A frozen-feature probe localizes the cube to 4.3/7.9/11.6 mm median (A/B/C), so the learned action mapping, not the
  encoder alone, fails to interpolate across the gap.
- 1,329 rollouts, zero invalid-physics successes; 1 rollout hit the cube–table tolerance and was correctly rejected.
- **STOPPED** per the spec. Next: exposure-matched data scaling, then a density sweep; capacity scaling only after those.

## PHYSICS-V2 EXPOSURE-MATCHED DATA SCALING (2026-09-21) → EXPOSURE_MATCHED_DATA_SCALING_REPORT.md
- **Pre-registered** (`docs/research/exposure_matched_data_scaling_spec.md`, commit 5836929) before training: exposure equation, derived
  budgets, nested subsets, training-scene diagnostic, distance bins, statistics, decision gate.
- **Exposure** = steps x batch / windows, anchored on the existing 80-demo/80k model → 147.87 passes. Budgets derived from *measured* window
  counts (544/1084/2165/4328) → 10,055 / 20,037 / 40,018 / 80,000 steps. Only the anchor qualified for reuse; the old 20k models were retrained.
- **Result (K=8, seed 0, the same frozen 56 positions):** 15 → 17 → 26 → 28 of 56. Only the 8x endpoint is significant (McNemar 14 vs 1,
  p = 0.001); adjacent scales are not.
- **CASE B:** conditioned on distance to the nearest training cube, the effect of doubling the dataset is −0.05 logit [−0.35, +0.21], while
  distance keeps −1.97 per 10 mm. At matched distance, larger datasets show no systematic advantage. **CASE E:** extrapolation flat (2/3/2/4).
  **Partial CASE D:** matched exposure did NOT equalise closed-loop trainability — training-scene success was 9/10, 7/10, 6/10, 10/10.
- **Offline stays disconnected:** TRAIN40_e and TRAIN80_e have identical held-out MAE (0.0101) but A 10/20 vs 20/20 and B 14/20 vs 4/20.
- **Biggest limitation:** one training seed per scale, so the non-monotone category results cannot be attributed. Sampler seeds are NOT the
  cause (positions flipping across 3 seeds: 7/16/11/3).
- 452 rollouts, zero invalid-physics successes. **STOPPED** per the spec: no density sweep, no capacity scaling.
