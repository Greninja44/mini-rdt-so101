# CLAUDE_PROGRESS — MiniRDT-SO101 TinyRDT overfit gate

Running log for the next agent. Newest entries at the bottom of each section.
See `CLAUDE_HANDOFF_AUDIT.md` for the repository state at takeover.

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
- Balanced sweep: `scripts/run_closed_loop_diagnostics.sh`, covering 3 policies × the same 10 seeds × K ∈ {1,2,4,8}, max 150 steps,
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

### E5 — closed-loop sweep COMPLETE (2026-09-18) → `CLOSED_LOOP_DIAGNOSTIC_REPORT.md`
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
- A CLEAN10: the existing 10 demos, unchanged (hashes in ARTIFACT_MANIFEST.json). Model: the existing `cosine_x0_hold_ema/ema_last.pt`.
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

### Phase 5 RESULT (2026-09-18 20:30 UTC) → `CORRECTIVE_DATA_REPORT.md`
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
