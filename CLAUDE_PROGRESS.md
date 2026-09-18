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
