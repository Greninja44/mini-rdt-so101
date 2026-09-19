# Phase 6: isolating the remaining closed-loop failure (2M TinyRDT, 10 memorised scenes)

> ⚠️ **PHYSICS-V1 / INVALID COLLISION BENCHMARK.** A later audit (`TABLE_COLLISION_AUDIT.md`, 2026-09-19) found that the robot had
> **no collision with the table**, and that every successful grasp was a sandwich grasp with one pad ~10 mm inside the table. All
> **manipulation success rates, grasp-tolerance measurements and grasp-geometry conclusions in this document are invalid** as
> physical results. Offline diffusion and sampler findings remain valid. It is kept unchanged as research history; frozen state is in
> `docs/research/physics_v1_freeze.json` (git tag `physics-v1-invalid`). The physics-v2 revalidation is `PHYSICS_V2_REPORT.md`.


> **Correction (2026-09-19):** statements below that call the model "state-dominant" rest on the wrong-state ablation. Actions are absolute
> joint targets (≈ current state + a small step), so a wrong state inflates the error regardless of how the scene is identified. That ablation
> does not measure scene identification. The observational findings are unaffected: the expert prefix, the oracle split, the shared-start
> trajectory overlap, and the neighbour-directed lateral errors. See the CORRECTION in docs/research/research_log.md.


Pre-registered in `docs/research/research_log.md` (Phase 6) before any result. Evaluator, seeds, K values and analysis are identical to Phase 5
(`evaluation/phase5_analysis.py`, output `artifacts/phase5_analysis/phase5.json`).

## Result in one paragraph
The failure is **arm placement at the start of the episode, not the gripper and not visual precision**. Evidence:
- The policy's gripper, paired with the expert's arm, succeeds 40/40.
- The policy's arm, paired with the expert's close rule, reaches the 4 mm close condition only 19/40 times.
- Spatial visual tokens leave success unchanged (29/40 vs 28/40).
- If the expert drives only the first 9 steps (0.45 s), the unchanged policy succeeds on **all 10 seeds**, including the two it almost
  always fails (ep0, ep7).

Every episode begins from the identical home state, so early actions can only come from the image. The model is state-dominant, so its
first motion blends neighbouring memorised scenes. The state it then reaches selects a neighbour's trajectory, which explains why failing
closes point toward the nearest other cube (CORRECTIVE_DATA_REPORT §16).

## 6a Oracle split (A = gate-passing TinyRDT EMA; 10 seeds × K ∈ {1,2,4,8})

| condition | K=1 | K=2 | K=4 | K=8 | total | vs A |
|---|---|---|---|---|---|---|
| A (policy arm + policy gripper) | 5 | 6 | 8 | 9 | 28/40 | — |
| **expert arm + policy gripper** | 10 | 10 | 10 | 10 | **40/40** | p=0.0002 |
| policy arm + expert gripper (close only within 4 mm) | 4 | 2 | 6 | 7 | 19/40 | p=0.069 |

- With the policy arm and the expert close rule, 19 of 21 failures are "descent": the grasp center never came within 4 mm of the target,
  so the expert rule never closed. When it did close, the lateral error was 1.8 mm median and all 19 attempts succeeded.
- With the expert arm, the policy's own close command lands within 0.4 mm median (p90 2.1 mm), with 0 early closes.
- The policy's gripper decision is therefore adequate; its arm is not.

## 6b Spatial visual tokens (S)
- The 576×4×5 frozen MobileNet map is used as 20 tokens instead of one pooled token. Trainable parameters: 2,013,318 (+3,648 position
  embedding). Trained on CLEAN10 with A's exact config.
- Offline: MAE 0.0068, worst joint 0.0156, episode start 0.0053 (A: 0.0067 / 0.016 / 0.006). Passes the gate.
- It uses the image more: wrong-image MAE 0.047 vs A's 0.024; zeroed features 0.156 vs 0.061.
- Closed loop: 29/40 (K=1 7, K=2 5, K=4 7, K=8 10), p=1.0 vs A. Recovery 61/80 vs 63/80, p=0.85. Seed 7 is 1/4.
- **Spatial precision of the visual features alone does not fix the failure.**

## 6c Expert-prefix control
The legacy expert drives steps 0–8 (no perturbation), then the policy runs at K=4:

| | prefix run | same model, normal K=4 |
|---|---|---|
| A | **10/10** | 8/10 (ep0 and ep7 fail) |
| S | **10/10** | 7/10 |

## Mechanism and consistency with earlier findings
- The initial robot state is identical across the 10 demos (std about 1e-7). The first actions must come from vision. The model's
  conditioning is state-dominant: wrong-state MAE 0.21 vs wrong-image 0.02–0.05.
- Divergence beyond 0.03 rad starts in approach (median step 8). Failing lateral errors point toward the nearest other cube.
  - ep7, whose neighbours are 16–18 mm away, fails in almost every variant.
  - ep4 and ep8, 5.5 mm apart and inside the ≈7 mm envelope, blend harmlessly.
- Corrective data (Phase 5) cannot resolve an early scene-identification ambiguity. Its labels teach how to return to *a* trajectory,
  while the ambiguity is *which* trajectory. That explains the null Phase-5 result.
- K dependence (K=1 worst): re-deciding every step from the blended early state compounds the ambiguity. Long chunks execute the first
  image-based plan open-loop.

## Next (Phase 7, pre-registered, running)
- **7a:** prefix-length sweep N ∈ {1,2,3,5} expert steps. How quickly does a correct start disambiguate the scene?
- **7b:** state dropout during training (p=0.3; the state token is replaced by a learned null, +192 params; labels unchanged). This
  forces scene identification from the image. It is evaluated normally, with recovery and the offline gate.

Rule unchanged: the 80-demo held-out phase starts only after memorised scenes are reliable. No scaling.

## Reproduction
```
scripts/experiments/run_phase6.sh     # 6a oracle split, 6b spatial (train separately, see below), closed-loop + recovery
scripts/experiments/run_phase6c.sh    # expert-prefix control
.venv/bin/python -m training.audit_overfit --output artifacts/phase6/spatial_clean10 --vision-tokens spatial --schedule cosine \
  --prediction x0 --padding hold --steps 20000 --seed 17 --batch-size 8 --learning-rate 0.001 --device cuda --eval-interval 2000 --ema-decay 0.999
.venv/bin/python -m evaluation.phase5_analysis
```
