<div align="center">

# MiniRDT-SO101

**A small vision-conditioned Diffusion Transformer policy for the SO-101 robot arm, trained and evaluated on a physically validated
MuJoCo pick task, built and audited one stage at a time.**

![python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![pytorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![mujoco](https://img.shields.io/badge/MuJoCo-3.x-1f6feb)
![params](https://img.shields.io/badge/TinyRDT-2.0M_params-6f42c1)
![physics](https://img.shields.io/badge/benchmark-physics--v2-1baf7a)
![status](https://img.shields.io/badge/status-research_preview-orange)

<img src="docs/assets/rollout_success.gif" width="640" alt="TinyRDT performing a side grasp in closed loop"/>

<em>TinyRDT (2.0M trainable parameters) picking the cube with a genuine side pinch, closed loop from a 160×120 camera image and joint state.
Left: task camera. Right: side view at table height. The fingers never enter the table.</em>

</div>

> [!NOTE]
> **Benchmark version change.** A collision audit ([`TABLE_COLLISION_AUDIT.md`](TABLE_COLLISION_AUDIT.md)) found that the original
> simulator (**physics-v1**) had no robot–table collision, and that every "successful" grasp was a sandwich grasp with one jaw inside the
> table. All v1 manipulation results are **invalid** and kept only as history (tag `physics-v1-invalid`). Results below use
> **physics-v2**: table collision, calibrated contacts, a validated side-pinch expert and a stricter success criterion
> ([`PHYSICS_V2_REPORT.md`](PHYSICS_V2_REPORT.md)).

---

## Overview

MiniRDT-SO101 is a scaled-down, RDT-inspired robot-policy pipeline, built stage by stage and validated at each step before scaling:

<p align="center"><img src="docs/assets/pipeline.png" width="92%" alt="MiniRDT pipeline"/></p>

- **Simulation (physics-v2):**
  - SO-101 (official CAD-derived MJCF) in MuJoCo, with robot and fingertip collision against the table.
  - Contacts calibrated so that the table and the grasped cube resist the arm's actuator forces.
  - PickCube with randomised cube positions.
- **Expert:** a state-feedback, top-down **side-pinch** expert: 6-D pose IK, a table-clearance-checked grasp height, and a slow
  re-centring final descent. Validated **100/100** on random cubes with zero table penetration.
- **Policy:** TinyRDT, a 4-layer Transformer (d=192, 6 heads) over visual, proprioceptive and diffusion-time tokens plus 16 noisy
  action tokens. It predicts a 16-step (0.8 s) chunk of absolute joint targets.
- **Diffusion:** cosine noise schedule, x0-prediction, hold-last-action padding, EMA weights, deterministic DDIM (10 steps).
- **Control:** receding horizon: observe, predict 16 actions, execute K of them, re-observe.
- **Success (v2)** requires all of:
  - a valid opposing side pinch (contact normals within 30° of horizontal, no top/underside contact) in each of 5 hold steps;
  - the cube lifted above 10 cm;
  - no invalid robot/cube/pad penetration at any point.

<p align="center"><img src="docs/assets/rollout_strip.png" width="100%" alt="Side-view keyframes of a TinyRDT side grasp"/></p>

## Highlights

| | |
|---|---|
| **Physical validity first** | Found that the v1 simulator let the gripper pass ~100 mm through the table, and that every v1 success depended on it. Rebuilt the benchmark as a versioned physics-v2 with pre-registered tolerances and adversarial regression tests. |
| **Genuine side grasp, learned** | The same 2.0M TinyRDT trained on 10 physics-v2 demos succeeds **49/50** closed loop on the memorised cubes (**10/10 at every K ≥ 2**). Every success is a valid side pinch with **0.00 mm** robot–table penetration. |
| **Diffusion fix** | Found and fixed a terminal-SNR / ε-parameterisation failure: sampled error dropped **15×**; DDIM-5…100 and DDPM agree. |
| **Model bugs found** | A fingertip collision pad sat ~10 mm inside the jaw, and mass-normalised soft contacts let the gripper squeeze ~9 mm into the 10 g cube. Both were measured, fixed and regression-tested. |
| **Honest history** | Every experiment was pre-registered in [`docs/research/research_log.md`](docs/research/research_log.md), including the invalidated v1 results and the corrections. |

## Results (physics-v2)

<table>
<tr>
<td width="50%"><img src="docs/assets/k_sweep_v2.png" alt="physics-v2 closed-loop success vs K"/></td>
<td width="50%"><img src="docs/assets/physics_fix.png" alt="v1 vs v2 penetration"/></td>
</tr>
<tr>
<td colspan="2" align="center"><img src="docs/assets/diffusion_fix.png" width="60%" alt="diffusion fix"/></td>
</tr>
</table>

| | result |
|---|---|
| expert validation | 10/10 on the CLEAN10 cubes; **100/100** on random cubes. Robot–table 0.00 mm, cube–table ≤ 0.25 mm, pad–cube ≤ 0.96 mm |
| dataset v2 | 100/100 successful episodes (5,408 frames), per-step contact diagnostics, validator: 0 errors |
| TinyRDT offline (10 demos) | all-window MAE 0.0050, worst joint 0.010 rad, episode start 0.0051; pre-registered gate **passes** |
| TinyRDT closed loop (10 memorised cubes) | K=1 9/10 · K=2 10/10 · K=4 10/10 · K=8 10/10 · K=16 10/10 (**49/50**); expert replay 10/10 |

### Generalisation to unseen cube positions ([`PHYSICS_V2_GENERALIZATION_REPORT`](PHYSICS_V2_GENERALIZATION_REPORT.md))

<img src="docs/assets/v2_gen_success_map.png" width="100%" alt="held-out success by cube position"/>

Pre-registered spatial split (80 training demos; 56 expert-validated held-out positions in three categories), same 2M model:

| held-out category | nearest training cube | TinyRDT (80 demos, matched budget) | TinyRDT (10 demos) |
|---|---|---|---|
| A interpolation, inside dense coverage | 4.3 mm median | **20/20** | 11/20 |
| B sparse interpolation (30 mm hole cut out of training) | 10.5 mm median | 4/20 | 0/20 |
| C extrapolation, 3–15 mm outside the workspace | 14.1 mm median | 4/16 | 2/16 |

Success falls steeply with distance to the nearest demonstration (logistic slope −2.87 per 10 mm, 95% CI [−5.04, −1.63]): 87% within
5 mm, 0/6 beyond 15 mm. Two results worth knowing before scaling anything:
- **Offline error does not predict closed-loop skill.** The same recipe at 20k and 80k steps is offline-identical (test MAE 0.0104 vs
  0.0101) but solves 7/20 vs 18/20 of its *own training* scenes in closed loop.
- **At a fixed step budget, more demonstrations do not help** (13 → 17 → 13 → 9 of 56 for 10 → 20 → 40 → 80 demos). The optimisation
  budget has to grow with the dataset.

### Exposure-matched data scaling ([`EXPOSURE_MATCHED_DATA_SCALING_REPORT`](EXPOSURE_MATCHED_DATA_SCALING_REPORT.md))

<img src="docs/assets/v2_scaling_curve.png" width="100%" alt="exposure-matched data scaling"/>

Same 2M model, same 56 held-out positions, optimisation **exposure held constant** at 147.87 passes over each subset's own windows
(budgets derived from measured window counts: 10,055 / 20,037 / 40,018 / 80,000 steps):

| demos | held out (K=8) | A | B | C | nearest demo (median) |
|---|---|---|---|---|---|
| 10 | 15/56 | 11/20 | 2/20 | 2/16 | 17.7 mm |
| 20 | 17/56 | 12/20 | 2/20 | 3/16 | 11.4 mm |
| 40 | 26/56 | 10/20 | 14/20 | 2/16 | 9.5 mm |
| 80 | 28/56 | 20/20 | 4/20 | 4/16 | 7.2 mm |

**More data helps only through coverage.** Doubling the dataset looks beneficial on its own (+0.37 logit per doubling), but conditioned on
the distance to the nearest training cube the effect vanishes (−0.05, 95% CI [−0.35, +0.21]) while the distance effect stays (−1.97 per
10 mm). Extrapolation never improves (2 → 3 → 2 → 4). The previous fixed-step curve (13 → 17 → 13 → 9) was an exposure artifact.

### Training-seed replication ([`TRAINING_SEED_REPLICATION_REPORT`](TRAINING_SEED_REPLICATION_REPORT.md))

<img src="docs/assets/v2_seed_variance.png" width="100%" alt="held-out success by data scale, every training seed visible"/>

5 independent training seeds per scale (20 models, 1,120 held-out rollouts), everything else frozen:

| demos | held out per seed | mean | SD | training scenes |
|---|---|---|---|---|
| 10 | 15, 10, 9, 11, 8 | 10.6/56 | 2.7 | 7–10 of 10 |
| 20 | 17, 24, 21, 25, 8 | 19.0/56 | **6.9** | 5–9 of 10 |
| 40 | 26, 23, 25, 28, 27 | 25.8/56 | 1.9 | 6–9 of 10 |
| 80 | 28, 28, 31, 33, 28 | 29.6/56 | 2.3 | **10/10 every seed** |

- **The coverage conclusion replicates.** Distance to the nearest demonstration: −2.51 logit per 10 mm [−3.53, −1.79]; dataset size after
  conditioning on distance: **+0.06 [−0.22, +0.32]**.
- **80 demos solve all 20 dense-interpolation positions in all five runs** (A = 20/20, SD 0).
- **The TRAIN40 B = 14/20 result did not replicate** (14, 3, 3, 7, 5) — it was one lucky run.
- **Position, not seed, decides outcomes:** 90–99% of explained variance is between positions, and independently trained policies fail the
  same position the same way 78.5% of the time.
- Fitted success probability vs nearest demonstration: 75% at 4.5 mm, 50% at 8.8 mm [7.2, 10.3], 10% at 17.4 mm.

### Controlled density sweep ([`CONTROLLED_DENSITY_SWEEP_REPORT`](CONTROLLED_DENSITY_SWEEP_REPORT.md))

<img src="docs/assets/v2_density_response.png" width="100%" alt="controlled density response"/>

Seven **80-demonstration** datasets differing only in local geometry: every evaluation position sits in a hole of radius r with a ring of
demonstrations at exactly r. Demonstration count is identical, so density is manipulated rather than observed. 3 seeds per condition.

| nearest demo | 2.5 mm | 5 mm | 7.5 mm | 10 mm | 15 mm | 20 mm | 7.5 mm **one-sided** |
|---|---|---|---|---|---|---|---|
| success | **36/36** | 34/36 | 28/30 | 18/24 | 7/12 | 3/9 | **17/30** |
| rate | 100% | 94% | 93% | 75% | 58% | 33% | 57% |

- **Density is causal:** success falls monotonically as the manipulated distance grows, with training seed accounting for 0.26% of explained
  variance against 34% for the density condition.
- **Support geometry matters as much as distance.** At an identical 7.5 mm, surrounded support gives 93% and one-sided 57%
  (−2.00 logit, ≈ 7.8 mm equivalent, Fisher p = 0.002). **Nearest-demo distance alone is not coverage.**
- **The earlier observational curve was pessimistic by ~7 mm:** same slope (−2.56 vs −2.55 logit per 10 mm), different offset. Estimated 50%
  point 16.0 mm [11.9, 21.8] surrounded, versus 8.8 mm observationally — and the observational curve matches the *one-sided* result almost
  exactly (predicts 58%, measured 57%).

### Interim policy-capacity result (incomplete; [`progress record`](docs/research/capacity_density_scaling_progress.md))

The pre-registered capacity × density study reuses these exact controlled-density datasets, frozen MobileNet vision encoder, seeds 0/1/2,
matched 80,222-step exposure, DDIM-10 sampler and physics-v2 K=8 evaluator. Its completed **4.3M** policy member (d=256, 5 blocks,
8 heads; 4,304,902 trainable policy parameters) improves over the frozen 2M baseline in every completed primary aggregate, with no invalid
physics rollouts:

| condition | frozen 2M | 4.3M (seeds 0/1/2) | 4.3M aggregate |
|---|---:|---:|---:|
| r10 surrounded | 18/24 (75.0%) | 7, 7, 7 | **21/24 (87.5%)** |
| r15 surrounded | 7/12 (58.3%) | 3, 3, 2 | **8/12 (66.7%)** |
| r20 surrounded | 3/9 (33.3%) | 1, 1, 2 | **4/9 (44.4%)** |
| r7.5 one-sided | 17/30 (56.7%) | 6, 8, 6 | **20/30 (66.7%)** |
| r7.5 surrounded control | 28/30 (93.3%) | 10, 9, 4 | **23/24 (95.8%)** |

This is directionally encouraging but **not a final capacity-scaling conclusion**: the larger-model matrix has been paused before its
closed-loop evaluation, so there is no capacity-specific distance curve, d50 estimate, interaction analysis, or diminishing-returns claim.
The prior v1 ablations remain archived in `docs/reports/` and `docs/assets/physics_v1_invalid/`, clearly marked invalid.

## Installation

```bash
git clone https://github.com/Greninja44/mini-rdt-so101.git && cd mini-rdt-so101
uv venv --python 3.12 && uv pip install --python .venv/bin/python -e '.[dev]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
```

Headless rendering uses `MUJOCO_GL=egl`, selected automatically. On machines without GPU EGL (e.g. WSL) rendering falls back to
software at about 0.75 s/frame, which dominates closed-loop evaluation time.

## Quick start (physics-v2)

```bash
# 1. Validate the side-pinch expert (10 fixed + 100 random cubes; task + side-camera frames)
.venv/bin/python -m evaluation.validate_expert_v2

# 2. Collect and validate the physics-v2 dataset (160x120 RGB, 20 Hz, per-step contact diagnostics)
.venv/bin/python -m data.collect_v2 --start 0 --count 100
.venv/bin/python -m data.validate_v2 artifacts/pickcube_physics_v2_rgb160

# 3. Train TinyRDT on the 10-demo subset (validated recipe: configs/tinyrdt_clean10.json)
.venv/bin/python -m training.audit_overfit --dataset artifacts/pickcube_physics_v2_rgb160 --output artifacts/physics_v2/tinyrdt_clean10 \
    --schedule cosine --prediction x0 --padding hold --steps 20000 --seed 17 --batch-size 8 --learning-rate 0.001 --device cuda \
    --eval-interval 2000 --ema-decay 0.999

# 4. Offline gate (sampler comparison, timestep errors, conditioning ablations)
.venv/bin/python -m evaluation.research_audit --checkpoint artifacts/physics_v2/tinyrdt_clean10/ema_last.pt \
    --dataset artifacts/pickcube_physics_v2_rgb160 --output artifacts/physics_v2/tinyrdt_clean10/diag --device cuda

# 5. Closed-loop evaluation (receding horizon; GIF + plot + v2 validity fields per rollout), then summarise
.venv/bin/python -m evaluation.closed_loop --checkpoint artifacts/physics_v2/tinyrdt_clean10/ema_last.pt \
    --dataset artifacts/pickcube_physics_v2_rgb160 --k 8 --max-steps 150 --output artifacts/physics_v2/closed_loop
.venv/bin/python -m evaluation.v2_closed_loop_summary --rollouts artifacts/physics_v2/closed_loop
```

Steps 3–5 are automated and resumable in `scripts/experiments/run_physics_v2.sh`. `PickCubeConfig(physics="v1")` reproduces the
invalid historical benchmark bit-exactly.

**Also available:**
- `evaluation.recovery`: perturbation-recovery benchmark.
- `evaluation.oracle_ablation`: arm / gripper oracle split.
- `evaluation.table_collision_audit`: numeric robot/pad/cube penetration audit of recorded rollouts.
- `evaluation.squeeze_test`: grasp-contact stiffness calibration.
- `data.collect_corrective`: DAgger and perturbation data with counterfactual expert labels.
- `evaluation.annotate`: annotated rollout GIFs.
- `scripts/make_readme_assets.py`: regenerates every figure in this README.

## Repository layout

```
simulation/   MuJoCo scene, SO-101 env, IK controllers, experts (legacy_expert = dataset generator, bit-exact)
data/         episode format, collection/validation, windowed ML dataset, corrective & varied-start data
models/       TinyRDT (pooled/spatial vision tokens, state dropout, frozen/trainable encoder), BC baseline
training/     diffusion process (DDPM/DDIM, cosine/linear, x0/ε), controlled training runner, inference policies
evaluation/   closed-loop evaluator, recovery & oracle benchmarks, offline gate, analysis and plotting
configs/      task config and the validated TinyRDT recipe
scripts/      artifact manifest, README asset generation, experiments/ (resumable pipelines per phase)
docs/         reports/ (findings), research/ (pre-registered log, handoff audit), diffusion.md, assets/
tests/        diffusion oracle tests, env/dataset tests, bit-exact expert/counterfactual regression
```

Datasets, checkpoints and rollouts are not versioned (`artifacts/`, about 800 MB). `docs/artifact_manifest.json` lists each file's
size and sha256 for backup and verification.

## Documentation

| report | question |
|---|---|
| [`PHYSICS_V2_REPORT`](PHYSICS_V2_REPORT.md) | **physics-v2: can MiniRDT learn a genuinely physical side grasp? (yes, on memorised scenes)** |
| [`PHYSICS_V2_GENERALIZATION_REPORT`](PHYSICS_V2_GENERALIZATION_REPORT.md) | **can the same 2M model pick cubes it never saw? (yes, inside dense data coverage)** |
| [`physics_v2_generalization_spec`](docs/research/physics_v2_generalization_spec.md) | pre-registered generalisation protocol: split, budget, primary K, statistics, failure taxonomy |
| [`EXPOSURE_MATCHED_DATA_SCALING_REPORT`](EXPOSURE_MATCHED_DATA_SCALING_REPORT.md) | **does more data help, or just closer data? (coverage, not count)** |
| [`exposure_matched_data_scaling_spec`](docs/research/exposure_matched_data_scaling_spec.md) | pre-registered exposure definition, derived budgets, distance bins, decision gate |
| [`TRAINING_SEED_REPLICATION_REPORT`](TRAINING_SEED_REPLICATION_REPORT.md) | **how much of the result is data geometry vs one training run? (mostly geometry)** |
| [`training_seed_replication_spec`](docs/research/training_seed_replication_spec.md) | pre-registered seed matrix, variance statistics, decision gate |
| [`CONTROLLED_DENSITY_SWEEP_REPORT`](CONTROLLED_DENSITY_SWEEP_REPORT.md) | **manipulating demonstration density: distance is causal, and support geometry matters as much** |
| [`controlled_density_sweep_spec`](docs/research/controlled_density_sweep_spec.md) | pre-registered density manipulation, conditions, statistics, decision gate |
| [`TABLE_COLLISION_AUDIT`](TABLE_COLLISION_AUDIT.md) | the audit that invalidated physics-v1 |
| [`physics_v2_spec`](docs/research/physics_v2_spec.md) | pre-registered validity spec, tolerances and amendments |
| [`00_summary`](docs/reports/00_summary.md) | *(physics-v1, invalid)* all v1 variants in one table |
| [`01_closed_loop_diagnostics`](docs/reports/01_closed_loop_diagnostics.md) | *(physics-v1, invalid)* why offline accuracy did not transfer to closed loop |
| [`02_corrective_data`](docs/reports/02_corrective_data.md) | *(physics-v1, invalid)* does DAgger / perturbation data make the same 2M model robust? (no) |
| [`03_failure_isolation`](docs/reports/03_failure_isolation.md) | *(physics-v1, invalid)* oracle split, expert-prefix and mechanism |
| [`research_log`](docs/research/research_log.md) | every experiment pre-registered before its result, including corrections |
| [`diffusion.md`](docs/diffusion.md) | equations, schedule and parameterisation choices |

## Roadmap

- [x] SO-101 MuJoCo PickCube, expert, dataset
- [x] TinyRDT with a correct diffusion formulation; 10-demo offline overfit gate passes
- [x] Collision audit; physics-v2 benchmark (table collision, calibrated contacts, validated side-pinch expert, stricter success)
- [x] Reliable closed loop on memorised cubes under physics-v2 (49/50)
- [x] Physics-v2 generalisation: 80 training demos → 56 held-out cube positions (20/20 inside dense coverage; collapses beyond ~5 mm from data)
- [x] Exposure-matched data scaling (10/20/40/80 demos at 147.87 passes each): 15 → 17 → 26 → 28 of 56, explained by local coverage, not count
- [x] Seed replication (5 seeds x 4 data scales): coverage conclusion replicates; 80 demos give A = 20/20 in every run
- [x] Controlled density sweep: density is causal; surrounded support tolerates ~2x the distance of one-sided support
- [~] Capacity × density: interim 4.3M result recorded; larger-model study paused before evaluation and final analysis
- [ ] Rotations, sizes and shapes; multiple objects and tasks; language conditioning
- [ ] External SO-100/101 datasets, sim-to-real on a physical SO-101

## Acknowledgements

- Robot model: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (Apache-2.0; licence in `simulation/assets/`).
- Inspired by [RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation](https://github.com/thu-ml/RoboticsDiffusionTransformer).
- Built on [MuJoCo](https://mujoco.org), [PyTorch](https://pytorch.org) and torchvision's MobileNetV3.
- Conventions follow [LeRobot](https://github.com/huggingface/lerobot).

```bibtex
@article{liu2024rdt,
  title   = {RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation},
  author  = {Liu, Songming and Wu, Lingxuan and Li, Bangguo and Tan, Hengkai and Chen, Huayu and Wang, Zhengyi and Xu, Ke and Su, Hang and Zhu, Jun},
  journal = {arXiv preprint arXiv:2410.07864},
  year    = {2024}
}
```

<details>
<summary><b>Simulator details (Phase 1)</b>: SO-101 conventions, API, expert, success criterion, dataset format</summary>

### SO-101 source and conventions
The model is vendored from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101), commit
`eecbe3e0a9ebb23e25ad7b2759b03884c6660903`. It is the CAD-derived **new calibration** MJCF with the original STL assets. Joint order:

| index | joint | MJCF radian range |
|---:|---|---:|
| 0 | shoulder_pan | [-1.919862, 1.919862] |
| 1 | shoulder_lift | [-1.745329, 1.745329] |
| 2 | elbow_flex | [-1.690000, 1.690000] |
| 3 | wrist_flex | [-1.658063, 1.658063] |
| 4 | wrist_roll | [-2.743847, 2.841206] |
| 5 | gripper hinge | [-0.174533, 1.745329] |

Names and ordering match LeRobot. The gripper is a normalised opening in [0, 1] (0 closed), converted to the MJCF hinge range.
Coordinates are MuJoCo world frame in metres, +Z up, with the table top at Z=0.

### API
```python
from simulation.env import SO101PickCubeEnv
env = SO101PickCubeEnv()
obs, info = env.reset(seed=123)          # obs: rgb uint8[120,160,3], joint_pos float32[5], gripper float32[1]
obs, reward, terminated, truncated, info = env.step(action)   # action: float32[6] absolute joint targets + opening
```

### Expert and success
The expert is a state machine: `HOME → MOVE_ABOVE_OBJECT → DESCEND → CLOSE_GRIPPER → LIFT → HOLD → SUCCESS`. It uses bounded
damped-least-squares position IK.

Success requires all of:
- a two-pad physical pinch;
- the cube lifted above 0.10 m;
- both held for 5 control steps.

Two convex fingertip pads, placed from the CAD fingertip transforms, provide contact. They never weld or teleport the cube.

The 100-episode dataset was generated by an earlier expert version: state-feedback IK in every phase, approach target 0.10 m, joint
margin 0.01. It is reconstructed bit-exactly in `simulation/legacy_expert.py`.

### Dataset format
Each `episodes/episode_XXXXXX/` holds equal-length `.npy` arrays (`rgb`, `joint_pos`, `gripper`, `action`, `cube_pose`,
`end_effector_pose`, `timestamp`, `expert_state`, `success`) and an `episode.json` with task, seed, rates, camera, action definition
and joint limits. `observation_t → action_t` alignment is explicit. `python -m data.validate DATASET` checks integrity.
</details>
