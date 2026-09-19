# MiniRDT-SO101

A small RDT-inspired, vision-conditioned **Diffusion Transformer (TinyRDT, ~2M trainable parameters)** for the SO-101 arm. It is
trained and evaluated closed-loop on a MuJoCo PickCube task. The project is staged deliberately: understand and fix closed-loop
behaviour at small scale before scaling toward a ~40M MiniRDT.

```
RGB (frozen or fine-tuned MobileNetV3-Small) + joint state → TinyRDT (cosine DDPM, x0-prediction, H=16 action chunk)
  → DDIM-10 sampling → receding-horizon execution (execute K of 16 actions, replan) → SO-101 in MuJoCo
```

## Status (2026-09-19)

| stage | result |
|---|---|
| Simulator, expert, 100-demo dataset | done (Phase 1, below) |
| Offline overfit gate on 10 demos | **passes**: all-window MAE 0.0067, worst joint 0.016 rad (fixed: terminal-SNR/ε-param, cosine+x0, hold padding, EMA) |
| Closed-loop on the 10 memorised cubes | **not yet reliable**: 28/40 over K ∈ {1,2,4,8}; ~90% at K ≥ 8 |
| Corrective data (DAgger / perturbation) | does not improve robustness (`CORRECTIVE_DATA_REPORT.md`) |
| Failure mechanism | arm placement in the first ~5–9 steps. The 10 demo paths overlap from the shared home pose, and misses point toward a neighbouring cube. The gripper is fine: expert arm + policy gripper 40/40 (`PHASE6_REPORT.md`) |
| Spatial tokens, state dropout, image-only, 80-demo density, varied start poses | none fixes it (`OVERNIGHT_SUMMARY.md`) |
| Phase 10: fine-tuned vision encoder | running. V1 offline 2× more precise (worst joint 0.0085 rad) |

The scaling phase (2M → 40M) has **not** started: closed-loop must be reliable first.

## Reports (read in this order)
1. `CLAUDE_HANDOFF_AUDIT.md`: repository state at takeover.
2. `CLOSED_LOOP_DIAGNOSTIC_REPORT.md`: why offline accuracy did not transfer closed-loop, with grasp tolerance measurements.
3. `CORRECTIVE_DATA_REPORT.md`: CLEAN vs PERTURB vs DAGGER ablation.
4. `PHASE6_REPORT.md`: oracle split and the expert-prefix finding.
5. `OVERNIGHT_SUMMARY.md`: all variants in one table, and the decisions pending.
6. `CLAUDE_PROGRESS.md`: the full pre-registered experiment log, including corrections.

Large artifacts (dataset, checkpoints, rollouts) are gitignored. `ARTIFACT_MANIFEST.json` lists every file with its sha256.

## Code map
- `simulation/`: MuJoCo scene, env, controllers, current expert. `legacy_expert.py` is the bit-exact expert that generated the dataset.
- `data/`: dataset format and collection. `corrective.py` / `collect_corrective.py` produce counterfactual expert-label data;
  `varied_start.py` / `collect_varied_start.py` produce varied-start data.
- `models/tiny_rdt.py`: TinyRDT, with options for pooled or spatial vision tokens, state dropout, and frozen or trainable vision.
- `training/`: `diffusion.py` (DDPM/DDIM, cosine schedule, x0/ε), `audit_overfit.py` (the controlled training runner), `policy.py`
  (inference policies).
- `evaluation/`: `closed_loop.py` (receding-horizon MuJoCo evaluator), `recovery.py` (perturbation recovery benchmark),
  `oracle_ablation.py`, `grasp_sensitivity.py`, `closed_loop_analysis.py`, `phase5_analysis.py`, `annotate.py` (annotated GIFs),
  `research_audit.py` / `gate_check.py` (offline gate).
- `scripts/run_*.sh`: resumable experiment pipelines, one per phase.

## Reproduce the core result
```bash
# train the gate-passing TinyRDT on the 10-demo subset
.venv/bin/python -m training.audit_overfit --output artifacts/research_audit/cosine_x0_hold_ema --schedule cosine --prediction x0 \
  --padding hold --steps 20000 --seed 17 --batch-size 8 --learning-rate 0.001 --device cuda --eval-interval 2000 --ema-decay 0.999
# offline gate
.venv/bin/python -m evaluation.research_audit --checkpoint artifacts/research_audit/cosine_x0_hold_ema/ema_last.pt --output /tmp/diag --device cuda
.venv/bin/python -m evaluation.gate_check --diagnostics /tmp/diag/diagnostics.json --gate artifacts/research_audit/cosine_x0_hold_ema/gate_definition.json
# closed-loop on the memorised cubes (software rendering is ~1 s/step)
.venv/bin/python -m evaluation.closed_loop --checkpoint artifacts/research_audit/cosine_x0_hold_ema/ema_last.pt --k 1 2 4 8 --max-steps 150 --output artifacts/closed_loop_demo
```

---

# Phase 1: simulator and dataset

## Quick start

```bash
cd /home/batman/mini-rdt-so101
uv venv --python 3.12
uv pip install --python .venv/bin/python -e '.[dev]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest
.venv/bin/python -m simulation.demo --seed 123
.venv/bin/python -m evaluation.expert_benchmark --episodes 100 --output artifacts/expert_benchmark.json
.venv/bin/python -m data.collect --output dataset_phase1 --episodes 100 --seed 3000 --width 160 --height 120 --workers 4
.venv/bin/python -m data.validate dataset_phase1
```

`simulation.demo --seed 123 --gui` opens MuJoCo's passive viewer when a display is available and prints state transitions. `env.render()` always returns the fixed external RGB camera view. Under WSL/headless use `MUJOCO_GL=egl` (the simulation package selects it unless already set). To replay a benchmark/data seed, use that episode's recorded seed, for example `.venv/bin/python -m simulation.demo --seed 3023 --gui`.

## Architecture

`simulation/scene.py` assembles the table, cube, camera, fixed lighting and robot. `simulation/env.py` owns reset/step, state, camera and success logic. `simulation/controllers.py` contains the action conversion and DLS Cartesian-position IK. `simulation/expert.py` is the deterministic state machine. `data/` provides a transparent on-disk transition format and validator. `evaluation/` performs controlled regression benchmarks.

## SO-101 source and conventions

The model is vendored from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101), commit `eecbe3e0a9ebb23e25ad7b2759b03884c6660903` (Apache-2.0 license copied to `simulation/assets/SO_ARM100_LICENSE`). It is the CAD-derived **new calibration** MJCF and original STL assets, not hand-created approximate geometry. Its joint order and ranges are:

| index | joint | MJCF radian range |
|---:|---|---:|
| 0 | shoulder_pan | [-1.919862, 1.919862] |
| 1 | shoulder_lift | [-1.745329, 1.745329] |
| 2 | elbow_flex | [-1.690000, 1.690000] |
| 3 | wrist_flex | [-1.658063, 1.658063] |
| 4 | wrist_roll | [-2.743847, 2.841206] |
| 5 | gripper hinge | [-0.174533, 1.745329] |

The first six names/ordering match current LeRobot follower code: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`. LeRobot represents gripper opening as 0–100 (0 closed, 100 open); this environment maps that to 0–1 and explicitly converts it to the vendor MJCF hinge range. The model source notes that this mapping is not itself encoded in its MJCF.

Coordinates are MuJoCo world coordinates in metres: right-handed, +Z up; table top is Z=0, the robot base rests at Z=0, cube free-joint pose is XYZ + WXYZ quaternion. `gripperframe` from the CAD MJCF is the end-effector reference site. The SO-101 arm has five arm DOF, therefore its IK controls Cartesian position only and does not claim full orientation control.

## API

```python
from simulation.env import SO101PickCubeEnv
env = SO101PickCubeEnv()
obs, info = env.reset(seed=123)
obs, reward, terminated, truncated, info = env.step(action)
```

Observation:

- `rgb`: `uint8[H,W,3]`, default `120×160×3`, fixed external camera.
- `joint_pos`: `float32[5]`, five arm joint radians in the table order above.
- `gripper`: `float32[1]`, normalized opening, 0 closed / 1 open.

Action is `float32[6]`: absolute targets `[five arm radians, gripper_open]`. Arm targets are clipped to the exact model limits; gripper is clipped to `[0,1]` and converted internally. `info` exposes requested/applied action, cube/EE pose, contact count, grasp status, and success. `reset(seed=...)` is deterministic; `reset(options={"cube_xy": [x,y]})` allows an in-workspace replay pose.

## Expert and success

The expert commands only the public six-value action vector. It uses bounded damped-least-squares position IK and transitions:

`HOME → MOVE_ABOVE_OBJECT → DESCEND → CLOSE_GRIPPER → LIFT → HOLD → SUCCESS`.

Every state has a timeout. Success requires a verified closed/proximate grasp, cube elevation above the configured 0.10 m threshold, and persistence for five control steps. It is not defined from end-effector pose alone.

The IK command is clipped to exact actuator limits and kept 0.04 rad inside hard arm stops to absorb finite-stiffness servo settling. Failure categories are recorded as `HOME_TIMEOUT`, `APPROACH_TIMEOUT`, `DESCENT_TIMEOUT`, `GRASP_FAILED`, `LIFT_FAILED`, `OBJECT_DROPPED`, or `HOLD_TIMEOUT`.

The CAD collision meshes stay visual because their non-convex fingertip contacts are unstable in this lightweight MuJoCo model. Two small convex collision pads, placed from the CAD fingertip transforms, provide the fixed and moving finger contacts. They are enabled only for a closing gripper command, use normal MuJoCo rigid-body contact, and never weld, teleport, or otherwise edit cube state. Success requires simultaneous contact with both pads plus a held physical lift. This is still a sim-to-real limitation requiring hardware calibration before transfer.

## Dataset format

The simple intermediate format is chosen over direct `LeRobotDataset` integration so Phase 1 has no unverified API/version dependency. It is intentionally direct to convert: each `episodes/episode_000000/` has equal-length `.npy` arrays (`rgb`, `joint_pos`, `gripper`, `action`, `cube_pose`, `end_effector_pose`, `timestamp`, `expert_state`, `success`) and `episode.json`. End-effector pose is XYZ plus row-major 3×3 rotation matrix. Metadata carries task/seed/outcome, control and simulation rates, camera configuration, action definition, units, and joint names/limits. This maps cleanly to LeRobot observation/action features plus task metadata in Phase 2.

`.venv/bin/python -m data.validate DATASET` checks lengths, dimensions, finite values, action limits, timestamps, metadata and labels. `success[t]` labels the state reached *after* applying `action[t]`; the RGB/joint/gripper fields remain the pre-action `observation_t`.

## Investigation record

- Host: Ubuntu 26.04.1 LTS under WSL2; system Python 3.14.4; project targets available CPython 3.12.14 because the existing MuJoCo wheel/toolchain supports it.
- MuJoCo 3.13.0, NumPy 2.5.3, SciPy 1.18.1 and imageio 2.37.4 were found in an existing local Python 3.12 environment. Gymnasium, LeRobot, Torch, h5py and pytest were absent from system Python at investigation time.
- CUDA driver libraries are installed (`libcuda.so.1` through WSL), but this sandbox cannot query NVML (`GPU access blocked by the operating system`) and no `nvcc` toolkit binary is installed. That is a sandbox visibility finding, not a claim that the host lacks CUDA.
- No local SO-101/LeRobot asset source was present. The downloaded official CAD-derived source above was selected rather than manually reconstructing geometry.
- Current LeRobot source confirms STS3215 follower motor IDs 1–6 in the listed order, position control, body joints represented in degrees or calibrated ranges, and the gripper as `RANGE_0_100`. Its current kinematic processor also documents partial/soft orientation handling for the five-DOF SO-101.

## Phase 1 limitations

The camera is rendered for every stored demonstration. Under WSL, EGL rendering is software (llvmpipe, about 0.75 s/frame), so the
100-episode *control benchmark* disables rendering; it measures expert/contact robustness only.

**Note:** the 100-episode dataset was generated by an earlier version of the expert (state-feedback DLS IK in every phase, approach
target 0.10 m, joint margin 0.01), not the current `simulation/expert.py`. That expert is reconstructed bit-exactly in
`simulation/legacy_expert.py`, and all corrective-data labels use it.
