# Physics-v2 report: can MiniRDT learn a genuinely physical side-grasp PickCube task?

**Answer: yes.** With table collisions modelled correctly, the same 2,009,670-parameter TinyRDT (frozen MobileNetV3-S, 4 layers,
d=192, H=16; cosine / x0 / hold-padding / EMA) memorises 10 physically valid side-grasp demonstrations offline and executes them in
closed loop:
- **49/50** rollouts over K ∈ {1, 2, 4, 8, 16} on the 10 memorised cubes (**10/10 at every K ≥ 2**);
- every success is a valid opposing side pinch with **0.00 mm** robot–table penetration;
- expert replay 10/10.

This is a new benchmark version. All physics-v1 manipulation results are invalid (`TABLE_COLLISION_AUDIT.md`) and are **not** compared as
equals below. This phase answers only the memorised-scene question. Held-out generalisation is untested.

## 1. Root cause (original audit)
- Physics-v1 disabled every robot collision geom (contype/conaffinity 0), and the fingertip pads collided only with the cube. Robot–table
  contact was impossible.
- The data-generating expert swept the gripper up to ~100 mm through the table during approach.
- **Every** success (10/10 expert demos, 28/28 TinyRDT) was a sandwich grasp: one pad ~10 mm inside the table under the cube, the other
  pressing down, the cube pushed 3–6 mm into the table, all contact normals vertical.
- The v1 success rule accepted any two-pad contact and only required a pinch *at some point*, not during the hold.

The v2 work found two further v1 model defects:
- **The moving pad sat ~10 mm inside the jaw**, near its outer face, so the jaw closed *through* the cube.
- **The 10 g cube's soft contacts let the force-limited gripper squeeze ~9 mm into it** (MuJoCo soft-contact stiffness is mass-normalised).

## 2. Exact collision changes (`SO101PickCubeEnv._configure_collisions_v2`; `physics="v2"` is the default, `"v1"` is kept bit-exact)

| pair | v1 | v2 |
|---|---|---|
| robot links ↔ table | none | vendor collision meshes of shoulder, upper_arm, lower_arm, wrist, gripper and moving jaw (convex hulls), bit 4 |
| pads ↔ table | none | bit 4 |
| pads ↔ cube | only while closing | always, bit 2 |
| cube ↔ table | yes | yes, bit 1 |
| links ↔ cube, links ↔ links | none | none (by design; §3) |
| moving pad | jaw-frame centre x = 0 (inside the jaw) | x = −10.3 mm, face at the measured jaw inner surface (−12.3 mm), half (2, 6, 6) mm |
| contact solref / solimp | default (0.02 s) | links 0.005 s; **pads, cube and table 0.004 s with solimp 0.99–0.999** |
| success | ever pinched + elevated | side pinch *each* hold step + elevated + no invalid penetration in the episode |

## 3. Collision geometry design
- **Convex hulls of the vendor collision meshes, not ~30 high-resolution mesh–mesh contacts.** Against the flat tabletop, a hull penetrates
  exactly as deep as its lowest vertex, so it is exact for the failure mode that matters.
- **Links do not collide with the cube.** The gripper-body hull (servo + fixed finger) spans the space between the jaws and would block
  legitimate grasps. The cube is grasped through the pads.
  - Link intrusion into the cube is instead checked at validation time: the finger-mesh ↔ cube check, ≤ 2 mm; measured ≤ 1.18 mm.
- **The shoulder is included** even though it cannot reach the table (lowest point 16 mm, vertical rotation axis). The base is static.
- **The fixed pad already matched its finger face** (−8.0 vs −7.9 mm). **The moving pad was relocated** to the jaw's measured flat fingertip
  face (x = −12.3 mm over y −82…−70, z 15…23 mm).

## 4. Contact tests (`tests/test_physics_v2.py`; physics only; no dataset needed)

| test | result |
|---|---|
| collision bits: robot/pad↔table, pad↔cube and cube↔table enabled; links↔cube disabled | pass |
| **A** fingertips commanded 30 mm *below* the tabletop | robot–table contacts on >10 steps, penetration < 1.0 mm (at full actuator force: 0.03–0.14 mm) |
| **B** old sandwich trajectory replayed on v2 | no success, no penetration; the lower jaw never gets under the cube |
| **C** 20 N downward push on the cube (≈200× its weight) | cube–table 0.40 mm (1.78 mm before amendment 2) |
| legacy (v1) sandwich grasp classified | `vertical_pad_contact` True, `side_pinch` False |
| **D** v2 expert on 3 seeds | success, side pinch through the hold, opposing normals |
| success impossible with invalid penetration (cube tolerance 0) | no success, `invalid_reason = cube_table_penetration` |
| v1 still selectable and distinct | pass; `tests/test_corrective.py` reproduces the v1 demos bit-exactly |

## 5. New expert (`simulation/expert_v2.py`, `expert-v2-sidepinch-3`)
- **Phases:** HOME → PREGRASP → DESCEND → CLOSE → LIFT → HOLD → SUCCESS.
- **Grasp pose:** top-down, with the jaw-closing axis aligned to a cube face normal (yaw = cube yaw + k·90°, nearest to radial). The cube sits
  5 mm from the fixed pad before closing.
- **Grasp height:** chosen at reset by a read-only check that every finger, jaw and pad hull clears the table by ≥ 3 mm and that ≥ 6 mm of pad
  overlaps the cube's side (median +9 mm above the cube centre).
- **State feedback:**
  - Approach is joint-space feedback from the *actual* configuration toward the pre-grasp IK solution. A Cartesian approach from HOME hit an
    IK local minimum (wrist at its limit).
  - Descent and lift use Cartesian feedback from the actual pose, solved by 6-D pose IK (`simulation/pose_ik.py`).
  - Final descent (≤ 30 mm above the grasp point) moves 3 mm per tick and re-centres whenever the lateral error exceeds 2.5 mm. This fixed
    seed 4079, where a pad landed on the cube's top.
- **Close / lift:** close until a valid side pinch persists 3 steps. Lift with relaxed orientation, because top-down poses above ~0.08 m hit
  the wrist limit.

## 6. Side-grasp definition (per step, from MuJoCo contacts)
- Both pads touch the cube.
- Each pad's mean contact normal (pad → cube) is within **30° of horizontal** (|n_z| ≤ 0.5).
- The two mean normals are **opposing** (dot product ≤ −0.5).
- **No pad–cube contact is vertical** (|n_z| > 0.7).

## 7. Success criterion v2
5 consecutive control steps, each with:
- a valid side pinch;
- cube centre z ≥ 0.10 m;
- no invalid penetration anywhere in the episode so far.

Invalidity flags, recorded separately: robot–table contact count and penetration, cube–table penetration, pad–cube penetration, vertical pad
contact and `invalid_reason`. Analysis adds early close, failed pinch and drop.

## 8. Tolerances (pre-registered in `docs/research/physics_v2_spec.md` from measured solver behaviour, before any expert evaluation)

| quantity | tolerance | justification |
|---|---|---|
| robot/pad ↔ table penetration | ≤ 1.0 mm | full-force invalid drive-in reaches 0.03–0.40 mm |
| cube ↔ table penetration | ≤ 1.0 mm | resting 0.003 mm; 20 N push 0.40 mm |
| pad ↔ cube penetration (amendment 1, added) | ≤ 1.5 mm | full squeeze 0.6–0.96 mm |
| finger mesh ↔ cube (validation-level) | ≤ 2.0 mm | measured ≤ 1.18 mm |

The two amendments only **added** checks or **stiffened** contacts. No tolerance was loosened, and no failure was fixed by relaxing a
criterion.

## 9. Expert validation: 10 deterministic seeds (CLEAN10 cubes, seeds 3000 + {0,2,3,4,5,6,7,8,9,11})
- **10/10** success, all valid side pinches through the hold.
- Penetration: robot–table **0.00 mm**; cube–table ≤ 0.24 mm; pad–cube ≤ 0.96 mm; finger mesh ≤ 1.15 mm.
- Minimum joint-limit margin 3.8 mrad (a joint briefly reaches a hard stop during the orientation-relaxed lift; commands stay within limits).
- Median time to grasp 2.0 s, to success 2.7 s.

## 10. Expert validation: 100 randomised seeds (4000–4099)
- **100/100** success, all valid side pinches.
- Penetration: robot–table **0.00 mm**; cube–table ≤ 0.25 mm; pad–cube ≤ 0.96 mm; finger mesh ≤ 1.18 mm.
- Minimum joint-limit margin 2.5 mrad.

History: the first pass (before the descent fix) was 99/100. The failure (seed 4079, a pad landing on the cube's top) was investigated and
fixed in the controller.

## 11. Representative frames
- `docs/assets/rollout_strip.png`: TinyRDT side view. Start, approach, aligned above the cube, side pinch with fingertips above the table,
  lifted.
- `docs/assets/rollout_success.gif`: task camera + side view.
- Expert task and side frames: `artifacts/physics_v2/expert_validation/*.png` (inspected manually). E.g. the seed-4000 close frame shows both
  fingers straddling the cube, pad normals exactly (±1, 0, 0), and the cube resting on the table.
- The one pre-grasp nudge case: `artifacts/physics_v2/ep007_k1_side_strip.png` (§16).

## 12. Dataset v2 (`artifacts/pickcube_physics_v2_rgb160`; the v1 dataset is untouched)
- **Size:** 100 episodes, **100 successes, 0 discarded**, seeds 3000–3099 (same cube positions as v1). 5,408 frames, lengths 52–56 (mean
  54.1; v1 mean 48.3). CLEAN10 = episodes [0,2,3,4,5,6,7,8,9,11] (544 frames and windows).
- **Metadata:** physics version, expert version, git commit, effective-model fingerprint
  (`ce8794577b46…`, a hash of the compiled collision/contact arrays), environment and camera config, cube randomisation, success definition.
- **Per-step contact diagnostics:** side pinch, the three penetration depths, pad normals and robot–table contact count.
- `data/validate_v2.py`: **valid, 0 errors**. Max robot–table 0.00 mm, cube–table 0.33 mm, pad–cube 0.96 mm; every episode ends in 5
  side-pinch steps; no vertical contact; joints and actions within limits; single expert version and fingerprint.

## 13. 2M CLEAN10 offline results (same recipe as physics-v1 model A; 20k steps, 13 min; encoder = the same frozen ImageNet MobileNet)

| metric (EMA, 3 sampler seeds) | physics-v2 | physics-v1 model A (offline only) |
|---|---|---|
| all-window MAE | 0.0049–0.0050 | 0.0067 |
| worst arm joint MAE (rad) | 0.0100–0.0101 | 0.016 |
| gripper MAE | 0.0025–0.0028 | 0.003 |
| worst quarter / episode start | 0.0068–0.0070 / 0.0051 | 0.017 / 0.006 |
| pre-registered gate | **PASS** | PASS |

- **Per-joint MAE:** pan 0.0016, lift 0.0060, elbow 0.0082, wrist_flex 0.0100, roll 0.0013, gripper 0.0025.
- **Sampler agreement:** DDIM-5/10/20/50/100 0.0049–0.0050; DDPM-100 0.0045; q-terminal start 0.0050.
- **Timestep error:** raw x0 MSE 4.2e-5 (t=0), 8.1e-5 (t=50), 1.2e-4 (t=99). No high-noise collapse.
- **Conditioning at t=99:** wrong image 0.024, zero image features 0.025 (vs 0.0051 correct).

## 14. 2M CLEAN10 closed-loop (same 10 scenes, DDIM-10, max 150 steps)
- **49/50 success**; expert replay 10/10.
- **All 49 successes satisfy the v2 criterion** (side pinch every hold step, no invalid penetration).
- Across all 50 rollouts: max robot–table **0.00 mm**, cube–table 0.57 mm, pad–cube 1.14 mm.
- **Early closes: 0.** Median lateral grasp-centre ↔ cube offset at the close: **1.95 mm**. Median 53 steps (2.65 s) to success.

## 15. K sweep

| K | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| success | 9/10 | **10/10** | **10/10** | **10/10** | **10/10** |
| replans (median) | 51.5 | 27 | 14 | 7 | 4 |

With n=10 per K this doesn't rank K (K=1 vs K≥2: a single failure). Every K ≥ 2 was perfect.

## 16. Failure breakdown
- **The only failure: ep0, K=1** (`closed_without_valid_side_pinch` phase label; really *never closed*). The policy hovered with a pad resting
  on the cube's top from t=33 to the 150-step limit, never commanding a close. Cube–table penetration stayed at 0.55 mm, within tolerance.
  The rollout is correctly **not** counted as a success.
- **Two successes (ep7, K=1 and K=2)** had a pre-grasp vertical pad contact (t≈31–42). The descending moving jaw tipped the cube onto an
  edge (~40°); the closing jaws re-squared it into a clean side pinch, and it lifted.
  - Physically valid: everything stayed above the table (cube–table 0.57 mm) and the hold had valid side pinches. It's a messier grasp
    than the expert's.
  - They count as successes under the pre-registered criterion. My stricter diagnostic flag `all_successes_physically_valid`, which also
    forbids *any* pre-grasp vertical contact, is therefore false because of these 2.
- **Drops: 0. Invalid-penetration failures: 0.**

## 17. Comparison with physics-v1 (**v1 manipulation numbers are INVALID** and shown only as history)

| | physics-v1 (INVALID) | physics-v2 |
|---|---|---|
| expert success | 100/100 (all sandwich grasps through the table) | 100/100 + 10/10, all valid side pinches |
| TinyRDT CLEAN10 closed loop | 28/40 (K 1–8), 9/10 at K=16 | 39/40 (K 1–8); 49/50 including K=16 |
| max robot–table penetration | ~100 mm | 0.00 mm |
| pad inside table at pinch | ~10 mm | 0 mm |
| cube pushed into table | 3–6 mm | ≤ 0.57 mm |

The v1 closed-loop difficulty (misaligned "grasps" toward neighbouring cubes) was measured on physically invalid behaviour. Under valid
physics, with a well-conditioned expert, the same model is near-perfect on memorised scenes. **Do not** read this as "v1 conclusions
reversed"; the v1 numbers simply do not describe a physical task.

## 18. Old conclusions that remain valid
- **The diffusion diagnosis:**
  - the linear-β terminal-SNR mismatch (ᾱ₉₉ = 0.364);
  - ε-parameterisation blow-up at zero SNR (×4.1e6 error amplification);
  - the cosine + x0 fix, hold padding and EMA.
- **Sampler oracle tests; DDIM/DDPM agreement.**
- **Infrastructure:**
  - the bit-exact legacy-expert reconstruction (as a record of what produced the v1 data);
  - counterfactual expert-label machinery;
  - evaluation and recovery tooling;
  - pre-registration practice;
  - resumable pipelines.

## 19. Invalidated conclusions
- **Every physics-v1 manipulation success rate:** expert, TinyRDT, BC, DAgger / perturbation, spatial tokens, state dropout, image-only,
  80-demo, varied-start and Phase-10 fine-tuned vision.
- **Physical measurements on sandwich grasps:** the grasp-tolerance envelope (±0.03 rad / 7 mm).
- **Failure-isolation conclusions involving grasp geometry:** the oracle split, the expert-prefix and scene-blending mechanism, and K
  comparisons.

These all describe the invalid benchmark. They should be re-measured only if the question matters under v2.

## 20. Reproduction
```bash
git checkout physics-v2
.venv/bin/python -m pytest -q                                          # 41 tests (dataset-dependent ones skip in a fresh clone)
.venv/bin/python -m evaluation.squeeze_test                            # grasp-contact calibration
.venv/bin/python -m evaluation.validate_expert_v2                      # expert: 10 fixed + 100 random seeds
.venv/bin/python -m data.collect_v2 --start 0 --count 100              # dataset v2 (split across workers with --start/--count)
.venv/bin/python -m data.validate_v2 artifacts/pickcube_physics_v2_rgb160
scripts/experiments/run_physics_v2.sh                                  # train 2M TinyRDT, offline gate, closed-loop K sweep (resumable)
.venv/bin/python -m evaluation.v2_closed_loop_summary --output artifacts/physics_v2/closed_loop_summary.json
.venv/bin/python -m evaluation.table_collision_audit --artifacts artifacts   # re-audit of the v1 history (runs under physics v1)
.venv/bin/python scripts/make_readme_assets.py --artifacts artifacts
```

## 21. Files changed
- **New:**
  - `simulation/expert_v2.py`, `simulation/pose_ik.py`;
  - `data/collect_v2.py`, `data/validate_v2.py`;
  - `evaluation/validate_expert_v2.py`, `evaluation/squeeze_test.py`, `evaluation/v2_closed_loop_summary.py`;
  - `tests/test_physics_v2.py`;
  - `scripts/experiments/run_physics_v2.sh`;
  - `docs/research/physics_v2_spec.md`, `docs/research/physics_v1_freeze.json`;
  - this report.
- **Modified:**
  - `simulation/env.py`: versioned physics, v2 collisions, contact diagnostics, v2 success, v1 reset guard;
  - `evaluation/closed_loop.py`: v2 validity fields, `--physics`;
  - `evaluation/table_collision_audit.py`: pinned to v1;
  - v1-era tools and tests pinned to `physics="v1"`: `data/collect.py`, `evaluation/expert_benchmark.py`, `simulation/demo.py`,
    `tests/test_phase1.py`, `tests/test_corrective.py`;
  - `scripts/make_readme_assets.py`: v2 media;
  - README; invalidation banners on the v1 reports and log.
- **Moved:** the v1 manipulation figures → `docs/assets/physics_v1_invalid/`.
- **Nothing deleted:** the v1 datasets, checkpoints, rollouts and reports are kept; tag `physics-v1-invalid`.

## 22. Remaining limitations
- **Memorised scenes only** (10 cubes seen in training). Held-out cube positions under v2 are untested.
- **Links do not collide with the cube.** It's checked, not enforced (finger mesh ≤ 1.2 mm).
- **Grasp contacts sit at MuJoCo's stability limit** (timeconst = 2·dt). A smaller timestep would allow stiffer contacts.
- **Narrow task:** the cube yaw is always 0, and there is a single cube size and colour.
- **Wrist limit:** the lift uses relaxed orientation, and a joint touches a hard stop briefly.
- **n=10 per K.**
- **Sim-to-real untouched:** pad friction, servo model and camera realism are uncalibrated.

**Recommended next phase (not started):**
1. Physics-v2 generalisation: train the same 2M recipe on the 80-episode v2 training split, evaluate on the 10 held-out validation cubes
   (not test), with a fixed K chosen a priori (e.g. K=8).
2. Only then consider the capacity-scaling series.
