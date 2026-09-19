# Physics-v2 specification (PRE-REGISTERED 2026-09-19, before any physics-v2 expert evaluation)

Physics-v1 had no robot↔table collision. Every success was a sandwich grasp with one pad ~10 mm inside the table
(`TABLE_COLLISION_AUDIT.md`, frozen as tag `physics-v1-invalid`). Physics-v2 is a new benchmark version (`PickCubeConfig.physics="v2"`,
now the default). `physics="v1"` stays available only to reproduce history, and it reproduces the old demos bit-exactly (`tests/test_corrective.py`).

## Collision model (`SO101PickCubeEnv._configure_collisions_v2`)

| pair | enabled | geometry |
|---|---|---|
| cube ↔ table | yes (bit 1) | boxes |
| fingertip pads ↔ cube | **always** (bit 2); v1 toggled them on only while closing | vendor-transform convex boxes |
| fingertip pads ↔ table | **yes** (bit 4) | boxes |
| moving links ↔ table | **yes** (bit 4) | vendor collision-class meshes (group 3); MuJoCo collides their **convex hulls** |
| links ↔ cube, links ↔ links | no | — |

Design decisions:
- **Robot collision uses the vendor collision meshes' convex hulls, not the high-resolution non-convex meshes.** Against the flat tabletop,
  a hull penetrates exactly as deep as the mesh's lowest vertex, so hull–table contact is exact for the relevant failure mode (going
  through the table).
- **Links do not collide with the cube.** The gripper-body hull (servo + fixed finger) spans the space between the jaws, so a hull–cube
  contact would block legitimate grasps. The cube is grasped through the pads. Finger–cube interpenetration is not physically enforced
  by the links; it is covered by the side-pinch contact geometry below.
- **Shoulder is included.** Its lowest point is 16 mm above the table and it rotates about a vertical axis, so it can never touch; it is
  included for consistency. The base is static, and MuJoCo never collides static bodies with the world.
- **Contact stiffness:** solref time constant 0.005 s (MuJoCo default 0.02 s) on the table, cube, pads and robot collision geometry.
  - At the default, servo-driven fingertips sank 6.8–7.0 mm into the table (the old trajectory replayed on v2).
  - At 0.005 s the resting cube sinks **0.047 mm**, and full-force table contact from the old trajectory reaches **0.14–0.40 mm**.
  - 0.005 s is 2.5× the 2 ms timestep, and the cube is stable at rest (|qvel| ~1e-9).

## Tolerances (from the measurements above)

| quantity | measured normal / adversarial | tolerance | rationale |
|---|---|---|---|
| robot/pad ↔ table penetration (contact depth) | 0 in normal motion; 0.14–0.40 mm when an invalid trajectory drives into the table at full actuator force | **≤ 1.0 mm** at every step | 2.5× the worst full-force value. Exceeding it means a solver failure or invalid geometry. |
| cube ↔ table penetration | 0.047 mm at rest | **≤ 1.0 mm** at every step | 20× the resting sink. The v1 sandwich pushed the cube 3–6 mm in. |

Incidental robot–table contact within tolerance is allowed and recorded, not failed.

## Valid side pinch (per step, from MuJoCo contacts: `contact_diagnostics`)
- Both pads have ≥1 contact with the cube.
- Each pad's mean contact normal (pad → cube) is within **30° of horizontal**: |n_z| ≤ 0.5.
- The two pads' mean normals are **opposing**: n_fixed · n_moving ≤ −0.5, i.e. at least 120° apart.
- **No pad–cube contact is vertical** (|n_z| > 0.7). That would be a top or underside (sandwich) contact.

## Success (physics-v2)
Success is 5 consecutive control steps satisfying all of:
- a valid side pinch **in that step**;
- cube center z ≥ 0.10 m;
- the episode has never exceeded either penetration tolerance.

This fixes two v1 weaknesses. v1 counted *any* two-pad contact, including the sandwich, and it only required a pinch *at some earlier time*
(`ever_grasped`), not during the hold.

## Invalidity and failure flags (recorded separately)
- robot–table contact (count);
- robot–table penetration (max);
- cube–table penetration (max);
- vertical / underside pad contact;
- `invalid_reason`: the first tolerance exceeded;
- early close (analysis: close commanded before the pads straddle the cube);
- failed pinch (closed without a valid side pinch);
- cube drop (lifted ≥ 3 cm, then fell below 2 cm).

## Expert validation protocol (before any data collection)
1. 10 deterministic seeds (the CLEAN10 cube seeds 3000 + {0,2,3,4,5,6,7,8,9,11}), then **100** randomised seeds (4000–4099).
2. Record per rollout:
   - success;
   - max robot–table and cube–table penetration;
   - pad contact normals;
   - side-pinch validity;
   - joint-limit margin;
   - time to grasp;
   - time to success.
3. Render representative task-camera and side-camera frames and inspect them manually.
4. **The tolerances above are fixed.** Expert failures are investigated and fixed in the controller. They are never fixed by loosening
   physics, tolerances or the side-pinch test. The success rate is not forced to 100%.
