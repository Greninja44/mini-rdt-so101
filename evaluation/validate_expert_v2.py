"""Physics-v2 expert validation (docs/research/physics_v2_spec.md): physics-only rollouts with full diagnostics.

Per rollout:
- success;
- max robot-table and cube-table penetration;
- pad contact normals at the first pinch;
- side-pinch validity through the hold;
- finger-mesh / cube interpenetration (links do not collide with the cube);
- joint-limit margin;
- time to grasp and time to success.

Representative rollouts are rendered from the task camera and a side camera.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from simulation.env import PickCubeConfig, SO101PickCubeEnv
from simulation.expert_v2 import EXPERT_V2_VERSION, PickCubeExpertV2

CLEAN10_SEEDS = [3000 + e for e in (0, 2, 3, 4, 5, 6, 7, 8, 9, 11)]


def finger_cube_interpenetration(env):
    """Deepest penetration (m) of any gripper / moving-jaw collision-mesh vertex into the cube box."""
    m, d = env.model, env.data; half = env.config.cube_size / 2
    cpos = env.cube_pose[:3]; w, x, y, z = env.cube_pose[3:]
    Rc = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    worst = 0.0
    for g in env._robot_collision_geoms:
        if m.geom_bodyid[g] not in (env._gripper_body_id, env._moving_jaw_body_id) or m.geom_dataid[g] < 0: continue
        mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        local = ((v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]) - cpos) @ Rc
        inside = np.all(np.abs(local) < half, axis=1)
        if inside.any(): worst = max(worst, float((half - np.abs(local[inside])).min(axis=1).max()))
    return worst


def rollout(env, seed, frames_for=None):
    obs, info = env.reset(seed=seed); ex = PickCubeExpertV2(env); ex.reset(); t = 0; info = {}
    lo, hi = env._lo[:5], env._hi[:5]; margin = np.inf; t_grasp = None; pinch_normals = None; fc = 0.0; frames = []
    hold_valid = []; vertical = False
    while not ex.done and t < env.config.max_episode_steps:
        a = ex.action(); obs, _, term, _, info = env.step(a); ex.observe(info); t += 1
        q = env.data.qpos[env._joint_qposadr[:5]]; margin = min(margin, float(min((q - lo).min(), (hi - q).min())))
        c = info["contacts"]; vertical |= c["vertical_pad_contact"]; fc = max(fc, finger_cube_interpenetration(env))
        if c["side_pinch"] and t_grasp is None: t_grasp, pinch_normals = t, c["pad_normals"]
        if info["elevated"]: hold_valid.append(c["side_pinch"])
        if frames_for is not None: frames.append((t, ex.state.value, frames_for(), c, float(env.cube_pose[2])))
    return {"seed": seed, "success": bool(info.get("success")), "expert_state": ex.state.value, "failure": ex.failure_category,
            "steps": t, "time_to_grasp_s": None if t_grasp is None else t_grasp / 20, "time_to_success_s": t / 20 if info.get("success") else None,
            "max_robot_table_penetration_mm": 1000 * info.get("max_robot_table_penetration", 0.0),
            "max_cube_table_penetration_mm": 1000 * info.get("max_cube_table_penetration", 0.0),
            "max_finger_cube_interpenetration_mm": 1000 * fc, "any_vertical_pad_contact": bool(vertical),
            "pad_normals_at_first_pinch": pinch_normals, "side_pinch_all_lifted_steps": bool(hold_valid) and all(hold_valid),
            "joint_limit_margin_rad": margin, "grasp_dz_mm": 1000 * (ex.plan.get("grasp_dz") or 0), "cube_xy": env.cube_pose[:2].tolist(),
            "invalid_reason": info.get("invalid_reason")}, frames


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", default="artifacts/physics_v2/expert_validation")
    p.add_argument("--random", type=int, default=100); p.add_argument("--render", type=int, default=3)
    a = p.parse_args(); out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    env = SO101PickCubeEnv(PickCubeConfig(render_observations=False))
    renderer = mujoco.Renderer(env.model, height=300, width=400); side = mujoco.MjvCamera(); side.type = mujoco.mjtCamera.mjCAMERA_FREE
    def shot():
        renderer.update_scene(env.data, camera="external_rgb"); task = renderer.render().copy()
        c = env.cube_pose[:3]; side.lookat[:] = [c[0], c[1], 0.03]; side.distance = 0.22; side.azimuth = 90.0; side.elevation = -8.0
        renderer.update_scene(env.data, camera=side); return task, renderer.render().copy()
    results = {"expert_version": EXPERT_V2_VERSION, "physics": env.config.physics}
    for name, seeds in (("clean10", CLEAN10_SEEDS), ("random100", list(range(4000, 4000 + a.random)))):
        rows = []
        for i, seed in enumerate(seeds):
            r, frames = rollout(env, seed, shot if i < a.render else None); rows.append(r)
            for t, state, (task, sv), c, cz in frames:
                if state in ("CLOSE", "LIFT", "HOLD", "SUCCESS", "DESCEND") and (t % 4 == 0 or state == "SUCCESS"):
                    img = Image.new("RGB", (800, 340), (252, 252, 251)); img.paste(Image.fromarray(task), (0, 0)); img.paste(Image.fromarray(sv), (400, 0))
                    ImageDraw.Draw(img).text((6, 304), f"{name} seed {seed} t={t} {state} | side pinch {c['side_pinch']} | pad normals {[None if n is None else [round(x, 2) for x in n] for n in c['pad_normals']]} | cube z {1000 * cz:.1f} mm", fill=(11, 11, 11))
                    ImageDraw.Draw(img).text((6, 320), f"robot-table pen {1000 * c['robot_table_penetration']:.2f} mm  cube-table pen {1000 * c['cube_table_penetration']:.3f} mm  robot-table contacts {c['robot_table_contacts']}", fill=(82, 81, 78))
                    img.save(out / f"{name}_seed{seed}_t{t:03d}_{state}.png")
        ok = [r for r in rows if r["success"]]
        results[name] = {"n": len(rows), "success": len(ok),
                         "all_successes_valid_side_pinch": all(r["side_pinch_all_lifted_steps"] and not r["any_vertical_pad_contact"] for r in ok),
                         "max_robot_table_penetration_mm": max(r["max_robot_table_penetration_mm"] for r in rows),
                         "max_cube_table_penetration_mm": max(r["max_cube_table_penetration_mm"] for r in rows),
                         "max_finger_cube_interpenetration_mm": max(r["max_finger_cube_interpenetration_mm"] for r in rows),
                         "min_joint_limit_margin_rad": min(r["joint_limit_margin_rad"] for r in rows),
                         "time_to_success_s_median": float(np.median([r["time_to_success_s"] for r in ok])) if ok else None,
                         "failures": [{k: r[k] for k in ("seed", "failure", "expert_state", "cube_xy", "invalid_reason")} for r in rows if not r["success"]], "rows": rows}
        print(name, {k: v for k, v in results[name].items() if k != "rows"}, flush=True)
    (out / "validation.json").write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
