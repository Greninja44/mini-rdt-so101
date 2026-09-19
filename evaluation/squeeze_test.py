"""Pad-cube squeeze penetration under candidate contact parameters (physics-v2 calibration, read-only)."""
from __future__ import annotations
import itertools, json
from dataclasses import replace

import numpy as np

from simulation.env import PickCubeConfig, SO101PickCubeEnv
from simulation.expert_v2 import PickCubeExpertV2


def squeeze(solref_pad_cube, solimp=None, timestep=0.002, seed=3006, hold_steps=30):
    env = SO101PickCubeEnv(PickCubeConfig(render_observations=False, simulation_timestep=timestep))
    m, d = env.model, env.data
    for g in list(env._grasp_pad_ids) + [env._cube_geom]:
        m.geom_solref[g] = solref_pad_cube
        if solimp is not None: m.geom_solimp[g] = solimp
    env.reset(seed=seed); ex = PickCubeExpertV2(env); ex.reset(); t = 0
    while ex.state.value not in ("CLOSE", "FAILED") and t < 200:
        a = ex.action(); _, _, _, _, info = env.step(a); ex.observe(info); t += 1
    q_hold = d.qpos[env._joint_qposadr[:5]].copy(); pen = []; jaw = []; cube_v = []
    for _ in range(hold_steps):
        env.step(np.r_[q_hold, 0.0])
        p = [max([-d.contact[i].dist for i in range(d.ncon) if {d.contact[i].geom1, d.contact[i].geom2} == {g, env._cube_geom}] or [0]) for g in env._grasp_pad_ids]
        pen.append(p); jaw.append(float(d.qpos[env._joint_qposadr[5]])); cube_v.append(float(np.abs(d.qvel[env._cube_dofadr:env._cube_dofadr + 3]).max()))
    pen = np.array(pen)
    return {"final_pad_penetration_mm": (1000 * pen[-1]).round(2).tolist(), "max_pad_penetration_mm": float(1000 * pen.max()),
            "final_jaw_q": jaw[-1], "cube_speed_last10_max": max(cube_v[-10:]), "finite": bool(np.isfinite(d.qpos).all())}


if __name__ == "__main__":
    cases = [("default-ish 0.005", (0.005, 1.0), None, 0.002), ("stiffest stable 0.004 + solimp .99", (0.004, 1.0), (0.99, 0.999, 0.001, 0.5, 2.0), 0.002),
             ("dt 1 ms, 0.002", (0.002, 1.0), None, 0.001)]
    cases += [(f"direct k={k:g} b={b:g}", (-k, -b), None, 0.002) for k, b in itertools.product((2e4, 1e5, 5e5), (50, 200))]
    for name, ref, imp, dt in cases:
        print(json.dumps({"case": name, **squeeze(ref, imp, dt)}), flush=True)
