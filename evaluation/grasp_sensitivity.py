"""Physical grasp tolerance: replay recorded expert actions with controlled errors.

Rendering is disabled (physics only). Perturbations start at the expert's first
DESCEND step and persist to the end, like a biased policy:
  joint:  constant offset on one arm joint command
  timing: the gripper-close command shifted earlier/later by dt control steps
Recorded at the close step: grasp-center minus cube position, joint deviation,
and later pad contacts / grasp / lift / success.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.research_audit import save_json
from simulation.env import PickCubeConfig, SO101PickCubeEnv

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]


def run(env, seed, actions, close_t, extra_steps=40):
    obs, _ = env.reset(seed=seed)
    close_geom = None; grasped = False; max_z = float(env.cube_pose[2]); success = False; max_pads = 0
    seq = np.concatenate((actions, np.repeat(actions[-1:], extra_steps, 0)))
    for t, a in enumerate(seq):
        if t == close_t:
            close_geom = {"grasp_center_minus_cube_mm": (1000 * (env.grasp_center_position - env.cube_pose[:3])).tolist(),
                          "joint_pos": obs["joint_pos"].tolist()}
        obs, _, terminated, truncated, info = env.step(a)
        grasped |= bool(info["grasped"]); max_z = max(max_z, float(env.cube_pose[2]))
        bodies = {env._cube_gripper_contact_body(i) for i in range(env.data.ncon)}
        max_pads = max(max_pads, len(bodies & {env._gripper_body_id, env._moving_jaw_body_id}))
        if terminated:
            success = bool(info["success"]); break
    return {"success": success, "grasped": grasped, "max_cube_z": max_z, "max_pad_contacts": max_pads, "close": close_geom}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="artifacts/pickcube_smoke100_rgb160")
    p.add_argument("--episodes", type=int, nargs="+", default=[0, 2, 3, 4, 5, 6, 7, 8, 9, 11])
    p.add_argument("--offsets", type=float, nargs="+", default=[-0.08, -0.05, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.05, 0.08])
    p.add_argument("--timing", type=int, nargs="+", default=[-5, -3, -2, -1, 1, 2, 3, 5])
    p.add_argument("--output", required=True)
    a = p.parse_args()
    env = SO101PickCubeEnv(PickCubeConfig(render_observations=False))
    rows = []
    for ep in a.episodes:
        root = Path(a.dataset) / "episodes" / f"episode_{ep:06d}"
        meta = json.loads((root / "episode.json").read_text()); actions = np.load(root / "action.npy"); states = np.load(root / "expert_state.npy")
        descend = int(np.argmax(states == "DESCEND")); close = int(np.argmax(actions[:, 5] < 0.5))
        expert_close_q = np.load(root / "joint_pos.npy")[close]
        base = run(env, meta["seed"], actions, close)
        rows.append({"episode": ep, "kind": "none", "joint": None, "delta": 0.0, **base})
        for j, name in enumerate(JOINTS):
            for d in a.offsets:
                pert = actions.copy(); pert[descend:, j] += d
                r = run(env, meta["seed"], pert, close)
                r["close_joint_dev_rad"] = (np.asarray(r["close"]["joint_pos"]) - expert_close_q).tolist()
                rows.append({"episode": ep, "kind": "joint", "joint": name, "delta": d, **r})
        for dt in a.timing:
            pert = actions.copy(); grip = actions[:, 5].copy()
            shifted = np.clip(np.arange(len(grip)) - dt, 0, len(grip) - 1)
            pert[:, 5] = grip[shifted]
            new_close = int(np.argmax(pert[:, 5] < 0.5))
            r = run(env, meta["seed"], pert, new_close)
            rows.append({"episode": ep, "kind": "timing", "joint": "gripper_close", "delta": dt, **r})
        print(ep, "baseline success", base["success"], flush=True)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    save_json(out, {"description": __doc__, "rows": rows})


if __name__ == "__main__":
    main()
