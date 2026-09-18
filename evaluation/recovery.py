"""Recovery benchmark (pre-registered, CLAUDE_PROGRESS.md Phase 5).

The data-generating (legacy) expert drives to step T0=6, a command offset delta
is applied to one joint for 3 steps, then the policy takes over with
receding-horizon execution (K). Measures whether the policy returns to a
successful grasp from a controlled, physically realised deviation.
Rendering is off until handover. Each rollout is saved on completion (resumable).
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from evaluation.closed_loop import _physical, _state
from evaluation.research_audit import save_json, sha256
from simulation.env import SO101PickCubeEnv
from simulation.legacy_expert import LEGACY_JOINT_LIMIT_MARGIN, LegacyPickCubeExpert
from training.policy import TinyRDTPolicy

CLEAN10 = [0, 2, 3, 4, 5, 6, 7, 8, 9, 11]
JOINTS = {"shoulder_pan": 0, "shoulder_lift": 1}
T0, DURATION = 6, 3


def one(env, policy, ep, seed, joint, delta, k, max_steps, nominal_q, nominal_gc):
    env.config = replace(env.config, render_observations=False)
    obs, info = env.reset(seed=seed); expert = LegacyPickCubeExpert(env); expert.reset()
    lo, hi = env._lo[:5] + LEGACY_JOINT_LIMIT_MARGIN, env._hi[:5] - LEGACY_JOINT_LIMIT_MARGIN
    for t in range(T0 + DURATION):
        a = expert.action()
        if t >= T0: a = a.copy(); a[joint] = np.clip(a[joint] + delta, lo[joint], hi[joint])
        obs, _, _, _, info = env.step(a); expert.observe(info)
    env.config = replace(env.config, render_observations=True); obs = dict(obs, rgb=env.render())
    handover = {"joint_dist_rad": float(np.linalg.norm(nominal_q - obs["joint_pos"], axis=1).min()),
                "grasp_center_path_mm": 1000 * float(np.linalg.norm(nominal_gc - env.grasp_center_position, axis=1).min())}
    rec = {"rgb": [obs["rgb"]], "state": [_state(obs)], "cube": [], "ee": [], "grasp_center": [], "grasped": [], "pad_contacts": [],
           "executed": [], "chunks": [], "chunk_steps": []}
    _physical(env, rec, False); policy.reset(); step = 0; success = False
    while step < max_steps and not success:
        chunk = policy.predict(obs["rgb"], _state(obs)).numpy(); rec["chunks"].append(chunk); rec["chunk_steps"].append(step)
        for a in chunk[:k]:
            obs, _, terminated, truncated, info = env.step(a)
            rec["executed"].append(np.asarray(a, np.float32)); rec["rgb"].append(obs["rgb"]); rec["state"].append(_state(obs)); _physical(env, rec, bool(info["grasped"]))
            step += 1
            if terminated or step >= max_steps:
                success = bool(info["success"]); break
    rec = {k2: np.asarray(v) for k2, v in rec.items()}
    return {"success": success, "steps_after_handover": step, "handover": handover, "grasped": bool(rec["grasped"].any()),
            "max_cube_z": float(rec["cube"][:, 2].max()), "replans": len(rec["chunks"])}, rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--magnitudes", type=float, nargs="+", default=[.01, .02, .03, .04])
    p.add_argument("--episodes", type=int, nargs="+", default=CLEAN10)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--dataset", default="artifacts/pickcube_smoke100_rgb160")
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    policy = TinyRDTPolicy(a.checkpoint, seed=0); env = SO101PickCubeEnv()
    for i, ep in enumerate(a.episodes):
        root = Path(a.dataset) / "episodes" / f"episode_{ep:06d}"; seed = json.loads((root / "episode.json").read_text())["seed"]
        nq = np.load(root / "joint_pos.npy"); ngc = np.load(Path("artifacts/closed_loop_v2/tinyrdt_ema") / f"ep{ep:03d}_expert_replay.npz")["grasp_center"]
        sign = 1.0 if CLEAN10.index(ep) % 2 == 0 else -1.0  # fixed per seed, independent of --episodes subset
        for jname, j in JOINTS.items():
            for m in a.magnitudes:
                stem = out / f"ep{ep:03d}_{jname}_{m:.2f}"
                if stem.with_suffix(".json").exists(): continue
                result, rec = one(env, policy, ep, seed, j, sign * m, a.k, a.max_steps, nq, ngc)
                result.update({"episode": ep, "seed": seed, "joint": jname, "magnitude_rad": m, "sign": sign, "k": a.k, "t0": T0, "duration_steps": DURATION,
                               "checkpoint": a.checkpoint})
                np.savez_compressed(stem.with_suffix(".npz"), **rec)
                imageio.mimsave(stem.with_suffix(".gif"), [np.kron(f, np.ones((2, 2, 1), dtype=np.uint8)) for f in rec["rgb"]], duration=50, loop=0)
                save_json(stem.with_suffix(".json"), result)
                print(json.dumps(result), flush=True)
    rows = [json.loads(f.read_text()) for f in sorted(out.glob("ep*.json"))]
    table = {f"{m:.2f}": {"n": len(r := [x for x in rows if abs(x["magnitude_rad"] - m) < 1e-9]), "success": sum(x["success"] for x in r)} for m in a.magnitudes}
    save_json(out / "recovery_summary.json", {"checkpoint": a.checkpoint, "checkpoint_sha256": sha256(a.checkpoint), "k": a.k, "t0": T0, "table": table, "rows": rows})
    print(json.dumps(table))


if __name__ == "__main__":
    main()
