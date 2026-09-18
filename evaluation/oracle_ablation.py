"""Phase 6a oracle split: which half of the action causes closed-loop failure?

gripper_oracle: policy arm joints + the data-generating expert's gripper
                command (shadow FSM on the actual state: close only once the
                grasp center is within 4 mm of its target).
arm_oracle:     expert arm (DLS IK from the actual state) + policy gripper.
Receding horizon K as in evaluation/closed_loop.py; the same per-rollout
JSON/NPZ schema, so closed_loop_analysis applies unchanged. Resumable.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.closed_loop import _physical, _state
from evaluation.research_audit import save_json, sha256
from simulation.env import SO101PickCubeEnv
from simulation.legacy_expert import LegacyPickCubeExpert
from training.policy import TinyRDTPolicy

CLEAN10 = [0, 2, 3, 4, 5, 6, 7, 8, 9, 11]


def rollout(env, policy, seed, k, mode, max_steps):
    obs, info = env.reset(seed=seed); expert = LegacyPickCubeExpert(env); expert.reset(); policy.reset()
    rec = {"rgb": [obs["rgb"]], "state": [_state(obs)], "cube": [], "ee": [], "grasp_center": [], "grasped": [], "pad_contacts": [],
           "executed": [], "chunks": [], "chunk_steps": [], "policy_action": [], "expert_action": []}
    _physical(env, rec, False); plan = []; step = 0; success = False
    while step < max_steps:
        if not plan:
            chunk = policy.predict(obs["rgb"], _state(obs)).numpy(); rec["chunks"].append(chunk); rec["chunk_steps"].append(step); plan = list(chunk[:k])
        p = plan.pop(0); e = expert.action(); a = p.copy()
        if mode == "gripper_oracle": a[5] = e[5]
        else: a[:5] = e[:5]
        obs, _, terminated, truncated, info = env.step(a); expert.observe(info)
        for key, v in (("executed", a), ("policy_action", p), ("expert_action", e)): rec[key].append(np.asarray(v, np.float32))
        rec["rgb"].append(obs["rgb"]); rec["state"].append(_state(obs)); _physical(env, rec, bool(info["grasped"])); step += 1
        if terminated or truncated:
            success = bool(info["success"]); break
    return success, step, {key: np.asarray(v) for key, v in rec.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("gripper_oracle", "arm_oracle"), required=True)
    p.add_argument("--checkpoint", default="artifacts/research_audit/cosine_x0_hold_ema/ema_last.pt")
    p.add_argument("--episodes", type=int, nargs="+", default=CLEAN10)
    p.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--dataset", default="artifacts/pickcube_smoke100_rgb160")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    policy = TinyRDTPolicy(a.checkpoint, seed=0); env = SO101PickCubeEnv()
    for ep in a.episodes:
        seed = json.loads((Path(a.dataset) / "episodes" / f"episode_{ep:06d}" / "episode.json").read_text())["seed"]
        for k in a.k:
            stem = out / f"ep{ep:03d}_k{k}"
            if stem.with_suffix(".json").exists(): continue
            success, steps, rec = rollout(env, policy, seed, k, a.mode, a.max_steps)
            result = {"episode": ep, "seed": seed, "k": k, "mode": a.mode, "success": success, "steps": steps, "replans": len(rec["chunks"]),
                      "ever_grasped": bool(rec["grasped"].any()), "max_cube_z": float(rec["cube"][:, 2].max()),
                      "gripper_closed_ever": bool((rec["executed"][:, 5] < .5).any()), "checkpoint": a.checkpoint, "checkpoint_sha256": sha256(a.checkpoint)}
            np.savez_compressed(stem.with_suffix(".npz"), **rec)
            save_json(stem.with_suffix(".json"), result); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
