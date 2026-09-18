"""Receding-horizon TinyRDT execution in MuJoCo PickCube.

observe -> predict H action chunk -> execute first K -> observe again.
Cube configurations are selected by recorded episode seed, so "memorized"
runs reproduce exactly the training scenes. The recorded expert actions are
replayed open-loop as an environment-determinism control. Every rollout
stores camera frames, predicted chunks, executed actions, joint states, cube
and end-effector pose, success, and a GIF.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluation.research_audit import sha256, save_json
from simulation.env import SO101PickCubeEnv
from training.policy import BCPolicy, TinyRDTPolicy

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _state(obs):
    return np.r_[obs["joint_pos"], obs["gripper"]].astype(np.float32)


def _pad_contacts(env):
    bodies = {env._cube_gripper_contact_body(i) for i in range(env.data.ncon)}
    return len(bodies & {env._gripper_body_id, env._moving_jaw_body_id})


def _physical(env, rec, grasped):
    rec["cube"].append(env.cube_pose.copy()); rec["ee"].append(env.end_effector_pose.copy())
    rec["grasp_center"].append(env.grasp_center_position.copy()); rec["grasped"].append(grasped)
    rec["pad_contacts"].append(_pad_contacts(env))


def rollout(env, seed, act, max_steps, recorded_first_rgb=None):
    obs, info = env.reset(seed=seed)
    rec = {"rgb": [obs["rgb"]], "state": [_state(obs)], "cube": [], "ee": [], "grasp_center": [], "grasped": [], "pad_contacts": [],
           "executed": [], "chunks": [], "chunk_steps": []}
    _physical(env, rec, False)
    reset_rgb_error = None if recorded_first_rgb is None else int(np.abs(obs["rgb"].astype(int) - recorded_first_rgb.astype(int)).max())
    success = False; step = 0; max_cube_z = float(env.cube_pose[2]); ever_grasped = False
    while step < max_steps:
        actions = act(obs, step, rec)
        for a in actions:
            obs, _, terminated, truncated, info = env.step(a)
            rec["executed"].append(np.asarray(a, dtype=np.float32)); rec["rgb"].append(obs["rgb"])
            rec["state"].append(_state(obs)); _physical(env, rec, bool(info["grasped"]))
            step += 1; max_cube_z = max(max_cube_z, float(env.cube_pose[2])); ever_grasped |= bool(info["grasped"])
            if terminated or truncated or step >= max_steps:
                success = bool(info["success"]); break
        if success or truncated: break
    return {"success": success, "steps": step, "max_cube_z": max_cube_z, "ever_grasped": ever_grasped,
            "reset_rgb_max_abs_error_vs_recorded": reset_rgb_error}, {k: np.asarray(v) for k, v in rec.items()}


def save_plot(path, rec, expert_actions, title):
    executed = rec["executed"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(expert_actions[:, j], "k--", lw=1.2, label="expert (recorded)")
        ax.plot(executed[:, j], "C0", lw=1.4, label="policy executed")
        for s, chunk in zip(rec["chunk_steps"][::4], rec["chunks"][::4]):
            ax.plot(range(s, s + len(chunk)), chunk[:, j], "C1", lw=.6, alpha=.5)
        ax.set_title(NAMES[j]); ax.grid(alpha=.3)
    axes[0, 0].plot([], [], "C1", lw=.6, label="predicted chunks (every 4th)"); axes[0, 0].legend(fontsize=7)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _run_one(stem, fn):
    """Resumable rollout: a finished rollout's JSON is never recomputed."""
    done = stem.with_suffix(".json")
    if done.exists():
        return json.loads(done.read_text()), None
    result, rec = fn()
    return result, rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--policy", choices=("tinyrdt", "bc"), default="tinyrdt")
    p.add_argument("--dataset", default="artifacts/pickcube_smoke100_rgb160")
    p.add_argument("--episodes", type=int, nargs="+", help="dataset episode ids; default = checkpoint's training episodes")
    p.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 8], help="actions executed per replan")
    p.add_argument("--sampling-steps", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=240)
    p.add_argument("--policy-seed", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--skip-replay", action="store_true")
    p.add_argument("--no-media", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(2)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    ids = a.episodes or ck["train_episode_ids"]
    if a.policy == "tinyrdt":
        policy = TinyRDTPolicy(a.checkpoint, sampling_steps=a.sampling_steps, seed=a.policy_seed); horizon = policy.model.config.horizon
    else:
        policy = BCPolicy(a.checkpoint); horizon = policy.horizon
    env = SO101PickCubeEnv()
    for ep in ids:
        root = Path(a.dataset) / "episodes" / f"episode_{ep:06d}"
        meta = json.loads((root / "episode.json").read_text()); expert = np.load(root / "action.npy"); first_rgb = np.load(root / "rgb.npy")[0]
        if not a.skip_replay:
            # Control: exact open-loop replay of the recorded expert actions.
            stem = out / f"ep{ep:03d}_expert_replay"
            replay, rec = _run_one(stem, lambda: rollout(env, meta["seed"], lambda obs, step, rec: expert[step:step + 1] if step < len(expert) else expert[-1:], a.max_steps, first_rgb))
            if rec is not None:
                replay.update({"episode": ep, "seed": meta["seed"], "mode": "expert_replay", "expert_length": len(expert)})
                np.savez_compressed(stem.with_suffix(".npz"), **{k2: v for k2, v in rec.items() if k2 not in ("chunks", "chunk_steps")}, expert_actions=expert)
                save_json(stem.with_suffix(".json"), replay)
            print(json.dumps(replay), flush=True)
        for k in a.k:
            if not 1 <= k <= horizon: raise ValueError("K must lie in [1, H]")
            stem = out / f"ep{ep:03d}_k{k}"
            latency = []
            def act(obs, step, rec):
                started = time.perf_counter()
                cube = rec["cube"][-1] if policy.needs_cube_pose else None
                chunk = (policy.predict(obs["rgb"], _state(obs), cube) if cube is not None else policy.predict(obs["rgb"], _state(obs))).numpy()
                latency.append(time.perf_counter() - started)
                rec["chunks"].append(chunk); rec["chunk_steps"].append(step)
                return chunk[:k]
            def run():
                policy.reset(); return rollout(env, meta["seed"], act, a.max_steps, first_rgb)
            result, rec = _run_one(stem, run)
            if rec is not None:
                n = min(len(expert), len(rec["executed"]))
                result.update({"episode": ep, "seed": meta["seed"], "mode": f"policy_k{k}", "k": k, "policy": a.policy, "expert_length": len(expert),
                               "replans": len(rec["chunks"]), "mean_inference_ms": 1000 * float(np.mean(latency)),
                               "executed_vs_expert_mae_first_n": np.abs(rec["executed"][:n] - expert[:n]).mean(0).tolist(),
                               "cube_initial_xy": rec["cube"][0][:2].tolist()})
                np.savez_compressed(stem.with_suffix(".npz"), **{k2: v for k2, v in rec.items()}, expert_actions=expert)
                if not a.no_media:
                    imageio.mimsave(stem.with_suffix(".gif"), [np.kron(f, np.ones((2, 2, 1), dtype=np.uint8)) for f in rec["rgb"]], duration=50, loop=0)
                    save_plot(stem.with_suffix(".png"), rec, expert, f"{a.policy} episode {ep} (seed {meta['seed']}) K={k}: {'SUCCESS' if result['success'] else 'FAIL'} in {result['steps']} steps")
                save_json(stem.with_suffix(".json"), result)  # written last: marks the rollout complete
            print(json.dumps(result), flush=True)
    results = [json.loads(f.read_text()) for f in sorted(out.glob("ep*.json"))]
    table = {}
    for r in results:
        t = table.setdefault(r["mode"], {"n": 0, "success": 0, "steps": []}); t["n"] += 1; t["success"] += r["success"]
        if r["success"]: t["steps"].append(r["steps"])
    summary = {"checkpoint": a.checkpoint, "checkpoint_sha256": sha256(a.checkpoint), "policy": a.policy, "sampling_steps": a.sampling_steps,
               "policy_seed": a.policy_seed, "max_steps": a.max_steps, "results": results,
               "table": {m: {"success_rate": v["success"] / v["n"], "n": v["n"], "mean_success_steps": float(np.mean(v["steps"])) if v["steps"] else None} for m, v in table.items()}}
    save_json(out / "closed_loop_summary.json", summary)
    print(json.dumps(summary["table"], indent=2))


if __name__ == "__main__":
    main()
