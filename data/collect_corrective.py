"""Collect corrective datasets on the CLEAN10 seeds (see docs/research/research_log.md, Phase 5).

  perturb: legacy expert; at t0 a command offset on one joint for 3 steps;
           the expert then continues from the ACTUAL state.
  dagger : TinyRDT drives (receding horizon, K); the legacy expert shadows
           it and labels every visited state; the expert takes over past the
           pre-registered trigger.
Every recorded frame stores the observation of the actual state and the
counterfactual 16-step expert chunk from that exact state (data/corrective.py).
Each episode is written atomically (tmp dir -> rename) and skipped on resume;
discarded episodes are logged to discarded.jsonl with their seeds.
Only CLEAN10 (training) seeds are accepted: held-out seeds cannot leak in.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
from dataclasses import replace

import numpy as np

from data.corrective import expert_chunk
from simulation.env import SO101PickCubeEnv
from simulation.legacy_expert import EXPERT_VERSION, LEGACY_JOINT_LIMIT_MARGIN, LegacyPickCubeExpert

CLEAN10 = [0, 2, 3, 4, 5, 6, 7, 8, 9, 11]
DATASET = Path("artifacts/pickcube_smoke100_rgb160")
HORIZON = 16
PERTURB_JOINTS = {"shoulder_pan": 0, "shoulder_lift": 1, "elbow_flex": 2}
TAKEOVER_JOINT_DIST = 0.10


def _sha(path):
    h = hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()


def nominal(ep):
    root = DATASET / "episodes" / f"episode_{ep:06d}"
    meta = json.loads((root / "episode.json").read_text()); states = np.load(root / "expert_state.npy")
    replay = Path("artifacts/closed_loop_v2/tinyrdt_ema") / f"ep{ep:03d}_expert_replay.npz"  # physically identical to the demo
    gc = np.load(replay)["grasp_center"] if replay.exists() else None
    return {"seed": meta["seed"], "q": np.load(root / "joint_pos.npy"), "states": states, "grasp_center": gc,
            "descend": int(np.argmax(states == "DESCEND")), "lift": int(np.argmax(states == "LIFT"))}


class Recorder:
    def __init__(self): self.f = {k: [] for k in ("rgb", "joint_pos", "gripper", "cube_pose", "grasp_center", "executed_action", "label_chunk", "label_valid", "expert_state", "controller", "t")}
    def add(self, env, obs, chunk, n, expert, controller, t):
        self.f["rgb"].append(obs["rgb"]); self.f["joint_pos"].append(obs["joint_pos"]); self.f["gripper"].append(obs["gripper"])
        self.f["cube_pose"].append(env.cube_pose); self.f["grasp_center"].append(env.grasp_center_position)
        self.f["label_chunk"].append(chunk); v = np.zeros(HORIZON, bool); v[:n] = True; self.f["label_valid"].append(v)
        self.f["expert_state"].append(expert.state.value); self.f["controller"].append(controller); self.f["t"].append(t)
    def act(self, a): self.f["executed_action"].append(np.asarray(a, np.float32))
    def save(self, final: Path, meta: dict):
        assert len(self.f["executed_action"]) == len(self.f["t"])
        tmp = final.with_name(final.name + ".tmp"); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
        for k, v in self.f.items(): np.save(tmp / f"{k}.npy", np.asarray(v))
        (tmp / "episode.json").write_text(json.dumps(meta, indent=1) + "\n")
        os.replace(tmp, final)


def render_on(env, on: bool):
    env.config = replace(env.config, render_observations=on)


def run_perturb(env, ep, t0, joint, delta, max_steps=150):
    nom = nominal(ep); render_on(env, False)
    obs, info = env.reset(seed=nom["seed"]); expert = LegacyPickCubeExpert(env); expert.reset(); rec = Recorder()
    lo, hi = env._lo[:5] + LEGACY_JOINT_LIMIT_MARGIN, env._hi[:5] - LEGACY_JOINT_LIMIT_MARGIN
    t = 0; peak = {"joint_dist_rad": 0.0, "grasp_center_path_mm": 0.0}; grasped = False
    while not expert.done and t < max_steps:
        if t == t0:
            render_on(env, True); obs = dict(obs, rgb=env.render())
        if t >= t0:
            chunk, n, ok = expert_chunk(env, expert, HORIZON)
            a = expert.action()
            controller = "expert"
            if t < t0 + 3:
                a = a.copy(); a[joint] = np.clip(a[joint] + delta, lo[joint], hi[joint]); controller = "perturbed"
            rec.add(env, obs, chunk, n, expert, controller, t); rec.act(a)
            peak["joint_dist_rad"] = max(peak["joint_dist_rad"], float(np.linalg.norm(nom["q"] - obs["joint_pos"], axis=1).min()))
            if nom["grasp_center"] is not None:
                peak["grasp_center_path_mm"] = max(peak["grasp_center_path_mm"], 1000 * float(np.linalg.norm(nom["grasp_center"] - env.grasp_center_position, axis=1).min()))
        else:
            a = expert.action()
        obs, _, _, _, info = env.step(a); expert.observe(info); grasped |= bool(info["grasped"]); t += 1
    ok = expert.state.value == "SUCCESS"
    return rec, {"expert_outcome": expert.state.value, "failure_category": expert.failure_category, "recovered": ok, "grasped": grasped,
                 "steps": t, "peak_time_free_joint_dist_rad": peak["joint_dist_rad"], "peak_grasp_center_path_dev_mm": peak["grasp_center_path_mm"]}, ok


def run_dagger(env, ep, policy, k, max_steps=150):
    nom = nominal(ep); render_on(env, True)
    obs, info = env.reset(seed=nom["seed"]); expert = LegacyPickCubeExpert(env); expert.reset(); policy.reset(); rec = Recorder()
    t = 0; takeover = None; reason = None; plan = []; policy_steps = 0; max_dist = 0.0; grasped = False
    while not expert.done and t < max_steps:
        chunk, n, ok = expert_chunk(env, expert, HORIZON)
        state = np.r_[obs["joint_pos"], obs["gripper"]].astype(np.float32)
        dist = float(np.linalg.norm(nom["q"] - obs["joint_pos"], axis=1).min())
        if takeover is None:
            max_dist = max(max_dist, dist)
            if not plan: plan = list(policy.predict(obs["rgb"], state).numpy()[:k])
            proposed = plan[0]
            expert_open = expert.state.value in ("HOME", "MOVE_ABOVE_OBJECT", "DESCEND")
            if dist > TAKEOVER_JOINT_DIST: takeover, reason = t, f"joint_dist {dist:.3f} > {TAKEOVER_JOINT_DIST}"
            elif proposed[5] < .5 and expert_open: takeover, reason = t, f"policy close while expert open ({expert.state.value})"
        controller = "policy" if takeover is None else "expert"
        rec.add(env, obs, chunk, n, expert, controller, t)
        if takeover is None:
            a = plan.pop(0); policy_steps += 1
        else:
            a = expert.action()
        rec.act(a)
        obs, _, _, _, info = env.step(a); expert.observe(info); grasped |= bool(info["grasped"]); t += 1
        if takeover is None and info["success"]: break
    ok = expert.state.value == "SUCCESS" or (takeover is None and bool(info["success"]))
    return rec, {"expert_outcome": expert.state.value, "takeover_step": takeover, "takeover_reason": reason, "policy_steps": policy_steps,
                 "max_joint_dist_before_takeover_rad": max_dist, "grasped": grasped, "steps": t, "success": ok}, ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("perturb", "dagger"), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--episodes", type=int, nargs="+", default=CLEAN10)
    p.add_argument("--checkpoint", default="artifacts/research_audit/cosine_x0_hold_ema/ema_last.pt")
    p.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 8])
    a = p.parse_args()
    if not set(a.episodes) <= set(CLEAN10): raise ValueError("corrective collection is restricted to CLEAN10 training seeds")
    out = Path(a.output); (out / "episodes").mkdir(parents=True, exist_ok=True)
    env = SO101PickCubeEnv(); policy = None
    common = {"dataset_version": f"corrective-{a.mode}-v1", "source_dataset": str(DATASET), "expert_version": EXPERT_VERSION,
              "horizon": HORIZON, "control_frequency": env.config.control_frequency,
              "camera": {"name": "external_rgb", "width": env.config.camera_width, "height": env.config.camera_height},
              "action_representation": "[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll] radians + gripper_open normalized [0,1]",
              "label_contract": "obs of actual state -> counterfactual 16-step legacy-expert chunk from that exact state; hold-padded, label_valid marks prefix"}
    jobs = []
    for i, ep in enumerate(a.episodes):
        if a.mode == "perturb":
            nom = nominal(ep); rng = np.random.default_rng(10_000 + ep)
            for t0_name, t0 in (("approach_mid", 10), ("approach_late", nom["descend"] - 2), ("descent", nom["descend"]), ("early_lift", nom["lift"] + 1)):
                for jname, j in PERTURB_JOINTS.items():
                    mag = float(rng.choice([.01, .02, .03, .04], p=[.27, .27, .26, .20])); sign = float(rng.choice([-1., 1.]))
                    jobs.append((ep, {"t0_name": t0_name, "t0": int(t0), "joint": jname, "joint_index": j, "magnitude_rad": mag, "sign": sign, "duration_steps": 3}))
        else:
            for k in a.k: jobs.append((ep, {"k": k, "policy_seed": 0, "takeover_joint_dist_rad": TAKEOVER_JOINT_DIST}))
    if a.mode == "dagger":
        from training.policy import TinyRDTPolicy
        policy = TinyRDTPolicy(a.checkpoint, seed=0); common.update(source_checkpoint=a.checkpoint, source_checkpoint_sha256=_sha(a.checkpoint))
    for index, (ep, cfg) in enumerate(jobs):
        final = out / "episodes" / f"episode_{index:06d}"
        done_log = out / "discarded.jsonl"
        if final.exists() or (done_log.exists() and any(json.loads(l)["index"] == index for l in done_log.read_text().splitlines())): continue
        nom = nominal(ep)
        if a.mode == "perturb":
            rec, outcome, keep = run_perturb(env, ep, cfg["t0"], cfg["joint_index"], cfg["sign"] * cfg["magnitude_rad"])
        else:
            policy.seed = cfg["policy_seed"]
            rec, outcome, keep = run_dagger(env, ep, policy, cfg["k"])
        meta = {**common, "index": index, "source_episode": ep, "source_seed": nom["seed"], "config": cfg, "outcome": outcome,
                "frames": len(rec.f["t"]), "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "success": keep}
        if keep and len(rec.f["t"]):
            rec.save(final, meta)
        else:
            with done_log.open("a") as f: f.write(json.dumps(meta) + "\n")
        print(json.dumps({"index": index, "ep": ep, **cfg, **outcome, "frames": len(rec.f["t"]), "kept": keep}), flush=True)


if __name__ == "__main__":
    main()
