"""Collect the physics-v2 PickCube dataset with the validated side-pinch expert.

Writes a NEW dataset (default artifacts/pickcube_physics_v2_rgb160). The physics-v1 dataset is never touched.
It uses the same Phase-1 episode format plus per-step contact diagnostics and full provenance metadata.
Resumable and atomic per episode; failed seeds are logged in failed.jsonl.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import mujoco
import numpy as np

from data.collect import ACTION_REPRESENTATION
from data.dataset import EpisodeRecorder, make_episode_metadata
from simulation.env import PickCubeConfig, SO101PickCubeEnv
from simulation.expert_v2 import EXPERT_V2_VERSION, PickCubeExpertV2

SUCCESS_DEFINITION = ("physics-v2: 5 consecutive steps with a valid side pinch (both pads touching the cube, mean pad normals within 30 deg "
                      "of horizontal and >=120 deg apart, no vertical pad contact), cube z >= 0.10 m, and no robot/pad-table > 1.0 mm, "
                      "cube-table > 1.0 mm or pad-cube > 1.5 mm penetration at any step (docs/research/physics_v2_spec.md)")


def model_fingerprint(env) -> dict:
    """Hash of the effective collision / contact model (not just source files)."""
    m = env.model
    arrays = [m.geom_contype, m.geom_conaffinity, m.geom_solref, m.geom_solimp, m.geom_pos, m.geom_size, m.geom_friction, m.actuator_forcerange, m.jnt_range]
    h = hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes() for a in arrays)).hexdigest()
    groups = {}
    for g in range(m.ngeom):
        key = f"contype={m.geom_contype[g]} conaffinity={m.geom_conaffinity[g]}"
        groups.setdefault(key, []).append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])}#{g}")
    return {"mujoco_version": mujoco.__version__, "effective_model_sha256": h, "collision_groups": groups,
            "contact_bits": "1 cube-table, 2 pad-cube, 4 robot/pad-table", "robot_collision_geoms": len(env._robot_collision_geoms)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="artifacts/pickcube_physics_v2_rgb160")
    p.add_argument("--start", type=int, default=0); p.add_argument("--count", type=int, default=100); p.add_argument("--seed-base", type=int, default=3000)
    p.add_argument("--positions", help="generalization split JSON (data/generalization_split.py): collect its --split rows at fixed cube_xy")
    p.add_argument("--split", choices=("validation", "test", "train", "eval")); p.add_argument("--dataset-version", default="pickcube-physics-v2-1")
    p.add_argument("--workspace-margin", type=float, default=0.0, help="widen ONLY the reset bounds check (extrapolation positions); physics unchanged")
    a = p.parse_args(); out = Path(a.output); (out / "episodes").mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path(a.positions).read_text())[a.split] if a.positions else None
    base = PickCubeConfig(physics="v2"); m = a.workspace_margin
    env = SO101PickCubeEnv(PickCubeConfig(physics="v2", workspace_x=(base.workspace_x[0] - m, base.workspace_x[1] + m), workspace_y=(base.workspace_y[0] - m, base.workspace_y[1] + m)))
    fp = model_fingerprint(env)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    failed = out / "failed.jsonl"
    for index in range(a.start, min(a.start + a.count, len(rows)) if rows else a.start + a.count):
        final = out / "episodes" / f"episode_{index:06d}"; seed = rows[index]["seed"] if rows else a.seed_base + index
        if final.exists() or (failed.exists() and any(json.loads(l)["index"] == index for l in failed.read_text().splitlines())): continue
        obs, info = env.reset(seed=seed, options={"cube_xy": rows[index]["cube_xy"]} if rows else None); ex = PickCubeExpertV2(env); ex.reset()
        meta = make_episode_metadata(env, seed, ACTION_REPRESENTATION, expert=ex)
        meta.update({"physics_version": "v2", "expert_version": EXPERT_V2_VERSION, "dataset_version": a.dataset_version, "git_commit": commit,
                     "model_fingerprint": fp, "success_definition": SUCCESS_DEFINITION,
                     "cube_randomization": {"workspace_x": list(env.config.workspace_x), "workspace_y": list(env.config.workspace_y), "yaw": 0.0, "size_m": env.config.cube_size},
                     "expert_plan": {"yaw": ex.plan.get("yaw"), "grasp_dz_m": ex.plan.get("grasp_dz"), "side_overlap_m": ex.plan.get("side_overlap")}})
        if rows: meta.update({"cube_xy": rows[index]["cube_xy"], "generalization": rows[index], "split_file": a.positions, "split": a.split})
        tmp = out / f"tmp_{os.getpid()}"; shutil.rmtree(tmp, ignore_errors=True)
        rec = EpisodeRecorder(tmp, index, meta); diag = {k: [] for k in ("side_pinch", "robot_table_penetration", "cube_table_penetration", "pad_cube_penetration", "robot_table_contacts", "pad_normals")}
        while not ex.done and info["step"] < env.config.max_episode_steps:
            action = ex.action(); rec.append(obs, action, info, ex.state.value, bool(info["task_success"]))
            obs, _, _, _, info = env.step(action); ex.observe(info); c = info["contacts"]
            for k in ("side_pinch", "robot_table_penetration", "cube_table_penetration", "pad_cube_penetration", "robot_table_contacts"): diag[k].append(c[k])
            diag["pad_normals"].append([n if n is not None else [np.nan] * 3 for n in c["pad_normals"]])
        ok = bool(info.get("success")) and ex.state.value == "SUCCESS"
        if ok:
            rec.set_last_transition_outcome(success=True); rec.save(success=True)
            ep_dir = tmp / "episodes" / f"episode_{index:06d}"
            for k, v in diag.items(): np.save(ep_dir / f"contact_{k}.npy", np.asarray(v, dtype=np.float32 if k != "side_pinch" else bool))
            os.replace(ep_dir, final)
        else:
            with failed.open("a") as f: f.write(json.dumps({"index": index, "seed": seed, "failure": ex.failure_category, "invalid_reason": info.get("invalid_reason"),
                                                            "expert_state": ex.state.value, "cube_xy": rows[index]["cube_xy"] if rows else None}) + "\n")
        shutil.rmtree(tmp, ignore_errors=True)
        print(json.dumps({"index": index, "seed": seed, "success": ok, "steps": info.get("step", 0), "max_cube_table_mm": 1000 * info.get("max_cube_table_penetration", 0.0)}), flush=True)
    (out / "collection_summary.json").write_text(json.dumps({"physics_version": "v2", "expert_version": EXPERT_V2_VERSION, "model_fingerprint": fp,
        "episodes": len(list((out / "episodes").glob("episode_*"))), "failed": len(failed.read_text().splitlines()) if failed.exists() else 0,
        "seed_base": a.seed_base, "positions": a.positions, "split": a.split, "workspace_margin_m": a.workspace_margin, "success_definition": SUCCESS_DEFINITION}, indent=1) + "\n")


if __name__ == "__main__":
    main()
