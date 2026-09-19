"""Phase 9 dataset VS10: CLEAN10 + demos from randomised start poses on the SAME 10 cube seeds.

Each new demo is a complete rollout of the data-generating (legacy) expert
from its own start (state-feedback IK in every phase, so it starts directly
in MOVE_ABOVE_OBJECT; HOME would first drive back to the shared pose).
Standard Phase-1 episode format (data/dataset.py), so the unchanged loaders
work. CLEAN10 originals are symlinked under their ORIGINAL episode indices
(episode.json episode_index must match the directory); new demos start at 100. Resumable, atomic.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np

from data.collect import ACTION_REPRESENTATION
from data.dataset import EpisodeRecorder, make_episode_metadata
from data.varied_start import START_HALF_RANGE, apply_start, start_for
from simulation.env import SO101PickCubeEnv
from simulation.expert import ExpertState
from simulation.legacy_expert import EXPERT_VERSION, LegacyPickCubeExpert

CLEAN10 = [0, 2, 3, 4, 5, 6, 7, 8, 9, 11]
SOURCE = Path("artifacts/pickcube_smoke100_rgb160")
START_SEED = 9000  # training starts; evaluation uses a different seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="artifacts/varied_start10")
    p.add_argument("--per-seed", type=int, default=4)
    p.add_argument("--episodes", type=int, nargs="+", default=CLEAN10)
    a = p.parse_args()
    out = Path(a.output); (out / "episodes").mkdir(parents=True, exist_ok=True)
    for ep in CLEAN10:  # originals, unchanged, as symlinks under their own index
        link = out / "episodes" / f"episode_{ep:06d}"
        if not link.exists(): link.symlink_to((SOURCE / "episodes" / f"episode_{ep:06d}").resolve())
    env = SO101PickCubeEnv(); failed = out / "failed.jsonl"
    for ep in a.episodes:
        seed = json.loads((SOURCE / "episodes" / f"episode_{ep:06d}" / "episode.json").read_text())["seed"]
        for j in range(a.per_seed):
            index = 100 + CLEAN10.index(ep) * a.per_seed + j
            final = out / "episodes" / f"episode_{index:06d}"
            if final.exists() or (failed.exists() and any(json.loads(l)["index"] == index for l in failed.read_text().splitlines())): continue
            obs, info = env.reset(seed=seed); q0 = start_for(env, START_SEED, ep, j); obs, info = apply_start(env, q0)
            expert = LegacyPickCubeExpert(env); expert.reset(); expert._transition(ExpertState.MOVE_ABOVE_OBJECT)
            meta = make_episode_metadata(env, seed, ACTION_REPRESENTATION, expert=expert)
            meta.update({"expert_version": EXPERT_VERSION, "source_episode": ep, "start_seed": START_SEED, "start_index": j,
                         "start_joint_pos": q0.tolist(), "start_half_range_rad": START_HALF_RANGE.tolist(), "dataset_version": "varied_start10-v1"})
            tmp = out / f"tmp_{os.getpid()}"; shutil.rmtree(tmp, ignore_errors=True)  # per-process: two workers run in parallel
            rec = EpisodeRecorder(tmp, index, meta)
            while not expert.done and info["step"] < env.config.max_episode_steps:
                action = expert.action()
                rec.append(obs, action, info, expert.state.value, bool(info["task_success"]))
                obs, _, _, _, info = env.step(action); expert.observe(info)
            ok = expert.state.value == "SUCCESS"
            if ok:
                rec.save(success=True); os.replace(tmp / "episodes" / f"episode_{index:06d}", final)
            else:
                with failed.open("a") as f: f.write(json.dumps({"index": index, "source_episode": ep, "seed": seed, "start": q0.tolist(), "failure": expert.failure_category}) + "\n")
            shutil.rmtree(tmp, ignore_errors=True)
            print(json.dumps({"index": index, "ep": ep, "start_dev": (q0 - env.home_joint_pos).round(3).tolist(), "success": ok, "steps": info["step"]}), flush=True)


if __name__ == "__main__":
    main()
