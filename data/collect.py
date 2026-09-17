"""Collect a small number of verified expert episodes; not a bulk data generator."""
from __future__ import annotations
import argparse, json
import os
from pathlib import Path
import subprocess
import sys
from simulation.env import PickCubeConfig, SO101PickCubeEnv
from simulation.expert import PickCubeExpert
from .dataset import EpisodeRecorder, make_episode_metadata

ACTION_REPRESENTATION = "[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll] radians + gripper_open normalized [0,1]"

def collect(root: str | Path, episodes: int, seed: int, width: int = 160, height: int = 120, episode_index_offset: int = 0) -> list[dict]:
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    results = []
    for index in range(episodes):
        episode_seed = seed + index
        env = SO101PickCubeEnv(PickCubeConfig(camera_width=width, camera_height=height))
        obs, info = env.reset(seed=episode_seed); expert = PickCubeExpert(env); expert.reset()
        recorder = EpisodeRecorder(root, episode_index_offset + index, make_episode_metadata(env, episode_seed, ACTION_REPRESENTATION, expert=expert))
        while not expert.done:
            action = expert.action()
            # Record observation_t before it causes the transition to t+1.
            recorder.append(obs, action, info, expert.state.value, bool(info["task_success"]))
            obs, _, terminated, truncated, info = env.step(action); expert.observe(info)
            recorder.set_last_transition_outcome(success=bool(info["task_success"]))
            if terminated or truncated: break
        if info["task_success"]:
            recorder.save(success=True)
        results.append({"episode": episode_index_offset + index, "seed": episode_seed, "success": bool(info["task_success"]), "failure": expert.failure_category, "steps": info["step"]})
        env.close()
    (root / f"collection_summary_{episode_index_offset:06d}.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def collect_parallel(root: str | Path, episodes: int, seed: int, width: int, height: int, workers: int) -> list[dict]:
    """Run disjoint deterministic ranges under a foreground coordinator.

    This avoids shell-background jobs, which are not durable in several CI and
    sandbox environments. Each child renders genuine observations and writes a
    non-overlapping episode-index/seed range.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[list[int], subprocess.Popen[str]]] = []
    environment = os.environ.copy()
    environment.setdefault("MUJOCO_GL", "egl")
    offset = 0
    base, remainder = divmod(episodes, workers)
    for worker in range(workers):
        count = base + int(worker < remainder)
        if count == 0:
            continue
        worker_seed = seed + offset
        command = [
            sys.executable, "-m", "data.collect", "--output", str(root),
            "--episodes", str(count), "--seed", str(worker_seed),
            "--width", str(width), "--height", str(height),
            "--episode-index-offset", str(offset),
        ]
        log = (root / f"worker_{offset:06d}.log").open("w", encoding="utf-8")
        processes.append(([offset, count], subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True, env=environment)))
        log.close()
        offset += count
    failures = []
    for (offset, count), process in processes:
        code = process.wait()
        if code != 0:
            failures.append({"offset": offset, "episodes": count, "exit_code": code})
    if failures:
        raise RuntimeError(f"collection workers failed: {failures}")
    all_results: list[dict] = []
    for summary in sorted(root.glob("collection_summary_*.json")):
        all_results.extend(json.loads(summary.read_text()))
    if len(all_results) != episodes:
        raise RuntimeError(f"expected {episodes} collection results, found {len(all_results)}")
    (root / "collection_summary.json").write_text(json.dumps(sorted(all_results, key=lambda r: r["episode"]), indent=2) + "\n")
    return sorted(all_results, key=lambda r: r["episode"])

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--output", default="dataset_phase1"); p.add_argument("--episodes", type=int, default=3); p.add_argument("--seed", type=int, default=0); p.add_argument("--width", type=int, default=160); p.add_argument("--height", type=int, default=120); p.add_argument("--episode-index-offset", type=int, default=0); p.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    if a.workers == 1:
        results = collect(a.output, a.episodes, a.seed, a.width, a.height, a.episode_index_offset)
    elif a.episode_index_offset:
        raise ValueError("--workers cannot be combined with --episode-index-offset")
    else:
        results = collect_parallel(a.output, a.episodes, a.seed, a.width, a.height, a.workers)
    print(json.dumps(results, indent=2))
if __name__ == "__main__": main()
