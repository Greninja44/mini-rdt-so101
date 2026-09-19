"""Regression benchmark that saves all failed randomized seeds."""
from __future__ import annotations
import argparse, collections, json, time
from pathlib import Path
from simulation.env import PickCubeConfig, SO101PickCubeEnv
from simulation.expert import PickCubeExpert

def benchmark(episodes: int = 100, seed: int = 0, output: str | Path = "artifacts/expert_benchmark.json") -> dict:
    started = time.monotonic(); rows = []; failures = []
    # The expert does not consume pixels. Skip expensive WSL software rendering
    # only for this control benchmark; RGB is exercised by environment/record tests.
    env = SO101PickCubeEnv(PickCubeConfig(render_observations=False, physics="v1"))  # benchmarks the v1 (INVALID) expert
    for index in range(episodes):
        s = seed + index; _, info = env.reset(seed=s); expert = PickCubeExpert(env); expert.reset()
        while not expert.done:
            _, _, terminated, truncated, info = env.step(expert.action()); expert.observe(info)
            if terminated or truncated: break
        row = {"seed": s, "success": bool(info["task_success"]), "steps": info["step"], "completion_time_s": info["step"] / env.config.control_frequency, "failure": expert.failure_category}
        rows.append(row)
        if not row["success"]: failures.append(row)
    env.close(); success = [r for r in rows if r["success"]]
    report = {"episodes": episodes, "seed_start": seed, "success_rate": len(success) / episodes, "failure_count": len(failures), "average_episode_length": sum(r["steps"] for r in rows) / episodes, "average_completion_time_s": (sum(r["completion_time_s"] for r in success) / len(success)) if success else None, "failure_categories": dict(collections.Counter(r["failure"] or "unknown" for r in failures)), "failed_seeds": [r["seed"] for r in failures], "elapsed_wall_s": time.monotonic() - started}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps({"report": report, "episodes": rows}, indent=2) + "\n")
    return report

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--episodes", type=int, default=100); p.add_argument("--seed", type=int, default=0); p.add_argument("--output", default="artifacts/expert_benchmark.json"); a = p.parse_args(); print(json.dumps(benchmark(a.episodes, a.seed, a.output), indent=2))
if __name__ == "__main__": main()
