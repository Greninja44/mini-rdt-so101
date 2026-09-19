"""Run/replay a single expert episode; GUI is optional and requires display support."""
from __future__ import annotations
import argparse
import time
from .env import PickCubeConfig, SO101PickCubeEnv
from .expert import PickCubeExpert

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, default=0); p.add_argument("--gui", action="store_true"); p.add_argument("--width", type=int, default=160); p.add_argument("--height", type=int, default=120); p.add_argument("--hold", type=float, default=3.0, help="seconds to keep GUI open after completion")
    args = p.parse_args()
    env = SO101PickCubeEnv(PickCubeConfig(camera_width=args.width, camera_height=args.height, physics="v1"), render_mode="rgb_array")
    if args.gui:
        import mujoco.viewer
        env._viewer = mujoco.viewer.launch_passive(env.model, env.data)
    obs, info = env.reset(seed=args.seed); expert = PickCubeExpert(env, verbose=True); expert.reset()
    while not expert.done:
        obs, reward, terminated, truncated, info = env.step(expert.action()); expert.observe(info)
        if env._viewer is not None:
            env._viewer.sync()
            time.sleep(1 / env.config.control_frequency)
        if terminated or truncated: break
    print({"seed": args.seed, "state": expert.state.value, "transitions": expert.transitions, "failure": expert.failure_category, "steps": info["step"], "success": info["task_success"]})
    if env._viewer is not None:
        deadline = time.monotonic() + args.hold
        while time.monotonic() < deadline and env._viewer.is_running():
            env._viewer.sync(); time.sleep(0.05)
    env.close()

if __name__ == "__main__": main()
