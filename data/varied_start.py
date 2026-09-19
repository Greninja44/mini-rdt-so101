"""Varied initial arm poses (Phase 9): break the identical-HOME-start confound.

apply_start mirrors SO101PickCubeEnv.reset: set the arm (and its position
command) to q0 with the gripper open, then settle 20 physics steps, exactly
like reset's own settle. The cube placement from the reset seed is unchanged.
"""
from __future__ import annotations

import mujoco
import numpy as np

from simulation.controllers import action_to_ctrl

START_HALF_RANGE = np.array([0.15, 0.15, 0.15, 0.15, 0.0])  # rad; wrist_roll fixed


def sample_start(env, rng: np.random.Generator) -> np.ndarray:
    lo, hi = env._lo[:5] + 0.05, env._hi[:5] - 0.05
    return np.clip(env.home_joint_pos + rng.uniform(-START_HALF_RANGE, START_HALF_RANGE), lo, hi)


def apply_start(env, q0: np.ndarray):
    q0 = np.asarray(q0, dtype=np.float64)
    env.data.qpos[env._joint_qposadr[:5]] = q0
    env.data.qvel[:] = 0.0
    env.data.ctrl[:] = action_to_ctrl(env.model, np.r_[q0, 1.0])
    mujoco.mj_forward(env.model, env.data)
    for _ in range(20): mujoco.mj_step(env.model, env.data)
    env._step_count = env._success_streak = 0; env._ever_grasped = False
    return env._observation(), env._info()


def start_for(env, seed: int, episode: int, index: int = 0) -> np.ndarray:
    """Deterministic start pose for (start seed, dataset episode, index)."""
    return sample_start(env, np.random.default_rng([seed, episode, index]))
