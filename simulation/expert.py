"""Deterministic position-IK PickCube demonstrator using the public action API."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import mujoco
import numpy as np

from .controllers import position_ik_step
from .env import SO101PickCubeEnv


class ExpertState(str, Enum):
    HOME = "HOME"
    MOVE_ABOVE_OBJECT = "MOVE_ABOVE_OBJECT"
    DESCEND = "DESCEND"
    CLOSE_GRIPPER = "CLOSE_GRIPPER"
    LIFT = "LIFT"
    HOLD = "HOLD"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ExpertConfig:
    # A 0.10 m staging target is reachable but, from the calibrated home pose,
    # the local DLS solution can swing the wrist below the table before rising.
    above_offset: float = 0.18
    overhead_waypoint_steps: int = 30
    # Grasp center is the physical fingertip midpoint, unlike the old tool-tip
    # site, so it is aligned with the cube centre rather than held above it.
    grasp_offset: float = 0.001
    lift_height: float = 0.18
    position_tolerance: float = 0.018
    grasp_position_tolerance: float = 0.004
    home_timeout: int = 80
    move_timeout: int = 100
    descend_timeout: int = 90
    close_timeout: int = 50
    grasp_hold_steps: int = 5
    lift_timeout: int = 100


class PickCubeExpert:
    """State-machine expert. Every command is a six-element environment action."""
    def __init__(self, env: SO101PickCubeEnv, config: ExpertConfig | None = None, verbose: bool = False):
        self.env, self.config, self.verbose = env, config or ExpertConfig(), verbose
        self.state = ExpertState.HOME
        self.steps_in_state = 0
        self.transitions: list[str] = [self.state.value]
        self.failure_category: str | None = None
        self._grasp_streak = 0
        self._cube_reference = np.zeros(3)
        self._overhead_q = np.zeros(5)

    def reset(self) -> None:
        self.state = ExpertState.HOME; self.steps_in_state = 0
        self.transitions = [self.state.value]; self.failure_category = None
        self._grasp_streak = 0
        self._cube_reference = self.env.cube_pose[:3].copy()
        self._overhead_q = self._plan_overhead_joint_pose()

    def action(self) -> np.ndarray:
        target = self._target_xyz()
        if self.state == ExpertState.MOVE_ABOVE_OBJECT:
            # Execute a precomputed, collision-checked joint interpolation. A
            # fresh local DLS step from HOME can swing the wrist through the
            # tabletop even though its endpoint is safely overhead.
            alpha = min(1.0, (self.steps_in_state + 1) / self.config.overhead_waypoint_steps)
            arm = self.env.home_joint_pos + alpha * (self._overhead_q - self.env.home_joint_pos)
        else:
            arm = position_ik_step(self.env.model, self.env.data, self.env._grasp_site_id, target)
        opening = 0.0 if self.state in (ExpertState.CLOSE_GRIPPER, ExpertState.LIFT, ExpertState.HOLD, ExpertState.SUCCESS) else 1.0
        if self.state == ExpertState.HOME:
            # Joint home is a fixed, separately verified calibrated configuration.
            arm = self.env.home_joint_pos.copy()
        return np.r_[arm, opening].astype(np.float32)

    def observe(self, info: dict) -> None:
        self.steps_in_state += 1
        if self.state == ExpertState.HOME:
            q = self.env.data.qpos[self.env._joint_qposadr[:5]]
            if np.max(np.abs(q - self.env.home_joint_pos)) < 0.04: self._transition(ExpertState.MOVE_ABOVE_OBJECT)
            elif self.steps_in_state > self.config.home_timeout: self._fail("HOME_TIMEOUT")
        elif self.state == ExpertState.MOVE_ABOVE_OBJECT:
            if self._above_ready(): self._transition(ExpertState.DESCEND)
            elif self.steps_in_state > self.config.move_timeout: self._fail("APPROACH_TIMEOUT")
        elif self.state == ExpertState.DESCEND:
            if self._at_grasp_target(): self._transition(ExpertState.CLOSE_GRIPPER)
            elif self.steps_in_state > self.config.descend_timeout: self._fail("DESCENT_TIMEOUT")
        elif self.state == ExpertState.CLOSE_GRIPPER:
            self._grasp_streak = self._grasp_streak + 1 if info["grasped"] else 0
            if self._grasp_streak >= self.config.grasp_hold_steps: self._transition(ExpertState.LIFT)
            elif self.steps_in_state > self.config.close_timeout: self._fail("GRASP_FAILED")
        elif self.state == ExpertState.LIFT:
            if info["elevated"] and info["grasped"]: self._transition(ExpertState.HOLD)
            elif self.steps_in_state > self.config.lift_timeout: self._fail("LIFT_FAILED")
        elif self.state == ExpertState.HOLD:
            if info["success"]: self._transition(ExpertState.SUCCESS)
            elif not info["grasped"]: self._fail("OBJECT_DROPPED")
            elif self.steps_in_state > self.config.lift_timeout: self._fail("HOLD_TIMEOUT")

    @property
    def done(self) -> bool: return self.state in (ExpertState.SUCCESS, ExpertState.FAILED)

    def _target_xyz(self) -> np.ndarray:
        cube = self._cube_reference
        if self.state == ExpertState.HOME: return self.env.grasp_center_position
        if self.state == ExpertState.MOVE_ABOVE_OBJECT: return cube + np.array([0, 0, self.config.above_offset])
        if self.state in (ExpertState.DESCEND, ExpertState.CLOSE_GRIPPER): return cube + np.array([0, 0, self.config.grasp_offset])
        if self.state in (ExpertState.LIFT, ExpertState.HOLD, ExpertState.SUCCESS): return np.array([cube[0], cube[1], self.config.lift_height])
        return self.env.data.site_xpos[self.env._ee_site_id]

    def _plan_overhead_joint_pose(self) -> np.ndarray:
        """Solve overhead IK on a scratch qpos, without executing a teleport."""
        saved_qpos, saved_qvel = self.env.data.qpos.copy(), self.env.data.qvel.copy()
        target = self._cube_reference + np.array([0.0, 0.0, self.config.above_offset])
        q = saved_qpos[self.env._joint_qposadr[:5]].copy()
        for _ in range(120):
            q = position_ik_step(self.env.model, self.env.data, self.env._grasp_site_id, target)
            self.env.data.qpos[self.env._joint_qposadr[:5]] = q
            mujoco.mj_forward(self.env.model, self.env.data)
        self.env.data.qpos[:] = saved_qpos; self.env.data.qvel[:] = saved_qvel
        mujoco.mj_forward(self.env.model, self.env.data)
        return q

    def _at_target(self) -> bool:
        return np.linalg.norm(self.env.data.site_xpos[self.env._ee_site_id] - self._target_xyz()) < self.config.position_tolerance

    def _at_grasp_target(self) -> bool:
        return np.linalg.norm(self.env.grasp_center_position - self._target_xyz()) < self.config.grasp_position_tolerance

    def _above_ready(self) -> bool:
        """Accept a safe overhead staging pose near a 5-DOF IK singularity.

        Full XYZ equality is not required here: the next, low Cartesian target
        resolves the remaining horizontal error. This keeps the approach above
        the cube rather than declaring an unreachable high pose a failure.
        """
        center, cube = self.env.grasp_center_position, self.env.cube_pose[:3]
        return bool(np.linalg.norm(center[:2] - cube[:2]) < 0.012 and center[2] > cube[2] + self.config.above_offset - 0.025)

    def _transition(self, state: ExpertState) -> None:
        if state == self.state: return
        self.state = state; self.steps_in_state = 0; self.transitions.append(state.value)
        if self.verbose: print(f"expert -> {state.value}")

    def _fail(self, category: str) -> None:
        self.failure_category = category; self._transition(ExpertState.FAILED)
