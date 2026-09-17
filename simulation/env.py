"""Gymnasium-style, deterministic MuJoCo PickCube environment for SO-101."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tempfile

import mujoco
import numpy as np

try:  # Gymnasium is optional for direct source inspection; declared dependency for users.
    import gymnasium as gym
    from gymnasium import spaces
    _EnvBase = gym.Env
except ImportError:  # pragma: no cover
    gym = None
    spaces = None
    _EnvBase = object

from .controllers import ARM_JOINTS, JOINT_NAMES, action_to_ctrl, joint_ranges
from .scene import scene_xml


@dataclass(frozen=True)
class PickCubeConfig:
    control_frequency: int = 20
    camera_width: int = 160
    camera_height: int = 120
    simulation_timestep: float = 0.002
    # Conservative region verified by the Phase-1 IK regression benchmark.
    workspace_x: tuple[float, float] = (0.235, 0.28)
    workspace_y: tuple[float, float] = (-0.07, 0.07)
    # 20 mm is within the measured closed-pad aperture of the supplied SO-101
    # gripper; the previous 30 mm cube exceeded it and could only be pushed.
    cube_size: float = 0.02
    lift_success_height: float = 0.10
    success_hold_steps: int = 5
    max_episode_steps: int = 240
    # Useful for fast expert regression tests; normal environment/default data
    # collection always uses a rendered camera image.
    render_observations: bool = True
    # Convex fingertip contacts are enabled only for a closing command. The
    # vendor's open STL sweep is known to intersect an object at this approach.
    closing_contact_threshold: float = 0.95
    # Pinch-center coordinates in the vendor gripper body's local frame. This
    # is the midpoint of the fixed/moving collision geometry at closed gripper,
    # measured directly from the supplied MJCF transforms (metres).
    grasp_center_local: tuple[float, float, float] = (0.0023, 0.0, -0.0378)


class SO101PickCubeEnv(_EnvBase):
    """PickCube with absolute joint-position target actions.

    Observations: RGB uint8 HxWx3, five arm-hinge radians, and a [0,1]
    opening scalar (0 closed). Action is [five radians, normalized opening].
    The normalized gripper is deliberately separate from MJCF's hinge value to
    match LeRobot's 0=closed, 100=open convention.
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 20}
    home_joint_pos = np.array([0.0, 0.80, -1.00, -0.60, 0.0], dtype=np.float64)

    def __init__(self, config: PickCubeConfig | None = None, render_mode: str | None = None):
        self.config = config or PickCubeConfig()
        if self.config.control_frequency <= 0:
            raise ValueError("control_frequency must be positive")
        self.render_mode = render_mode
        self.n_substeps = round(1.0 / (self.config.control_frequency * self.config.simulation_timestep))
        if not np.isclose(self.n_substeps * self.config.simulation_timestep, 1.0 / self.config.control_frequency):
            raise ValueError("control frequency must be an integer multiple of simulation timestep")
        self._tmp = tempfile.NamedTemporaryFile(suffix=".xml", dir=Path(__file__).parent, delete=False)
        self._tmp.write(scene_xml().encode("utf-8")); self._tmp.close()
        self.model = mujoco.MjModel.from_xml_path(self._tmp.name)
        self.model.opt.timestep = self.config.simulation_timestep
        self.data = mujoco.MjData(self.model)
        self._rng = np.random.default_rng()
        self._renderer = mujoco.Renderer(self.model, height=self.config.camera_height, width=self.config.camera_width)
        self._viewer = None
        self._step_count = 0; self._success_streak = 0; self._ever_grasped = False
        self._cube_qposadr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")]
        self._cube_dofadr = self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")]
        self._cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._gripper_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self._moving_jaw_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
        self._ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self._grasp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_center")
        self._joint_qposadr = np.asarray([self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES])
        self._lo, self._hi = joint_ranges(self.model)
        # Disable high-resolution STL contacts (including the vendor gripper
        # meshes): they have non-convex contact artefacts at the fingertip. The
        # two explicit convex pads added at the CAD fingertip transforms remain
        # enabled. They are rigid MuJoCo contacts, never a weld or cube pose edit.
        cube_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        table_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        self._grasp_pad_ids = tuple(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("fixed_grasp_pad", "moving_grasp_pad"))
        self._gripper_visual_mesh_ids = tuple(
            geom_id for geom_id in range(self.model.ngeom)
            if self.model.geom_dataid[geom_id] >= 0
            and self.model.geom_bodyid[geom_id] in (self._gripper_body_id, self._moving_jaw_body_id)
        )
        # Contact bits: table↔cube (1), pads↔cube (2); pads do not collide
        # with the table, whose surface is below the fingertips at closure.
        self.model.geom_contype[table_geom] = self.model.geom_conaffinity[table_geom] = 1
        self.model.geom_contype[cube_geom] = 1
        self.model.geom_conaffinity[cube_geom] = 3
        for geom_id in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_id not in (cube_geom, table_geom) and geom_name not in ("fixed_grasp_pad", "moving_grasp_pad"):
                self.model.geom_contype[geom_id] = 0
                self.model.geom_conaffinity[geom_id] = 0
        self._set_grasp_pad_contacts(False)
        if spaces:
            self.action_space = spaces.Box(np.r_[self._lo[:5], 0.0].astype(np.float32), np.r_[self._hi[:5], 1.0].astype(np.float32), dtype=np.float32)
            self.observation_space = spaces.Dict({
                "rgb": spaces.Box(0, 255, (self.config.camera_height, self.config.camera_width, 3), dtype=np.uint8),
                "joint_pos": spaces.Box(self._lo[:5].astype(np.float32), self._hi[:5].astype(np.float32), dtype=np.float32),
                "gripper": spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            })

    def seed(self, seed: int | None = None) -> list[int]:
        self._rng = np.random.default_rng(seed)
        return [] if seed is None else [seed]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None: self.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._joint_qposadr[:5]] = self.home_joint_pos
        self.data.qpos[self._joint_qposadr[5]] = self._hi[5]  # open
        xy = self._sample_cube_xy(options)
        qa = self._cube_qposadr
        self.data.qpos[qa:qa + 7] = (xy[0], xy[1], self.config.cube_size / 2, 1, 0, 0, 0)
        self.data.ctrl[:] = action_to_ctrl(self.model, np.r_[self.home_joint_pos, 1.0])
        self._set_grasp_pad_contacts(False)
        mujoco.mj_forward(self.model, self.data)
        # Let the free object settle while keeping the robot at its home command.
        for _ in range(20): mujoco.mj_step(self.model, self.data)
        self._step_count = self._success_streak = 0; self._ever_grasped = False
        obs = self._observation()
        return obs, self._info()

    def step(self, action: np.ndarray):
        requested = np.asarray(action, dtype=np.float32)
        ctrl = action_to_ctrl(self.model, requested)
        self._set_grasp_pad_contacts(bool(requested[5] < self.config.closing_contact_threshold))
        self.data.ctrl[:] = ctrl
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        grasped = self._grasped()
        self._ever_grasped |= grasped
        elevated = self.cube_pose[2] >= self.config.lift_success_height
        if self._ever_grasped and elevated: self._success_streak += 1
        else: self._success_streak = 0
        success = self._success_streak >= self.config.success_hold_steps
        terminated = bool(success)
        truncated = self._step_count >= self.config.max_episode_steps
        info = self._info(); info.update({"requested_action": requested.copy(), "applied_action": np.r_[ctrl[:5], (ctrl[5]-self._lo[5])/(self._hi[5]-self._lo[5])].astype(np.float32), "grasped": grasped, "elevated": elevated, "success": success})
        return self._observation(), float(success), terminated, truncated, info

    def get_info(self) -> dict[str, Any]:
        """Debug state aligned with the current observation, without stepping."""
        return self._info()

    @property
    def cube_pose(self) -> np.ndarray:
        qa = self._cube_qposadr
        return self.data.qpos[qa:qa + 7].copy()

    @property
    def end_effector_pose(self) -> np.ndarray:
        return np.concatenate((self.data.site_xpos[self._ee_site_id], self.data.site_xmat[self._ee_site_id])).copy()

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]: return self._lo.copy(), self._hi.copy()

    @property
    def grasp_center_position(self) -> np.ndarray:
        return self.data.site_xpos[self._grasp_site_id].copy()

    def gripperframe_target_for_grasp_center(self, desired_center: np.ndarray) -> np.ndarray:
        """Convert a desired physical pinch-center location into the IK site target."""
        R = self.data.xmat[self._gripper_body_id].reshape(3, 3)
        local_site = R.T @ (self.data.site_xpos[self._ee_site_id] - self.data.xpos[self._gripper_body_id])
        return np.asarray(desired_center, dtype=float) - R @ (np.asarray(self.config.grasp_center_local) - local_site)

    @property
    def lowest_gripper_visual_z(self) -> float:
        """Lowest CAD gripper/moving-jaw mesh vertex in the current world pose."""
        lowest = np.inf
        for geom_id in self._gripper_visual_mesh_ids:
            mesh_id = self.model.geom_dataid[geom_id]
            start, count = self.model.mesh_vertadr[mesh_id], self.model.mesh_vertnum[mesh_id]
            vertices = self.model.mesh_vert[start:start + count]
            world = vertices @ self.data.geom_xmat[geom_id].reshape(3, 3).T + self.data.geom_xpos[geom_id]
            lowest = min(lowest, float(world[:, 2].min()))
        return lowest

    def render(self):
        if not self.config.render_observations:
            return np.zeros((self.config.camera_height, self.config.camera_width, 3), dtype=np.uint8)
        self._renderer.update_scene(self.data, camera="external_rgb")
        return self._renderer.render().copy()

    def close(self):
        self._renderer.close()
        if self._viewer is not None: self._viewer.close()
        Path(self._tmp.name).unlink(missing_ok=True)

    def _sample_cube_xy(self, options: dict[str, Any] | None) -> np.ndarray:
        if options and "cube_xy" in options:
            xy = np.asarray(options["cube_xy"], dtype=float)
            if xy.shape != (2,): raise ValueError("cube_xy must have shape (2,)")
            if not (self.config.workspace_x[0] <= xy[0] <= self.config.workspace_x[1] and self.config.workspace_y[0] <= xy[1] <= self.config.workspace_y[1]):
                raise ValueError("cube_xy lies outside configured workspace")
            return xy
        return np.array([self._rng.uniform(*self.config.workspace_x), self._rng.uniform(*self.config.workspace_y)])

    def _observation(self) -> dict[str, np.ndarray]:
        q = self.data.qpos[self._joint_qposadr]
        # Position servos may overshoot their target by a few milliradians; the
        # public LeRobot-aligned representation is nevertheless a bounded
        # normalized opening, so clip this derived value (never simulator qpos).
        opening = np.clip((q[5] - self._lo[5]) / (self._hi[5] - self._lo[5]), 0.0, 1.0)
        return {"rgb": self.render(), "joint_pos": q[:5].astype(np.float32).copy(), "gripper": np.array([opening], dtype=np.float32)}

    def _grasped(self) -> bool:
        q = self.data.qpos[self._joint_qposadr]
        closed = (q[5] - self._lo[5]) / (self._hi[5] - self._lo[5]) < 0.05
        distance = np.linalg.norm(self.grasp_center_position - self.cube_pose[:3])
        # Contact is preferred; proximity makes this robust to a single contact
        # being absent on an otherwise enclosed mesh configuration.
        contacted_bodies = {self._cube_gripper_contact_body(i) for i in range(self.data.ncon)}
        # A true pinch needs contact with both the fixed gripper and moving jaw;
        # cube/table or one-sided housing contact is explicitly not a grasp.
        pinched = {self._gripper_body_id, self._moving_jaw_body_id} <= contacted_bodies
        return bool(closed and distance < 0.045 and pinched)

    def _cube_gripper_contact_body(self, index: int) -> int | None:
        c = self.data.contact[index]
        b1 = self.model.geom_bodyid[c.geom1]; b2 = self.model.geom_bodyid[c.geom2]
        if b1 == self._cube_body_id and b2 in (self._gripper_body_id, self._moving_jaw_body_id): return int(b2)
        if b2 == self._cube_body_id and b1 in (self._gripper_body_id, self._moving_jaw_body_id): return int(b1)
        return None

    def _set_grasp_pad_contacts(self, enabled: bool) -> None:
        for geom_id in self._grasp_pad_ids:
            self.model.geom_contype[geom_id] = 2 if enabled else 0
            self.model.geom_conaffinity[geom_id] = 2 if enabled else 0

    def _info(self) -> dict[str, Any]:
        return {"cube_pose": self.cube_pose, "end_effector_pose": self.end_effector_pose, "step": self._step_count, "collision_count": int(self.data.ncon), "lowest_gripper_visual_z": self.lowest_gripper_visual_z, "task_success": self._success_streak >= self.config.success_hold_steps}
