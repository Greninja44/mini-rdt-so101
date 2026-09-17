"""Explicit SO-101 action conversion and damped least-squares position IK."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import mujoco

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER_JOINT = "gripper"
JOINT_NAMES = ARM_JOINTS + (GRIPPER_JOINT,)


@dataclass(frozen=True)
class SO101Convention:
    """Action convention aligned with LeRobot's named follower motor order.

    action[:5] are absolute calibrated MJCF hinge targets in radians. action[5]
    is normalized gripper opening: 0.0 closed, 1.0 open (LeRobot uses 0..100).
    """
    arm_names: tuple[str, ...] = ARM_JOINTS


def joint_ranges(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES])
    if np.any(ids < 0):
        raise RuntimeError("vendored SO-101 joint names do not match expected convention")
    ranges = model.jnt_range[ids].copy()
    return ranges[:, 0], ranges[:, 1]


def action_to_ctrl(model: mujoco.MjModel, action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (6,):
        raise ValueError(f"expected action shape (6,), got {action.shape}")
    lo, hi = joint_ranges(model)
    ctrl = np.empty(6, dtype=np.float64)
    ctrl[:5] = np.clip(action[:5], lo[:5], hi[:5])
    ctrl[5] = lo[5] + np.clip(action[5], 0.0, 1.0) * (hi[5] - lo[5])
    return ctrl


def ctrl_to_action(model: mujoco.MjModel, ctrl: np.ndarray) -> np.ndarray:
    lo, hi = joint_ranges(model)
    ctrl = np.asarray(ctrl, dtype=np.float64)
    result = np.empty(6, dtype=np.float32)
    result[:5] = np.clip(ctrl[:5], lo[:5], hi[:5])
    result[5] = np.clip((ctrl[5] - lo[5]) / (hi[5] - lo[5]), 0.0, 1.0)
    return result


def position_ik_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    target_xyz: np.ndarray,
    damping: float = 2e-3,
    max_delta: float = 0.08,
    joint_limit_margin: float = 0.04,
) -> np.ndarray:
    """One bounded DLS Cartesian-position IK update for five arm joints.

    This intentionally does not force orientation: SO-101 has five arm DOF and
    LeRobot treats orientation as soft/partial for that configuration.
    """
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    arm_qpos = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM_JOINTS])
    arm_dof = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM_JOINTS])
    J = jacp[:, arm_dof]
    error = np.asarray(target_xyz, dtype=float) - data.site_xpos[site_id]
    dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), error)
    dq = np.clip(dq, -max_delta, max_delta)
    lo, hi = joint_ranges(model)
    q = data.qpos[arm_qpos] + dq
    # The vendor position servo has finite stiffness, so a target only 0.01 rad
    # from a hard range can overshoot during a 50 ms control interval. Keep the
    # *command* 0.04 rad inside the exact MJCF range; this leaves headroom for
    # physical settling without reporting an invalid measured joint state.
    return np.clip(q, lo[:5] + joint_limit_margin, hi[:5] - joint_limit_margin)
