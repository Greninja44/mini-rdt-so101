"""Damped-least-squares 6-D pose IK for the SO-101 grasp site (physics-v2 expert).

The SO-101 has 5 arm joints. A top-down side grasp needs position (3), gripper
pointing straight down (pitch, which the in-plane pitch joints provide) and a jaw
yaw (wrist_roll), so the full 6-D target is consistent for top-down poses. The
solver works on a scratch copy of the model data, so the live simulation is never touched.
"""
from __future__ import annotations

import mujoco
import numpy as np

ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def top_down_rotation(yaw: float) -> np.ndarray:
    """Gripper frame for a top-down grasp: gripper z = world +z (fingers point down),
    gripper x (jaw closing axis) = (cos yaw, sin yaw, 0)."""
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0]); z = np.array([0.0, 0.0, 1.0]); y = np.cross(z, x)
    return np.stack((x, y, z), axis=1)


def rotation_error(R_current: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    """World-frame axis-angle vector rotating R_current onto R_target."""
    E = R_target @ R_current.T
    angle = np.arccos(np.clip((np.trace(E) - 1) / 2, -1, 1))
    if angle < 1e-9: return np.zeros(3)
    axis = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]]) / (2 * np.sin(angle))
    return axis * angle


class PoseIK:
    def __init__(self, model: mujoco.MjModel, site: str = "grasp_center", body: str = "gripper"):
        self.m = model; self.d = mujoco.MjData(model)
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        self.qadr = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM])
        self.dadr = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM])
        jid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM]
        self.lo = model.jnt_range[jid, 0]; self.hi = model.jnt_range[jid, 1]

    def pose(self, q):
        self.d.qpos[self.qadr] = q; mujoco.mj_kinematics(self.m, self.d); mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.xmat[self.body].reshape(3, 3).copy()

    def solve(self, q0, pos, R, iters=300, damping=1e-4, margin=0.02, rot_weight=0.05, tol=(2e-4, 2e-3)):
        """Returns (q, position error m, rotation error rad, converged). rot_weight scales radians to metres."""
        q = np.clip(np.asarray(q0, float).copy(), self.lo + margin, self.hi - margin)
        jacp = np.zeros((3, self.m.nv)); jacr = np.zeros((3, self.m.nv))
        for _ in range(iters):
            p, Rc = self.pose(q)
            e = np.r_[pos - p, rot_weight * rotation_error(Rc, R)]
            if np.linalg.norm(e[:3]) < tol[0] and np.linalg.norm(e[3:]) / rot_weight < tol[1]: break
            mujoco.mj_jacSite(self.m, self.d, jacp, jacr, self.site)
            J = np.r_[jacp[:, self.dadr], rot_weight * jacr[:, self.dadr]]
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo + margin, self.hi - margin)
        p, Rc = self.pose(q)
        pe = float(np.linalg.norm(pos - p)); re = float(np.linalg.norm(rotation_error(Rc, R)))
        return q, pe, re, pe < 1e-3 and re < 0.02
