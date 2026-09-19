"""Physics-v2 PickCube expert: a genuine top-down SIDE PINCH with table clearance.

HOME -> PREGRASP (above the cube, top-down, jaws aligned with cube faces, open)
     -> DESCEND (straight down to a clearance-checked grasp height)
     -> CLOSE (until a valid side pinch persists) -> LIFT -> HOLD -> SUCCESS

State feedback: every control tick the Cartesian setpoint moves from the ACTUAL
grasp-site pose toward the phase goal with bounded speed, and is converted to
joint targets by 6-D pose IK warm-started from the ACTUAL joint configuration.
Nothing is replayed on a clock. The grasp height is chosen at reset by a
read-only check that every finger/jaw/pad convex hull clears the table.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

import mujoco
import numpy as np

from .pose_ik import PoseIK, rotation_error, top_down_rotation

EXPERT_V2_VERSION = "expert-v2-sidepinch-3"


class State(str, Enum):
    HOME = "HOME"; PREGRASP = "PREGRASP"; DESCEND = "DESCEND"; CLOSE = "CLOSE"
    LIFT = "LIFT"; HOLD = "HOLD"; SUCCESS = "SUCCESS"; FAILED = "FAILED"


@dataclass(frozen=True)
class ExpertV2Config:
    approach_opening: float = 0.25          # ~47 mm pad gap around the 20 mm cube
    fixed_pad_clearance: float = 0.005      # cube face to fixed-pad face before closing
    table_clearance: float = 0.003          # lowest finger/jaw/pad hull above the table at the grasp pose
    min_side_overlap: float = 0.006         # pad band that must overlap the cube side face
    pregrasp_height: float = 0.045          # above the grasp point
    lift_height: float = 0.135              # grasp-point z while lifting (cube reaches > 0.10 m)
    max_linear_step: float = 0.010          # m per control tick
    max_angular_step: float = 0.12          # rad per control tick
    max_joint_step: float = 0.06            # rad per control tick (joint-space approach)
    descent_step: float = 0.006             # m per tick while descending (servo overshoot grows with speed)
    final_descent_height: float = 0.030     # below this height above the grasp point, descend slowly...
    final_descent_step: float = 0.003       # ...at most 3 mm per tick
    max_lateral_error: float = 0.0025       # ...and stop descending to re-centre if laterally off by more than this
    position_tolerance: float = 0.003
    rotation_tolerance: float = 0.05
    pinch_confirm_steps: int = 3
    timeouts: tuple = (("PREGRASP", 80), ("DESCEND", 60), ("CLOSE", 30), ("LIFT", 60), ("HOLD", 40))


class PickCubeExpertV2:
    def __init__(self, env, config: ExpertV2Config | None = None):
        self.env, self.cfg = env, config or ExpertV2Config()
        self.ik = PoseIK(env.model); self.state = State.HOME; self.steps_in_state = 0; self.failure_category = None
        self.transitions = [self.state.value]; self._pinch_streak = 0; self.plan = {}

    # ---------- planning (read-only; uses scratch MjData inside PoseIK) ----------
    def _grasp_site_target(self, point, R):
        """Site target that puts gripper-frame point P=(fixed pad face + clearance + cube half, 0, pad z) at `point`."""
        m = self.env.model; site_local = m.site_pos[self.env._grasp_site_id]
        pad = self.env._grasp_pad_ids[0]; pad_local = m.geom_pos[pad]; pad_half = m.geom_size[pad]
        p_local = np.array([pad_local[0] + pad_half[0] + self.cfg.fixed_pad_clearance + self.env.config.cube_size / 2, pad_local[1], pad_local[2]])
        return point + R @ (site_local - p_local)

    def _lowest_distal_z(self, q, opening):
        e, m = self.env, self.env.model; d = self.ik.d
        d.qpos[self.ik.qadr] = q; d.qpos[e._joint_qposadr[5]] = e._lo[5] + opening * (e._hi[5] - e._lo[5])
        mujoco.mj_kinematics(m, d); low = np.inf
        for g in e._robot_collision_geoms:
            if m.geom_bodyid[g] not in (e._gripper_body_id, e._moving_jaw_body_id): continue
            if m.geom_dataid[g] >= 0:
                mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            else:
                v = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * m.geom_size[g]
            low = min(low, float((v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min()))
        return low

    def reset(self) -> None:
        e, c = self.env, self.cfg
        self.state = State.HOME; self.steps_in_state = 0; self.transitions = [self.state.value]; self.failure_category = None; self._pinch_streak = 0
        cube = e.cube_pose; center = cube[:3].copy(); w, x, y, z = cube[3:]
        cube_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        natural = np.arctan2(center[1], center[0])  # jaws closing radially keeps wrist_roll near neutral
        yaw = min((cube_yaw + k * np.pi / 2 for k in range(-4, 5)), key=lambda a: abs(np.angle(np.exp(1j * (a - natural)))))
        R = top_down_rotation(yaw); table = 0.0; half = e.config.cube_size / 2
        pad_half_z = e.model.geom_size[e._grasp_pad_ids[0]][2]
        q = e.data.qpos[e._joint_qposadr[:5]].copy(); lift_z = None
        for dz in np.arange(0.0, 0.0151, 0.001):  # raise until every distal hull clears the table
            point = center + np.array([0, 0, dz])
            q_g, pe, re, ok = self.ik.solve(q, self._grasp_site_target(point, R), R)
            low = min(self._lowest_distal_z(q_g, c.approach_opening), self._lowest_distal_z(q_g, 0.0))
            overlap = (center[2] + half) - (point[2] - pad_half_z)
            if ok and low - table >= c.table_clearance: lift_z = dz; break
        feasible = lift_z is not None and overlap >= c.min_side_overlap
        grasp_point = center + np.array([0, 0, lift_z or 0.0])
        # Pre-grasp configuration solved from the grasp solution, so it lies on the same IK branch.
        q_pre, _, _, ok_pre = self.ik.solve(q_g, self._grasp_site_target(grasp_point, R) + np.array([0, 0, c.pregrasp_height]), R)
        feasible = feasible and ok_pre
        self.plan = {"yaw": float(yaw), "R": R, "grasp_point": grasp_point, "grasp_dz": lift_z, "q_grasp": q_g, "q_pregrasp": q_pre,
                     "lowest_hull_z_at_grasp": low, "side_overlap": float(overlap), "feasible": bool(feasible)}
        if not feasible: self._fail("PLAN_INFEASIBLE")

    # ---------- control ----------
    def _site_pose(self):
        e = self.env; return e.grasp_center_position.copy(), e.data.xmat[e._gripper_body_id].reshape(3, 3).copy()

    def _goal(self):
        p, c = self.plan, self.cfg; g = self._grasp_site_target(p["grasp_point"], p["R"])
        if self.state in (State.HOME, State.PREGRASP): return g + np.array([0, 0, c.pregrasp_height]), p["R"], 1.0
        if self.state in (State.DESCEND, State.CLOSE): return g, p["R"], 1.0
        return np.array([g[0], g[1], c.lift_height]), p["R"], 0.02  # LIFT/HOLD: position first, orientation relaxed

    def action(self) -> np.ndarray:
        e, c = self.env, self.cfg
        q = e.data.qpos[e._joint_qposadr[:5]].copy()
        if self.state in (State.FAILED, State.SUCCESS) or not self.plan.get("feasible"):
            return np.r_[q, 0.0 if self.state == State.SUCCESS else 1.0].astype(np.float32)
        opening = c.approach_opening if self.state in (State.HOME, State.PREGRASP, State.DESCEND) else 0.0
        if self.state in (State.HOME, State.PREGRASP):
            # Joint-space feedback from the ACTUAL configuration toward the planned pre-grasp configuration.
            dq = self.plan["q_pregrasp"] - q; scale = min(1.0, c.max_joint_step / max(np.abs(dq).max(), 1e-12))
            return np.r_[q + scale * dq, opening].astype(np.float32)
        goal_p, goal_R, rot_w = self._goal(); p, R = self._site_pose()
        step = c.descent_step if self.state == State.DESCEND else c.max_linear_step
        if self.state == State.DESCEND and p[2] - goal_p[2] < c.final_descent_height:
            step = c.final_descent_step  # slow final descent; never descend while laterally misaligned
            if np.linalg.norm((goal_p - p)[:2]) > c.max_lateral_error: goal_p = np.array([goal_p[0], goal_p[1], p[2]])
        dp = goal_p - p; n = np.linalg.norm(dp); p_cmd = p + (dp if n <= step else dp * step / n)
        rv = rotation_error(R, goal_R); ang = np.linalg.norm(rv)
        R_cmd = goal_R if ang <= c.max_angular_step else _rotvec(rv * c.max_angular_step / ang) @ R
        q_cmd, _, _, _ = self.ik.solve(q, p_cmd, R_cmd, rot_weight=0.05 * rot_w)
        return np.r_[q_cmd, opening].astype(np.float32)

    def observe(self, info: dict) -> None:
        self.steps_in_state += 1; c = self.cfg; s = self.state
        if s in (State.SUCCESS, State.FAILED): return
        if info.get("success"): self._transition(State.SUCCESS); return
        if info.get("invalid_reason"): self._fail("INVALID_" + info["invalid_reason"].upper()); return
        goal_p, goal_R, _ = self._goal(); p, R = self._site_pose()
        at = np.linalg.norm(goal_p - p) < c.position_tolerance and np.linalg.norm(rotation_error(R, goal_R)) < c.rotation_tolerance
        pinch = bool(info.get("contacts", {}).get("side_pinch"))
        self._pinch_streak = self._pinch_streak + 1 if pinch else 0
        if s == State.HOME: self._transition(State.PREGRASP)
        elif s == State.PREGRASP and at: self._transition(State.DESCEND)
        elif s == State.DESCEND and at: self._transition(State.CLOSE)
        elif s == State.CLOSE and self._pinch_streak >= c.pinch_confirm_steps: self._transition(State.LIFT)
        elif s in (State.LIFT, State.HOLD) and not pinch and self.env.cube_pose[2] > 0.03: self._fail("OBJECT_DROPPED"); return
        elif s == State.LIFT and info.get("elevated") and pinch: self._transition(State.HOLD)
        limit = dict(c.timeouts).get(self.state.value)
        if limit and self.steps_in_state > limit: self._fail(f"{self.state.value}_TIMEOUT")

    @property
    def done(self) -> bool: return self.state in (State.SUCCESS, State.FAILED)

    @property
    def config(self) -> ExpertV2Config: return self.cfg

    def _transition(self, state):
        if state != self.state: self.state = state; self.steps_in_state = 0; self.transitions.append(state.value)

    def _fail(self, category):
        self.failure_category = category; self._transition(State.FAILED)


def _rotvec(v):
    a = np.linalg.norm(v)
    if a < 1e-12: return np.eye(3)
    k = v / a; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
