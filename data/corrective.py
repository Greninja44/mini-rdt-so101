"""Corrective-data primitives: exact simulator snapshots and counterfactual expert chunks.

Label contract for every corrective sample:
    observation of the ACTUAL (perturbed / policy-induced) state s_t
    -> the H-step action chunk the data-generating expert would execute
       FROM s_t, obtained by simulating that expert forward from an exact
       snapshot of s_t (physics only) and then restoring s_t.
Never: nominal observation -> perturbed action, nor perturbed observation ->
nominal action. Chunks shorter than H (expert finished) are hold-padded, the
same convention as the validated TinyRDT training ('hold' padding), with the
valid prefix recorded separately.
"""
from __future__ import annotations
from dataclasses import replace

import mujoco
import numpy as np

from simulation.legacy_expert import LegacyPickCubeExpert

_ENV_FIELDS = ("_step_count", "_success_streak", "_ever_grasped")
_EXPERT_FIELDS = ("state", "steps_in_state", "failure_category", "_grasp_streak", "_cube_reference")


def snapshot(env, expert=None):
    """Full mjData copy. mj_step leaves kinematics (site_xpos, Jacobians) from
    the last pre-integration substep and the data-generating expert's IK reads
    them, so restoring only integration state + mj_forward is NOT exact."""
    sim = mujoco.MjData(env.model); mujoco.mj_copyData(sim, env.model, env.data)
    pads = (env.model.geom_contype.copy(), env.model.geom_conaffinity.copy())
    snap = {"sim": sim, "pads": pads, "env": {k: getattr(env, k) for k in _ENV_FIELDS}}
    if expert is not None:
        snap["expert"] = {k: (v.copy() if isinstance(v := getattr(expert, k), np.ndarray) else v) for k in _EXPERT_FIELDS}
        snap["transitions"] = list(expert.transitions)
    return snap


def restore(env, snap, expert=None):
    mujoco.mj_copyData(env.data, env.model, snap["sim"])
    env.model.geom_contype[:], env.model.geom_conaffinity[:] = snap["pads"]
    for k, v in snap["env"].items(): setattr(env, k, v)
    if expert is not None:
        for k, v in snap["expert"].items(): setattr(expert, k, v.copy() if isinstance(v, np.ndarray) else v)
        expert.transitions = list(snap["transitions"])


class _NoRender:
    """Physics-only stepping for counterfactuals (camera frames are not needed)."""
    def __init__(self, env): self.env = env
    def __enter__(self):
        self.saved = self.env.config; self.env.config = replace(self.saved, render_observations=False)
    def __exit__(self, *exc): self.env.config = self.saved


def expert_chunk(env, expert: LegacyPickCubeExpert, horizon: int = 16):
    """Expert action chunk from the current exact state; state is restored afterwards.

    Returns (chunk[H,6] hold-padded, n_valid, expert_done_ok) where
    expert_done_ok is False if the expert FAILED inside the horizon.
    """
    snap = snapshot(env, expert); actions = []; failed = False
    with _NoRender(env):
        for _ in range(horizon):
            if expert.done:
                failed = expert.state.value == "FAILED"; break
            a = expert.action(); actions.append(a)
            _, _, _, _, info = env.step(a); expert.observe(info)
    restore(env, snap, expert)
    n = len(actions)
    if n == 0:
        return None, 0, not failed
    chunk = np.stack(actions); chunk = np.concatenate((chunk, np.repeat(chunk[-1:], horizon - n, 0)))
    return chunk.astype(np.float32), n, not failed
