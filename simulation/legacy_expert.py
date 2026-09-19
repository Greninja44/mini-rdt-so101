"""Reconstruction of the expert that generated artifacts/pickcube_smoke100_rgb160.

The dataset (written 2026-09-16 11:22) predates the current PickCubeExpert
(expert.py modified 12:30). Its recorded actions are reproduced, in every
phase, by state-feedback DLS IK steps with:
  approach target  = cube + 0.10 m (not the current 0.18 m joint interpolation)
  joint_limit_margin = 0.01 rad (current code: 0.04)
All other behavior (targets, 4 mm geometric close condition, transitions) is
the current PickCubeExpert. Unlike the current approach, every phase here is
recomputed from the ACTUAL arm configuration, so it is a valid corrective
oracle for off-trajectory states in approach too. See docs/research/research_log.md.
"""
from __future__ import annotations
from dataclasses import replace

import numpy as np

from .controllers import position_ik_step
from .expert import ExpertConfig, ExpertState, PickCubeExpert

LEGACY_CONFIG = replace(ExpertConfig(), above_offset=0.10)
LEGACY_JOINT_LIMIT_MARGIN = 0.01
EXPERT_VERSION = "legacy-dls-above0.10-margin0.01"


class LegacyPickCubeExpert(PickCubeExpert):
    def __init__(self, env, config: ExpertConfig | None = None, verbose: bool = False):
        super().__init__(env, config or LEGACY_CONFIG, verbose)

    def reset(self) -> None:
        self.state = ExpertState.HOME; self.steps_in_state = 0
        self.transitions = [self.state.value]; self.failure_category = None
        self._grasp_streak = 0
        self._cube_reference = self.env.cube_pose[:3].copy()

    def action(self) -> np.ndarray:
        if self.state == ExpertState.HOME:
            arm = self.env.home_joint_pos.copy()
        else:
            arm = position_ik_step(self.env.model, self.env.data, self.env._grasp_site_id, self._target_xyz(),
                                   joint_limit_margin=LEGACY_JOINT_LIMIT_MARGIN)
        opening = 0.0 if self.state in (ExpertState.CLOSE_GRIPPER, ExpertState.LIFT, ExpertState.HOLD, ExpertState.SUCCESS) else 1.0
        return np.r_[arm, opening].astype(np.float32)
