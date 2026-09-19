"""Physics-v2 regression tests (docs/research/physics_v2_spec.md). None require dataset artifacts."""
from dataclasses import replace

import numpy as np
import pytest

from simulation.env import PickCubeConfig, SO101PickCubeEnv
from simulation.expert_v2 import PickCubeExpertV2
from simulation.legacy_expert import LegacyPickCubeExpert
from simulation.pose_ik import PoseIK, top_down_rotation


@pytest.fixture(scope="module")
def env():
    return SO101PickCubeEnv(PickCubeConfig(render_observations=False))


def run_expert(env, seed):
    env.reset(seed=seed); ex = PickCubeExpertV2(env); ex.reset(); info = {}; t = 0; pinches = []
    while not ex.done and t < 240:
        _, _, _, _, info = env.step(ex.action()); ex.observe(info); t += 1
        pinches.append(info["contacts"]["side_pinch"])
    return ex, info, pinches


def test_collision_bits_robot_pad_cube_table(env):
    m, table, cube = env.model, env._table_geom, env._cube_geom
    collide = lambda a, b: bool((m.geom_contype[a] & m.geom_conaffinity[b]) or (m.geom_contype[b] & m.geom_conaffinity[a]))
    assert collide(cube, table)
    for pad in env._grasp_pad_ids: assert collide(pad, table) and collide(pad, cube)
    links = [g for g in env._robot_collision_geoms if g not in env._grasp_pad_ids]
    assert len(links) >= 10 and all(collide(g, table) for g in links)
    assert not any(collide(g, cube) for g in links)  # links reach the cube only through the pads (spec)


def test_fingertips_commanded_below_table_cannot_penetrate(env):
    env.reset(seed=3006); ik = PoseIK(env.model); cube = env.cube_pose[:3]
    target = np.array([cube[0] + 0.04, cube[1], -0.03])  # 30 mm BELOW the tabletop, beside the cube
    q, _, _, _ = ik.solve(env.data.qpos[env._joint_qposadr[:5]], target, top_down_rotation(0.0))
    contacts = 0; info = {}
    for _ in range(60):
        _, _, _, _, info = env.step(np.r_[q, 0.3]); contacts += info["contacts"]["robot_table_contacts"] > 0
    assert contacts > 10
    assert info["max_robot_table_penetration"] < env.config.v2_robot_table_penetration_tol


def test_table_supports_cube_under_downward_force(env):
    env.reset(seed=3000); pen = 0.0
    for _ in range(40):
        env.data.xfrc_applied[env._cube_body_id, 2] = -20.0  # 20 N push, ~200x the cube's weight
        _, _, _, _, info = env.step(np.r_[env.home_joint_pos, 1.0]); pen = max(pen, info["contacts"]["cube_table_penetration"])
    env.data.xfrc_applied[:] = 0
    assert pen < env.config.v2_cube_table_penetration_tol


def test_old_sandwich_grasp_is_classified_invalid():
    v1 = SO101PickCubeEnv(PickCubeConfig(render_observations=False, physics="v1"))
    v1.reset(seed=3000); ex = LegacyPickCubeExpert(v1); ex.reset(); vertical = side = False
    while not ex.done:
        _, _, _, _, info = v1.step(ex.action()); ex.observe(info)
        vertical |= info["contacts"]["vertical_pad_contact"]; side |= info["contacts"]["side_pinch"]
    assert ex.state.value == "SUCCESS"          # it "succeeds" under the invalid v1 physics...
    assert vertical and not side                # ...but only through a vertical (sandwich) contact


def test_old_sandwich_trajectory_on_v2_does_not_succeed_or_penetrate(env):
    v1 = SO101PickCubeEnv(PickCubeConfig(render_observations=False, physics="v1"))
    v1.reset(seed=3006); ex = LegacyPickCubeExpert(v1); ex.reset(); actions = []
    while not ex.done:
        a = ex.action(); actions.append(a); _, _, _, _, info = v1.step(a); ex.observe(info)
    env.reset(seed=3006); info = {}
    for a in actions: _, _, _, _, info = env.step(a)
    assert not info["success"] and info["max_robot_table_penetration"] < env.config.v2_robot_table_penetration_tol


@pytest.mark.parametrize("seed", [3000, 3007, 4042])
def test_expert_produces_valid_side_pinch_success(env, seed):
    ex, info, pinches = run_expert(env, seed)
    assert info["success"] and ex.state.value == "SUCCESS" and info["invalid_reason"] is None
    assert all(pinches[-env.config.success_hold_steps:])
    n_fixed, n_moving = (np.array(n) for n in info["contacts"]["pad_normals"])
    assert abs(n_fixed[2]) <= 0.5 and abs(n_moving[2]) <= 0.5 and n_fixed @ n_moving <= -0.5
    assert info["max_robot_table_penetration"] < 1e-3 and info["max_cube_table_penetration"] < 1e-3 and info["max_pad_cube_penetration"] < 1.5e-3


def test_success_impossible_with_invalid_penetration():
    strict = SO101PickCubeEnv(PickCubeConfig(render_observations=False, v2_cube_table_penetration_tol=0.0))
    ex, info, _ = run_expert(strict, 3000)
    assert not info["success"] and info["invalid_reason"] == "cube_table_penetration"


def test_v1_remains_selectable_and_distinct():
    v1 = SO101PickCubeEnv(PickCubeConfig(render_observations=False, physics="v1"))
    table = v1._table_geom
    links = [g for g in range(v1.model.ngeom) if v1.model.geom_group[g] == 3 and v1.model.geom_dataid[g] >= 0]
    assert all(v1.model.geom_contype[g] == 0 and v1.model.geom_conaffinity[g] == 0 for g in links)
    assert v1.model.geom_conaffinity[table] == 1
