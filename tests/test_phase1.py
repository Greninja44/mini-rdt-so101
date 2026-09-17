from __future__ import annotations
import numpy as np
from data.dataset import EpisodeRecorder, load_episode, make_episode_metadata
from simulation.expert import ExpertConfig
from data.validate import validate_dataset
from simulation.env import PickCubeConfig, SO101PickCubeEnv
from simulation.expert import ExpertState, PickCubeExpert


def make_env() -> SO101PickCubeEnv:
    return SO101PickCubeEnv(PickCubeConfig(camera_width=24, camera_height=18, render_observations=False))


def run_expert(env: SO101PickCubeEnv, seed: int = 3):
    env.reset(seed=seed); expert = PickCubeExpert(env); expert.reset(); last = None
    while not expert.done:
        _, _, terminated, truncated, last = env.step(expert.action()); expert.observe(last)
        if terminated or truncated: break
    return expert, last


def test_reset_and_observation_shapes():
    env = make_env(); obs, info = env.reset(seed=1)
    assert obs["rgb"].shape == (18, 24, 3) and obs["rgb"].dtype == np.uint8
    assert obs["joint_pos"].shape == (5,) and obs["gripper"].shape == (1,)
    assert info["cube_pose"].shape == (7,); env.close()


def test_seed_is_deterministic_and_cube_is_randomized():
    env = make_env(); _, a = env.reset(seed=9); _, b = env.reset(seed=9); _, c = env.reset(seed=10)
    assert np.allclose(a["cube_pose"], b["cube_pose"])
    assert not np.allclose(a["cube_pose"][:2], c["cube_pose"][:2]); env.close()


def test_cube_workspace_and_explicit_pose_validation():
    env = make_env()
    for seed in range(20):
        _, info = env.reset(seed=seed)
        x, y = info["cube_pose"][:2]
        assert env.config.workspace_x[0] <= x <= env.config.workspace_x[1]
        assert env.config.workspace_y[0] <= y <= env.config.workspace_y[1]
    try:
        env.reset(options={"cube_xy": [env.config.workspace_x[1] + 0.01, 0.0]})
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-workspace cube pose was accepted")
    env.close()


def test_rendered_rgb_is_nonblank_uint8():
    env = SO101PickCubeEnv(PickCubeConfig(camera_width=24, camera_height=18, render_observations=True))
    obs, _ = env.reset(seed=4)
    assert obs["rgb"].dtype == np.uint8 and obs["rgb"].shape == (18, 24, 3)
    assert np.any(obs["rgb"])
    env.close()


def test_action_shape_and_joint_limit_enforcement():
    env = make_env(); env.reset(seed=1); lo, hi = env.joint_limits
    _, _, _, _, info = env.step(np.array([99, -99, 99, -99, 99, 8], dtype=np.float32))
    assert np.all(info["applied_action"][:5] <= hi[:5] + 1e-5) and np.all(info["applied_action"][:5] >= lo[:5] - 1e-5)
    assert info["applied_action"][5] == 1.0
    try: env.step(np.zeros(5))
    except ValueError: pass
    else: raise AssertionError("invalid action shape was accepted")
    env.close()


def test_expert_state_machine_and_success_detection():
    env = make_env(); expert, info = run_expert(env)
    assert expert.state is ExpertState.SUCCESS
    assert expert.transitions == [s.value for s in (ExpertState.HOME, ExpertState.MOVE_ABOVE_OBJECT, ExpertState.DESCEND, ExpertState.CLOSE_GRIPPER, ExpertState.LIFT, ExpertState.HOLD, ExpertState.SUCCESS)]
    assert info["task_success"] and env.cube_pose[2] >= env.config.lift_success_height
    env.close()


def test_expert_timeout_records_failure_category():
    env = make_env(); env.reset(seed=2)
    expert = PickCubeExpert(env, ExpertConfig(move_timeout=0)); expert.reset()
    expert.state = ExpertState.MOVE_ABOVE_OBJECT
    expert.observe(env.get_info())
    assert expert.state is ExpertState.FAILED and expert.failure_category == "APPROACH_TIMEOUT"
    env.close()


def test_trajectory_recording_and_validation(tmp_path):
    env = make_env(); obs, info = env.reset(seed=11); expert = PickCubeExpert(env); expert.reset()
    rec = EpisodeRecorder(tmp_path, 0, make_episode_metadata(env, 11, "test action representation"))
    while not expert.done:
        action = expert.action()
        rec.append(obs, action, info, expert.state.value, info["task_success"])
        obs, _, terminated, truncated, info = env.step(action); expert.observe(info)
        rec.set_last_transition_outcome(success=bool(info["task_success"]))
        if terminated or truncated: break
    path = rec.save(success=bool(info["task_success"]), failure_reason=expert.failure_category)
    metadata, arrays = load_episode(path)
    assert path.exists() and metadata["success"] and arrays["rgb"].shape[0] == metadata["length"]
    assert not validate_dataset(tmp_path)
    env.close()


def test_validator_rejects_out_of_range_action(tmp_path):
    env = make_env(); obs, info = env.reset(seed=7)
    rec = EpisodeRecorder(tmp_path, 0, make_episode_metadata(env, 7, "test action representation"))
    rec.append(obs, np.r_[env.home_joint_pos, 1.0], info, "HOME")
    path = rec.save(success=False, failure_reason="test")
    action = np.load(path / "action.npy"); action[0, 0] = 99.0; np.save(path / "action.npy", action)
    assert any("arm actions outside joint limits" in error for error in validate_dataset(tmp_path))
    env.close()
