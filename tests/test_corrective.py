"""Legacy expert and counterfactual labels must reproduce CLEAN10 bit-exactly."""
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "artifacts/pickcube_smoke100_rgb160/episodes/episode_000000"
pytestmark = pytest.mark.skipif(not ROOT.exists(), reason="CLEAN10 dataset artifact not present")


def test_legacy_expert_and_counterfactual_chunks_reproduce_demo_exactly():
    from simulation.env import PickCubeConfig, SO101PickCubeEnv
    from simulation.legacy_expert import LegacyPickCubeExpert
    from data.corrective import expert_chunk
    recorded = np.load(ROOT / "action.npy"); joints = np.load(ROOT / "joint_pos.npy")
    env = SO101PickCubeEnv(PickCubeConfig(render_observations=False))
    obs, _ = env.reset(seed=3000); expert = LegacyPickCubeExpert(env); expert.reset(); t = 0
    while not expert.done:
        chunk, n, ok = expert_chunk(env, expert, 16)
        assert ok and n == min(16, len(recorded) - t)
        assert np.array_equal(chunk[:n], recorded[t:t + n])
        assert np.array_equal(obs["joint_pos"], joints[t])
        obs, _, _, _, info = env.step(expert.action()); expert.observe(info); t += 1
    assert t == len(recorded) and expert.state.value == "SUCCESS"
