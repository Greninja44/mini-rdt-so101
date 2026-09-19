from __future__ import annotations
from pathlib import Path
import pytest
import torch

from data.ml_dataset import ActionWindowDataset, compute_normalization, make_episode_splits
from models.tiny_rdt import SinusoidalTimeEmbedding, TinyRDT, TinyRDTConfig
from models.tiny_bc import TinyActionBC
from training.diffusion import ActionDiffusion

DATASET = "artifacts/pickcube_smoke100_rgb160"
# Datasets are not versioned (see docs/artifact_manifest.json); dataset tests skip in a fresh clone.
needs_dataset = pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / DATASET / "episodes").exists(), reason="dataset artifact not present")


class _ZeroDenoiser(torch.nn.Module):
    def forward(self, rgb, state, noisy_actions, timestep):
        return torch.zeros_like(noisy_actions)


@needs_dataset
def test_episode_splits_are_disjoint_and_deterministic():
    a = make_episode_splits(DATASET, seed=19); b = make_episode_splits(DATASET, seed=19)
    assert a == b and len(a.train) == 80 and len(a.validation) == len(a.test) == 10
    assert not (set(a.train) & set(a.validation) or set(a.train) & set(a.test) or set(a.validation) & set(a.test))


@needs_dataset
def test_windows_never_cross_episode_and_pad_with_mask():
    splits = make_episode_splits(DATASET); dataset = ActionWindowDataset(DATASET, splits.train[:1], horizon=16)
    last = dataset[len(dataset) - 1]
    assert last["actions"].shape == (16, 6) and last["valid_mask"].sum() == 1
    assert torch.all(last["actions"][1:] == 0)


@needs_dataset
def test_normalization_round_trip_and_action_contract():
    splits = make_episode_splits(DATASET); stats = compute_normalization(DATASET, splits.train[:2]); dataset = ActionWindowDataset(DATASET, splits.train[:1])
    state, action = dataset[0]["state"], dataset[0]["actions"]
    assert torch.allclose(stats.denormalize_state(stats.normalize_state(state)), state, atol=1e-6)
    assert torch.allclose(stats.denormalize_action(stats.normalize_action(action)), action, atol=1e-6)
    assert action.shape[-1] == 6 and torch.all((action[:, 5] >= 0) & (action[:, 5] <= 1))


def test_timestep_embedding_and_diffusion_forward_process():
    emb = SinusoidalTimeEmbedding(192)(torch.tensor([0, 1, 2])); assert emb.shape == (3, 192) and torch.isfinite(emb).all()
    diffusion = ActionDiffusion(); clean = torch.zeros(2, 16, 6); noisy, noise = diffusion.q_sample(clean, torch.tensor([0, 99]))
    assert noisy.shape == noise.shape == clean.shape and torch.isfinite(noisy).all()


def test_forward_equation_and_exact_x0_reconstruction_at_boundaries():
    diffusion = ActionDiffusion(); clean = torch.randn(2, 16, 6); noise = torch.randn_like(clean)
    for timestep in (0, 99):
        t = torch.full((2,), timestep, dtype=torch.long)
        noisy, returned_noise = diffusion.q_sample(clean, t, noise)
        alpha_bar = diffusion.alpha_bars[timestep]
        expected = alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise
        assert torch.allclose(noisy, expected, atol=1e-6)
        assert returned_noise is noise
        assert torch.allclose(diffusion.predict_x0(noisy, t, noise), clean, atol=1e-6)


def test_ddim_schedule_and_exact_oracle_reverse_path():
    diffusion = ActionDiffusion(); clean = torch.randn(3, 16, 6); noise = torch.randn_like(clean)
    assert diffusion.ddim_schedule(10).tolist() == [99, 88, 77, 66, 55, 44, 33, 22, 11, 0]
    for steps in (5, 10, 20, 50, 100):
        reconstructed = diffusion.oracle_ddim(clean, steps, noise)
        assert torch.allclose(reconstructed, clean, atol=1e-5)


def test_model_forward_parameter_count_and_sampling():
    config = TinyRDTConfig(pretrained_vision=False); model = TinyRDT(config); counts = model.parameter_counts()
    assert 2_000_000 <= counts["trainable"] <= 5_000_000
    rgb = torch.zeros(1, 3, 120, 160); state = torch.zeros(1, 6); actions = torch.zeros(1, 16, 6); out = model(rgb, state, actions, torch.tensor([3]))
    assert out.shape == actions.shape and torch.isfinite(out).all()
    sample = ActionDiffusion().sample(model, rgb, state, 16, 6, steps=3, generator=torch.Generator().manual_seed(0))
    assert sample.shape == actions.shape and torch.isfinite(sample).all()
    # A tiny analytic stand-in keeps the all-100-step reference-sampler test
    # focused on scheduler shape/finite values rather than vision throughput.
    ddpm = ActionDiffusion().sample_ddpm(_ZeroDenoiser(), rgb, state, 16, 6, generator=torch.Generator().manual_seed(0))
    assert ddpm.shape == actions.shape and torch.isfinite(ddpm).all()


def test_deterministic_ddim_and_attention_mask_isolate_padding():
    model = TinyRDT(TinyRDTConfig(pretrained_vision=False)).eval(); diffusion = ActionDiffusion()
    rgb = torch.zeros(1, 3, 120, 160); state = torch.zeros(1, 6)
    a = diffusion.sample_ddim(_ZeroDenoiser(), rgb, state, 16, 6, 10, torch.Generator().manual_seed(44))
    b = diffusion.sample_ddim(_ZeroDenoiser(), rgb, state, 16, 6, 10, torch.Generator().manual_seed(44))
    assert torch.equal(a, b)
    actions = torch.zeros(1, 16, 6); changed = actions.clone(); changed[:, 8:] = 999.0
    mask = torch.tensor([[True] * 8 + [False] * 8])
    t = torch.tensor([30])
    assert torch.allclose(model(rgb, state, actions, t, mask)[:, :8], model(rgb, state, changed, t, mask)[:, :8], atol=1e-6)


def test_frozen_vision_preprocessing_is_identical_train_and_eval():
    model = TinyRDT(TinyRDTConfig(pretrained_vision=False)); rgb = torch.randint(0, 256, (1, 3, 120, 160), dtype=torch.uint8)
    model.train(); train_features = model.vision(rgb)
    model.eval(); eval_features = model.vision(rgb)
    assert torch.equal(train_features, eval_features)


def test_cosine_terminal_noise_and_bc_output_contract():
    assert ActionDiffusion(schedule="cosine").alpha_bars[-1] < 1e-5
    bc = TinyActionBC(pretrained_vision=False)
    assert bc(torch.zeros(1, 3, 120, 160), torch.zeros(1, 6)).shape == (1, 16, 6)


def test_checkpoint_round_trip(tmp_path):
    model = TinyRDT(TinyRDTConfig(pretrained_vision=False)); path = tmp_path / "checkpoint.pt"; torch.save({"model": model.state_dict(), "model_config": model.config.__dict__}, path)
    loaded = torch.load(path, weights_only=False); restored = TinyRDT(TinyRDTConfig(**loaded["model_config"])); restored.load_state_dict(loaded["model"])
    assert restored.parameter_counts() == model.parameter_counts()
