"""Analytic tests of the production reverse paths, not a duplicate sampler."""
import pytest
import torch
from training.diffusion import ActionDiffusion


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_known_epsilon_reconstruction_all_times(schedule):
    d = ActionDiffusion(schedule=schedule)
    g = torch.Generator().manual_seed(17)
    clean = torch.randn(100, 16, 6, generator=g)
    noise = torch.randn(clean.shape, generator=g)
    t = torch.arange(100)
    x, _ = d.q_sample(clean, t, noise)
    recovered = d.predict_x0(x, t, noise)
    # Floating-point error scales as 1/sqrt(alpha_bar), not a fixed 1e-7.
    bound = 8 * torch.finfo(x.dtype).eps / d.alpha_bars.sqrt()
    assert ((recovered-clean).abs().amax((1, 2)) < bound).all()


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
@pytest.mark.parametrize("prediction", ["epsilon", "x0"])
def test_production_samplers_with_point_mass_oracle(schedule, prediction):
    d = ActionDiffusion(schedule=schedule)
    clean = torch.randn(3, 16, 6, generator=torch.Generator().manual_seed(51))
    state = torch.zeros(3, 6)
    def oracle(rgb, state, x, t):
        return d.predict_epsilon(x, t, clean) if prediction == "epsilon" else clean
    for steps in (1, 5, 10, 20, 50, 100):
        out = d.sample_ddim(oracle, None, state, 16, 6, steps, torch.Generator().manual_seed(19), prediction)
        assert torch.allclose(out, clean, atol=5e-4 if steps == 1 else 2e-6)
    out = d.sample_ddpm(oracle, None, state, 16, 6, torch.Generator().manual_seed(19), prediction)
    assert torch.allclose(out, clean, atol=2e-6)


def test_posterior_matches_independent_gaussian_conditioning():
    d = ActionDiffusion()
    t = torch.tensor([0, 1, 30, 99])
    x0 = torch.ones(4, 2, 1, dtype=torch.float64)
    xt = 2 * x0
    mean, variance = d.posterior(x0, xt, t)
    for i, ti in enumerate(t):
        a = float(d.alphas[ti]); b = float(d.betas[ti])
        prev = float(d.alpha_bars[ti-1]) if ti else 1.
        # Gaussian observation x_t = sqrt(a) x_prev + sqrt(b) z.
        denom = a * (1-prev) + b
        gain = a**.5 * (1-prev) / denom
        expected_mean = prev**.5 + gain * (2 - (a*prev)**.5)
        expected_var = (1-prev) * b / denom
        assert torch.allclose(mean[i], torch.full_like(mean[i], expected_mean), atol=2e-6)
        assert abs(float(variance[i]) - expected_var) < 2e-7
    assert torch.equal(mean[0], x0[0]) and variance[0] == 0


def test_gaussian_optimal_score_exposes_terminal_prior_mismatch():
    # x0|c ~ N(mu, 1). This is the exact conditional epsilon mean, not an
    # oracle allowed to see the target x0. A wrong terminal prior still biases
    # the DDPM mean by alpha_bar_terminal * mu, even with a perfect score.
    mu = 2.
    for schedule in ("linear", "cosine"):
        d = ActionDiffusion(schedule=schedule)
        n = 16384
        def optimal(rgb, state, x, t):
            ab = d._extract(d.alpha_bars, t, x.shape)
            return (1-ab).sqrt() * (x-ab.sqrt()*mu)
        samples = d.sample_ddpm(optimal, None, torch.zeros(n, 1), 1, 1, torch.Generator().manual_seed(117))
        expected = mu * (1-float(d.alpha_bars[-1]))
        assert abs(float(samples.mean())-expected) < .03
        if schedule == "linear":
            assert abs(float(samples.mean())-mu) > .65


def test_restore_full_schedule_and_invalid_steps():
    d = ActionDiffusion.from_config({"timesteps": 73, "beta_start": .002, "beta_end": .03, "schedule": "linear"})
    assert len(d.betas) == 73 and float(d.betas[0]) == pytest.approx(.002)
    for steps in (0, 74):
        with pytest.raises(ValueError): d.ddim_schedule(steps)
    assert d.ddim_schedule(1).tolist() == [72]
