"""DDPM epsilon-prediction process for normalized action chunks."""
from __future__ import annotations
import torch


class ActionDiffusion:
    """q(a_t|a_0)=sqrt(alpha_bar_t)a_0 + sqrt(1-alpha_bar_t) epsilon.

    The network predicts epsilon; training uses masked epsilon MSE.
    """
    def __init__(self, timesteps: int = 100, beta_start: float = 1e-4, beta_end: float = .02, schedule: str = "linear", device: torch.device | str = "cpu"):
        if timesteps < 1 or not 0 < beta_start <= beta_end < 1:
            raise ValueError("need positive timesteps and 0 < beta_start <= beta_end < 1")
        self.timesteps = timesteps; self.schedule = schedule
        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        elif schedule == "cosine":
            # Nichol & Dhariwal cosine cumulative-alpha schedule.  Unlike the
            # 100-step linear schedule, its terminal alpha-bar is near zero,
            # so x_T is actually close to N(0, I), matching normal sampling.
            s = .008
            indices = torch.arange(timesteps + 1, device=device, dtype=torch.float32)
            target_alpha_bars = torch.cos(((indices / timesteps + s) / (1 + s)) * torch.pi / 2).square()
            target_alpha_bars = target_alpha_bars / target_alpha_bars[0]
            self.betas = (1 - target_alpha_bars[1:] / target_alpha_bars[:-1]).clamp(max=.999)
        else:
            raise ValueError(f"unknown schedule: {schedule}")
        self.alphas = 1 - self.betas; self.alpha_bars = torch.cumprod(self.alphas, 0)

    @classmethod
    def from_config(cls, config: dict | None, device="cpu"):
        """Restore the full noise process, including legacy Phase-2 checkpoints."""
        config = config or {}
        schedule = config.get("schedule", "linear")
        if schedule == "linear beta 1e-4..0.02":
            schedule = "linear"
        return cls(timesteps=config.get("timesteps", 100),
                   beta_start=config.get("beta_start", 1e-4),
                   beta_end=config.get("beta_end", .02), schedule=schedule, device=device)

    def _extract(self, x: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        return x.gather(0, t).reshape((shape[0],) + (1,) * (len(shape) - 1))

    def q_sample(self, clean: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(clean) if noise is None else noise; ab = self._extract(self.alpha_bars, t, clean.shape)
        return ab.sqrt() * clean + (1 - ab).sqrt() * noise, noise

    def ddim_schedule(self, steps: int) -> torch.Tensor:
        """Descending indices: T-1 through 0, or just T-1 for one step."""
        if not 1 <= steps <= self.timesteps: raise ValueError("steps must lie in [1, timesteps]")
        schedule = torch.linspace(self.timesteps - 1, 0, steps, device=self.betas.device).round().long()
        if len(torch.unique_consecutive(schedule)) != len(schedule): raise RuntimeError("DDIM schedule has duplicate timesteps")
        return schedule

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        ab = self._extract(self.alpha_bars, t, x_t.shape)
        return (x_t - (1 - ab).sqrt() * epsilon) / ab.sqrt()

    def predict_epsilon(self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """Invert ``predict_x0`` for an x0-parameterized denoiser."""
        ab = self._extract(self.alpha_bars, t, x_t.shape)
        return (x_t - ab.sqrt() * x0) / (1 - ab).sqrt().clamp_min(1e-12)

    def model_x0(self, model_output: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, prediction_type: str) -> torch.Tensor:
        if prediction_type == "epsilon":
            return self.predict_x0(x_t, t, model_output)
        if prediction_type == "x0":
            return model_output
        raise ValueError(f"unknown prediction type: {prediction_type}")

    def model_epsilon(self, model_output: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, prediction_type: str) -> torch.Tensor:
        if prediction_type == "epsilon":
            return model_output
        if prediction_type == "x0":
            return self.predict_epsilon(x_t, t, model_output)
        raise ValueError(f"unknown prediction type: {prediction_type}")

    def posterior(self, x0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        """Exact q(x_{t-1}|x_t,x0); virtual alpha_bar[-1]=1 is clean data.

        At t=0 return x0 exactly. Keeping the x0 form avoids subtracting two
        nearly equal noise terms for an x0-parameterized model near zero SNR.
        """
        ab = self._extract(self.alpha_bars, t, x_t.shape)
        previous = torch.cat((self.alpha_bars.new_ones(1), self.alpha_bars[:-1]))
        ab_prev = self._extract(previous, t, x_t.shape)
        beta = self._extract(self.betas, t, x_t.shape)
        alpha = self._extract(self.alphas, t, x_t.shape)
        # Equivalent to 1-alpha_bar[t], but avoids cancellation of two
        # independently rounded cumulative products at the first few steps.
        denominator = alpha * (1 - ab_prev) + beta
        mean = ab_prev.sqrt() * beta / denominator * x0 + alpha.sqrt() * (1 - ab_prev) / denominator * x_t
        variance = beta * (1 - ab_prev) / denominator
        terminal = (t == 0).reshape((-1,) + (1,) * (x_t.ndim - 1))
        return torch.where(terminal, x0, mean), variance

    def _initial(self, state, horizon, action_dim, generator, initial_noise):
        shape = (state.shape[0], horizon, action_dim)
        if initial_noise is None:
            return torch.randn(shape, device=state.device, dtype=state.dtype, generator=generator)
        if initial_noise.shape != shape or initial_noise.device != state.device or initial_noise.dtype != state.dtype:
            raise ValueError("initial_noise must match sampling shape, device, and dtype")
        return initial_noise.clone()

    @torch.no_grad()
    def sample_ddim(self, model, rgb: torch.Tensor, state: torch.Tensor, horizon: int, action_dim: int, steps: int = 10, generator: torch.Generator | None = None, prediction_type: str = "epsilon", *, initial_noise: torch.Tensor | None = None) -> torch.Tensor:
        """Deterministic eta=0 DDIM; initial_noise is a diagnostic override."""
        x = self._initial(state, horizon, action_dim, generator, initial_noise)
        schedule = self.ddim_schedule(steps)
        for i, t_scalar in enumerate(schedule):
            t = torch.full((x.shape[0],), int(t_scalar), device=x.device, dtype=torch.long); output = model(rgb, state, x, t)
            x0 = self.model_x0(output, x, t, prediction_type); eps = self.model_epsilon(output, x, t, prediction_type)
            if i + 1 == len(schedule): x = x0; break
            next_ab = self.alpha_bars[schedule[i + 1]]
            x = next_ab.sqrt() * x0 + (1 - next_ab).sqrt() * eps
        return x

    @torch.no_grad()
    def sample_ddpm(self, model, rgb: torch.Tensor, state: torch.Tensor, horizon: int, action_dim: int, generator: torch.Generator | None = None, prediction_type: str = "epsilon", *, initial_noise: torch.Tensor | None = None) -> torch.Tensor:
        """Reference ancestral DDPM sampler using all training timesteps."""
        x = self._initial(state, horizon, action_dim, generator, initial_noise)
        for t_scalar in range(self.timesteps - 1, -1, -1):
            t = torch.full((x.shape[0],), t_scalar, device=x.device, dtype=torch.long); output = model(rgb, state, x, t)
            x0 = self.model_x0(output, x, t, prediction_type)
            mean, variance = self.posterior(x0, x, t)
            if t_scalar == 0: x = mean
            else:
                x = mean + variance.sqrt() * torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        return x

    @torch.no_grad()
    def oracle_ddim(self, clean: torch.Tensor, steps: int, noise: torch.Tensor | None = None) -> torch.Tensor:
        """DDIM reverse path with exact epsilon(x_t,x0), for mathematical tests."""
        schedule = self.ddim_schedule(steps); start = torch.full((len(clean),), int(schedule[0]), device=clean.device, dtype=torch.long)
        x, noise = self.q_sample(clean, start, noise)
        for i, t_scalar in enumerate(schedule):
            t = torch.full((len(clean),), int(t_scalar), device=clean.device, dtype=torch.long)
            eps = (x - self._extract(self.alpha_bars, t, x.shape).sqrt() * clean) / (1 - self._extract(self.alpha_bars, t, x.shape)).sqrt()
            x0 = self.predict_x0(x, t, eps)
            if i + 1 == len(schedule): return x0
            next_ab = self.alpha_bars[schedule[i + 1]]; x = next_ab.sqrt() * x0 + (1 - next_ab).sqrt() * eps
        return x

    def sample(self, model, rgb, state, horizon, action_dim, steps: int = 10, generator=None, prediction_type: str = "epsilon") -> torch.Tensor:
        return self.sample_ddim(model, rgb, state, horizon, action_dim, steps, generator, prediction_type)
