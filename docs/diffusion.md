# Diffusion formulation

The validated configuration is the **cosine schedule** (Nichol & Dhariwal; terminal ᾱ ≈ 2.4e-7), **x0-prediction**, hold-last-action padding
and EMA 0.999, sampled with deterministic DDIM (10 steps).

Why:
- The original 100-step linear schedule leaves ᾱ₉₉ ≈ 0.364 of signal at the last training step, while sampling starts from N(0, I).
- Switching to cosine with ε-prediction diverges: x0 = (x − √(1−ᾱ)ε̂)/√ᾱ amplifies ε error by ×4.1e6 at zero SNR.
- x0-prediction avoids that division.

Implementation: `training/diffusion.py`. Tests: `tests/test_diffusion_oracles.py`.

## Equations and conventions

Timestep indices are zero-based: `t in [0, 99]`, and the deterministic
10-step DDIM schedule is `[99, 88, 77, 66, 55, 44, 33, 22, 11, 0]`.

For `beta_t`, `alpha_t = 1 - beta_t`, and
`alpha_bar_t = product_{s=0}^t alpha_s`, the forward process is:

`x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) epsilon`, where
`epsilon ~ N(0, I)`.

The original TinyRDT checkpoint predicts `epsilon`; its direct reconstruction
is:

`x0_hat = (x_t - sqrt(1 - alpha_bar_t) epsilon_hat) / sqrt(alpha_bar_t)`.

Reference ancestral DDPM uses:

`mean_t = (x_t - beta_t epsilon_hat / sqrt(1-alpha_bar_t)) / sqrt(alpha_t)`

and variance `beta_t (1-alpha_bar_{t-1}) / (1-alpha_bar_t)` for `t > 0`.
Deterministic DDIM (`eta=0`) transitions from `t` to `s < t` with:

`x_s = sqrt(alpha_bar_s) x0_hat + sqrt(1-alpha_bar_s) epsilon_hat`.

`training.diffusion.ActionDiffusion.oracle_ddim` is the permanent exact-noise
test for these equations.  The original 100-step linear schedule has
`alpha_bar_99 ~= 0.36356`; therefore its trained terminal distribution is not
near pure normal noise, despite ordinary sampling starting from `N(0,I)`.
The optional cosine diagnostic schedule instead has terminal alpha-bar close
to zero.  These are diagnostic configurations only; neither is authorization
to start full-data training.
