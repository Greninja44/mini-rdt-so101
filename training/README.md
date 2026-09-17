# Reserved for Phase 2+

Training intentionally begins only after the Phase 1 expert and data format are approved.
## Diffusion diagnostic contract

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
