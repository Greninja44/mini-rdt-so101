# CLAUDE_HANDOFF_AUDIT — repository state at takeover (2026-09-18)

This records what was actually found, before any change of mine. For what happened afterwards, see
`docs/research/research_log.md`.

## 1. Repository and environment

- Path: `/home/batman/mini-rdt-so101`, branch `main`, tracking `origin/main`.
- **One commit** (`037c402 Initial MiniRDT-SO101 simulator and diffusion diagnostics`); the working tree was clean.
- `.gitignore` excludes `artifacts/`, so **no dataset, checkpoint, log or plot is versioned**. They exist only on this disk.
- `ASTRA_FINDINGS.md` / `MINIRDT_RESEARCH_AUDIT.md`: **not present**. The previous audit exists only as code
  (`evaluation/research_audit.py`, `training/audit_overfit.py`, `evaluation/{vision_probe,data_forensics,architecture_budget}.py`)
  plus outputs in `artifacts/research_audit/`, with a short note in `training/README.md`.
- `artifacts/research_audit/original_source/` is a pre-audit snapshot of the source, with hashes in `original_source_sha256.json`.
  Compared with it, the current code changed `training/diffusion.py`, `training/policy.py`, `data/ml_dataset.py`, `data/collect.py`,
  `data/dataset.py`, `evaluation/{diffusion_diagnostics,tiny_rdt_offline}.py` and added the audit scripts and two test files.
- Python 3.12 in `.venv` (uv). torch **2.14.0+cu126**, **CUDA available** (RTX 4050 Laptop GPU, about 6 GB), MuJoCo 3.x.
  The host has 7.8 GB RAM, so at most about 2 parallel MuJoCo+torch workers.
- MuJoCo rendering (EGL, 160×120) costs about 1 s per control step here. This dominates closed-loop evaluation time.

## 2. Files named in the handoff

| Path | Present |
|---|---|
| `data/ml_dataset.py` | yes (now also has `hold_terminal_actions`) |
| `models/tiny_rdt.py` | yes |
| `training/diffusion.py` | yes (supports linear/cosine schedules and ε/x0 prediction, DDIM, DDPM, oracle) |
| `training/trainer.py` | yes (the original Phase-2 trainer) |
| `training/policy.py` | yes (single-observation inference, no closed-loop runner) |
| `evaluation/tiny_rdt_offline.py` | yes |
| `tests/test_phase2.py` | yes |

There was **no MuJoCo closed-loop policy runner**.

## 3. Actual configuration vs the handoff

| Item | Handoff | Found |
|---|---|---|
| Dataset | 100 eps, 80/10/10, split seed 17 | `artifacts/pickcube_smoke100_rgb160`: 100 eps, 4826 frames, 80/10/10 (3866/485/475 frames), lengths 46–51 (mean 48.3). Forensics: 0 validation errors |
| Overfit ids | 10 demos | `[0,2,3,4,5,6,7,8,9,11]` (first 10 train ids, seed 17), 483 windows, 150 padded |
| TinyRDT | 192/4/6, H=16, 2,009,670 trainable / 927,008 frozen | matches |
| Vision | frozen MobileNetV3-S | frozen, ImageNet-normalised RGB/255, BN in eval, **global average pooled to 1×576** (map 576×4×5) |
| Diffusion | linear 1e-4→0.02, T=100, ε, DDIM-10 | original config as stated. `configs/tiny_rdt_phase2.json` still describes it (stale; also lists lr 3e-4) |
| Overfit run | 5000 steps, bs 8, lr 1e-3 | `artifacts/tiny_rdt_overfit10/config.json` matches |
| Handoff metric "10-step MAE 0.146 / MSE 0.0549, elbow 0.33" | | `tiny_rdt_overfit10/offline_5000`: MAE 0.144, MSE 0.049, elbow 0.326, measured on only 8 windows. Consistent |
| Tests | 15 pass | **30 pass** (`tests/test_diffusion_oracles.py` and `tests/test_terminal_padding.py` added) |

## 4. Work already done by the previous agent but not written up

Controlled 10-demo, 5000-step, seed-17 comparisons (`training/audit_overfit.py`, cached frozen features, independent RNG streams),
each diagnosed by `evaluation/research_audit.py`:

| Run | Sampled MAE (all windows) | Note |
|---|---|---|
| `original_linear` (the handoff checkpoint) | 0.081 | ε, linear |
| `linear_epsilon` (retrained) | 0.102 | |
| `cosine_epsilon` | **48.8** (diverged) | ε-param at zero terminal SNR |
| `cosine_x0` | 0.027 | last quarter 0.097 (masked padding) |
| `cosine_x0_hold` | 0.018 | hold-last-action padding |
| `bc_rgb` / `bc_state` / `bc_privileged` | 0.022 / 0.025 / 0.020 | deterministic MLP, same windows |

It also pre-registered `gate_definition.json`: all-window MAE ≤ 0.02, each arm joint ≤ 0.03 rad, gripper ≤ 0.03, each
quarter ≤ 0.025, episode start ≤ 0.025, sampler seeds 4100–4102. Its note says "final dense gate evaluated separately",
and that evaluation **had not been run**.

## 5. Datasets and checkpoints available

- Datasets: `artifacts/pickcube_smoke100_rgb160` (main); small smoke/round-trip sets (`smoke_dataset`, `rgb_roundtrip_seed1000`,
  `rgb_160_roundtrip_seed2000`, `parallel_render_probe`).
- Checkpoints: `tiny_rdt_overfit10/{last,tiny_rdt_best}.pt` (original, also the frozen-encoder source for the audit),
  `tiny_rdt_overfit10_cosine_epsilon/`, `tiny_rdt_overfit_probe/`, and `research_audit/<run>/{best,last}.pt` for every run above.

## 6. Unresolved blocker at takeover

The pre-registered gate had never been evaluated on the winning configuration (cosine + x0 + hold). Checked mechanically
(`evaluation/gate_check.py`), the 5000-step `cosine_x0_hold` **fails**: worst joint 0.034, episode start 0.029–0.030,
last quarter 0.025. Closed-loop MuJoCo execution had never been attempted.
