"""Targeted reverse-diffusion diagnostics for the TinyRDT overfit gate.

This utility intentionally evaluates an existing checkpoint only.  It never
trains, invokes MuJoCo, or changes a dataset.  Every metric is computed on raw
actions: five joint-position targets in radians followed by normalized gripper
opening.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.ml_dataset import ActionWindowDataset, NormalizationStats
from models.tiny_rdt import TinyRDT, TinyRDTConfig
from training.diffusion import ActionDiffusion


ACTION_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _checkpoint_ids(checkpoint: dict, checkpoint_path: Path) -> list[int]:
    ids = list(checkpoint["splits"]["train"])
    config_path = checkpoint_path.with_name("config.json")
    if config_path.exists():
        subset = json.loads(config_path.read_text()).get("subset_episodes")
        if subset:
            ids = ids[:int(subset)]
    return ids


def _batches(dataset: ActionWindowDataset, indices: list[int], batch_size: int = 8):
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0)


def _metric(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict:
    valid = mask.bool()[..., None].expand_as(target)
    error = (prediction - target)[valid].reshape(-1, target.shape[-1])
    return {
        "action_mae": float(error.abs().mean()),
        "action_mse": float(error.square().mean()),
        "per_action_mae": error.abs().mean(0).cpu().tolist(),
        "count": int(error.shape[0]),
    }


def _merge(metrics: list[dict]) -> dict:
    # Metrics carry their valid-token count, so short end windows cannot be
    # over-weighted by averaging per-window means.
    count = sum(m["count"] for m in metrics)
    result = {key: sum(m[key] * m["count"] for m in metrics) / count for key in ("action_mae", "action_mse")}
    result["per_action_mae"] = (np.asarray([m["per_action_mae"] for m in metrics]) * np.asarray([m["count"] for m in metrics])[:, None]).sum(0).tolist()
    result["per_action_mae"] = (np.asarray(result["per_action_mae"]) / count).tolist()
    result["count"] = count
    return result


def _prepare(batch, stats, device):
    return (
        batch["rgb"].to(device),
        stats.normalize_state(batch["state"].to(device)),
        batch["actions"].to(device),
        stats.normalize_action(batch["actions"].to(device)),
        batch["valid_mask"].to(device),
    )


def _sample_metrics(model, diffusion, loader, stats, device, sampler: str, steps: int | None, seed: int, prediction_type: str = "epsilon") -> dict:
    results = []
    for batch_index, batch in enumerate(loader):
        rgb, state, raw_actions, _, mask = _prepare(batch, stats, device)
        generator = torch.Generator(device=device).manual_seed(seed + batch_index)
        if sampler == "ddim":
            predicted = diffusion.sample_ddim(model, rgb, state, 16, 6, int(steps), generator, prediction_type)
        elif sampler == "ddpm":
            predicted = diffusion.sample_ddpm(model, rgb, state, 16, 6, generator, prediction_type)
        else:
            raise ValueError(sampler)
        results.append(_metric(stats.denormalize_action(predicted), raw_actions, mask))
    return _merge(results)


def _indices_by_phase(dataset: ActionWindowDataset, per_group: int) -> dict[str, list[int]]:
    groups = {"first_quarter": [], "second_quarter": [], "third_quarter": [], "last_quarter": [], "non_padded": [], "padded": []}
    for index, (episode_id, t) in enumerate(dataset.entries):
        n = len(dataset._episodes[episode_id]["action"])
        phase = min(3, 4 * t // n)
        groups[("first_quarter", "second_quarter", "third_quarter", "last_quarter")[phase]].append(index)
        (groups["non_padded"] if t + dataset.horizon <= n else groups["padded"]).append(index)
    # Evenly spaced choices avoid accidentally reporting a contiguous slice.
    selected = {}
    for key, values in groups.items():
        if not values:
            selected[key] = []
        else:
            picks = np.linspace(0, len(values) - 1, min(per_group, len(values)), dtype=int)
            selected[key] = [values[i] for i in picks]
    return selected


def _action_statistics(dataset_root: Path, ids: list[int], stats: NormalizationStats) -> dict:
    raw = np.concatenate([np.load(dataset_root / "episodes" / f"episode_{idx:06d}" / "action.npy") for idx in ids])
    normalized = (raw - np.asarray(stats.action_mean)) / np.asarray(stats.action_std)
    def one(values):
        return {name: {"min": float(values[:, i].min()), "max": float(values[:, i].max()), "mean": float(values[:, i].mean()), "std": float(values[:, i].std()), "p01": float(np.percentile(values[:, i], 1)), "p05": float(np.percentile(values[:, i], 5)), "p50": float(np.percentile(values[:, i], 50)), "p95": float(np.percentile(values[:, i], 95)), "p99": float(np.percentile(values[:, i], 99))} for i, name in enumerate(ACTION_NAMES)}
    return {"raw": one(raw), "normalized": one(normalized)}


@torch.no_grad()
def run(checkpoint_path: str, dataset_root: str, output: str, samples_per_group: int = 16) -> dict:
    started = time.monotonic()
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_file = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    stats = NormalizationStats(**checkpoint["normalization"])
    model = TinyRDT(TinyRDTConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    diffusion_config = checkpoint.get("diffusion_config", {"schedule": "linear", "prediction": "epsilon"})
    prediction_type = diffusion_config["prediction"]
    diffusion = ActionDiffusion.from_config(diffusion_config, device=device)
    ids = _checkpoint_ids(checkpoint, checkpoint_file)
    dataset = ActionWindowDataset(dataset_root, ids, horizon=16)
    groups = _indices_by_phase(dataset, samples_per_group)
    reference = groups["non_padded"][:samples_per_group]
    if not reference:
        raise RuntimeError("no non-padded action windows")

    methods = {}
    for steps in (5, 10, 20, 50, 100):
        methods[f"ddim_{steps}"] = _sample_metrics(model, diffusion, _batches(dataset, reference), stats, device, "ddim", steps, 4100, prediction_type)
    methods["ddpm_100"] = _sample_metrics(model, diffusion, _batches(dataset, reference), stats, device, "ddpm", None, 4100, prediction_type)

    # Teacher-forced direct x0 diagnostics.  Known noise makes this a pure
    # network-denoising measurement, independent of reverse accumulation.
    direct = {}
    diagnostic_indices = reference[:min(16, len(reference))]
    for t_value in (10, 25, 50, 75, 99):
        values = []
        for batch_index, batch in enumerate(_batches(dataset, diagnostic_indices)):
            rgb, state, raw, normalized, mask = _prepare(batch, stats, device)
            generator = torch.Generator(device=device).manual_seed(7300 + 100 * t_value + batch_index)
            noise = torch.randn(normalized.shape, generator=generator, device=device)
            t = torch.full((len(rgb),), t_value, device=device, dtype=torch.long)
            x_t, _ = diffusion.q_sample(normalized, t, noise)
            x0 = diffusion.model_x0(model(rgb, state, x_t, t, mask), x_t, t, prediction_type)
            values.append(_metric(stats.denormalize_action(x0), raw, mask))
        direct[str(t_value)] = _merge(values)

    # Full timestep bins identify high-noise failures hidden by aggregate loss.
    bins = []
    timestep_indices = reference[:min(8, len(reference))]
    for lo in range(0, diffusion.timesteps, 10):
        epsilon_errors = []; x0_errors = []
        for t_value in range(lo, lo + 10):
            values_eps = []; values_x0 = []
            for batch_index, batch in enumerate(_batches(dataset, timestep_indices)):
                rgb, state, raw, normalized, mask = _prepare(batch, stats, device)
                generator = torch.Generator(device=device).manual_seed(100_000 + t_value * 10 + batch_index)
                noise = torch.randn(normalized.shape, generator=generator, device=device)
                t = torch.full((len(rgb),), t_value, device=device, dtype=torch.long)
                x_t, _ = diffusion.q_sample(normalized, t, noise); output = model(rgb, state, x_t, t, mask)
                epsilon = diffusion.model_epsilon(output, x_t, t, prediction_type)
                valid = mask.bool()[..., None].expand_as(epsilon)
                values_eps.append(float((epsilon - noise)[valid].square().mean()))
                values_x0.append(_metric(stats.denormalize_action(diffusion.model_x0(output, x_t, t, prediction_type)), raw, mask)["action_mse"])
            epsilon_errors.append(float(np.mean(values_eps))); x0_errors.append(float(np.mean(values_x0)))
        bins.append({"timestep_range": f"{lo}-{lo + 9}", "epsilon_mse": float(np.mean(epsilon_errors)), "x0_action_mse": float(np.mean(x0_errors))})

    # Train/eval feature equality is a direct BatchNorm regression check.
    probe = next(iter(_batches(dataset, reference[:1])))
    rgb_probe, state_probe, _, normalized_probe, mask_probe = _prepare(probe, stats, device)
    model.train(); train_feature = model.vision(rgb_probe).clone(); train_eps = model(rgb_probe, state_probe, normalized_probe, torch.tensor([50], device=device), mask_probe).clone()
    model.eval(); eval_feature = model.vision(rgb_probe).clone(); eval_eps = model(rgb_probe, state_probe, normalized_probe, torch.tensor([50], device=device), mask_probe).clone()
    preprocessing = {"vision_feature_max_abs_difference_train_vs_eval": float((train_feature - eval_feature).abs().max()), "model_output_max_abs_difference_train_vs_eval": float((train_eps - eval_eps).abs().max()), "vision_feature_std": float(eval_feature.std())}

    # Conditioning ablation at a hard high-noise timestep.
    ablation_batch = next(iter(_batches(dataset, reference[:min(8, len(reference))])))
    rgb, state, _, normalized, mask = _prepare(ablation_batch, stats, device)
    t = torch.full((len(rgb),), 99, device=device, dtype=torch.long)
    noise = torch.randn(normalized.shape, generator=torch.Generator(device=device).manual_seed(9901), device=device)
    x_t, _ = diffusion.q_sample(normalized, t, noise)
    correct = model(rgb, state, x_t, t, mask)
    wrong_rgb = model(rgb.roll(1, 0), state, x_t, t, mask)
    wrong_state = model(rgb, state.roll(1, 0), x_t, t, mask)
    wrong_both = model(rgb.roll(1, 0), state.roll(1, 0), x_t, t, mask)
    conditioning = {"epsilon_mean_abs_change_wrong_rgb": float((correct - wrong_rgb).abs().mean()), "epsilon_mean_abs_change_wrong_state": float((correct - wrong_state).abs().mean()), "epsilon_mean_abs_change_wrong_both": float((correct - wrong_both).abs().mean())}

    # Repeatable DDIM means exact same output for the same initial noise.
    generator_a = torch.Generator(device=device).manual_seed(5151)
    generator_b = torch.Generator(device=device).manual_seed(5151)
    deterministic_a = diffusion.sample_ddim(model, rgb_probe, state_probe, 16, 6, 10, generator_a, prediction_type)
    deterministic_b = diffusion.sample_ddim(model, rgb_probe, state_probe, 16, 6, 10, generator_b, prediction_type)
    samples = torch.stack([diffusion.sample_ddim(model, rgb_probe, state_probe, 16, 6, 10, torch.Generator(device=device).manual_seed(6000 + k), prediction_type) for k in range(20)])
    determinism = {"same_seed_max_abs_difference": float((deterministic_a - deterministic_b).abs().max()), "twenty_initial_noises_sample_std_per_action": samples.std(0).mean((0, 1)).cpu().tolist()}

    phase = {name: _sample_metrics(model, diffusion, _batches(dataset, indices), stats, device, "ddim", 10, 8800, prediction_type) for name, indices in groups.items() if indices}
    action_stats = _action_statistics(Path(dataset_root), ids, stats)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([f"{b['timestep_range']}" for b in bins], [b["epsilon_mse"] for b in bins], marker="o", label="epsilon MSE")
    ax.plot([f"{b['timestep_range']}" for b in bins], [b["x0_action_mse"] for b in bins], marker="o", label="raw x0 action MSE")
    ax.set_xlabel("training timestep bin"); ax.set_ylabel("error"); ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(out / "timestep_errors.png", dpi=140); plt.close(fig)

    terminal = float(diffusion.alpha_bars[-1].cpu())
    report = {
        "checkpoint": str(checkpoint_file), "device": str(device), "episodes": ids,
        "terminal": {"alpha_bar_t99": terminal, "sqrt_alpha_bar_t99": terminal ** .5, "sqrt_one_minus_alpha_bar_t99": (1 - terminal) ** .5, "ddim_schedule_10": diffusion.ddim_schedule(10).cpu().tolist()},
        "sampled_methods": methods, "direct_x0": direct, "timestep_bins": bins,
        "phase_and_padding": phase, "conditioning": conditioning, "preprocessing": preprocessing,
        "determinism": determinism, "action_statistics": action_stats,
        "elapsed_s": time.monotonic() - started,
    }
    (out / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="artifacts/pickcube_smoke100_rgb160")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-group", type=int, default=16)
    args = parser.parse_args()
    result = run(args.checkpoint, args.dataset, args.output, args.samples_per_group)
    print(json.dumps({"output": args.output, "methods": result["sampled_methods"], "terminal": result["terminal"], "elapsed_s": result["elapsed_s"]}, indent=2))


if __name__ == "__main__":
    main()
