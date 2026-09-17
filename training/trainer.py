"""Reproducible Phase-2 TinyRDT trainer."""
from __future__ import annotations
import argparse, json, random, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from data.ml_dataset import ActionWindowDataset, compute_normalization, make_episode_splits
from models.tiny_rdt import TinyRDT, TinyRDTConfig
from .diffusion import ActionDiffusion


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def device_for_training() -> torch.device: return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask[..., None]
    return ((prediction - target).square() * weight).sum() / (weight.sum() * prediction.shape[-1]).clamp_min(1)


def _prepare(batch: dict[str, torch.Tensor], stats, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb = batch["rgb"].to(device, non_blocking=True); state = stats.normalize_state(batch["state"].to(device)); actions = stats.normalize_action(batch["actions"].to(device)); mask = batch["valid_mask"].to(device)
    return rgb, state, actions, mask


@torch.no_grad()
def build_vision_cache(model, datasets, device) -> dict[tuple[int, int], torch.Tensor]:
    """Precompute fixed MobileNet features once; no image augmentation exists."""
    cache: dict[tuple[int, int], torch.Tensor] = {}
    model.eval()
    for dataset in datasets:
        for batch in DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0):
            keys = [(int(episode), int(t)) for episode, t in zip(batch["episode_id"].tolist(), batch["t"].tolist())]
            if all(key in cache for key in keys):
                continue
            rgb = batch["rgb"].to(device); features = model.vision(rgb).cpu()
            for key, feature in zip(keys, features): cache[key] = feature
    return cache


def vision_features_for(batch, cache, device):
    return torch.stack([cache[(int(episode), int(t))] for episode, t in zip(batch["episode_id"].tolist(), batch["t"].tolist())]).to(device)


def build_preloaded_bank(dataset, stats, vision_cache, device):
    """Move the bounded overfit subset into RAM/GPU once, byte-for-byte."""
    states = []; actions = []; masks = []; features = []
    for index, (episode, t) in enumerate(dataset.entries):
        item = dataset[index]
        states.append(item["state"]); actions.append(item["actions"]); masks.append(item["valid_mask"])
        features.append(vision_cache[(episode, t)])
    return {
        "state": stats.normalize_state(torch.stack(states).to(device)),
        "actions": stats.normalize_action(torch.stack(actions).to(device)),
        "mask": torch.stack(masks).to(device),
        "features": torch.stack(features).to(device),
    }


@torch.no_grad()
def evaluate_bank(model, bank, stats, diffusion, device, prediction_type: str) -> dict[str, float]:
    model.eval(); losses = []; action_mse = []
    for start in range(0, len(bank["state"]), 128):
        sl = slice(start, start + 128); state, actions, mask, features = (bank[k][sl] for k in ("state", "actions", "mask", "features"))
        t = torch.full((len(state),), diffusion.timesteps // 2, device=device, dtype=torch.long)
        noisy, noise = diffusion.q_sample(actions, t); output = model(None, state, noisy, t, mask, features)
        target = noise if prediction_type == "epsilon" else actions
        losses.append(masked_mse(output, target, mask).item())
        x0 = diffusion.model_x0(output, noisy, t, prediction_type)
        action_mse.append(masked_mse(stats.denormalize_action(x0), stats.denormalize_action(actions), mask).item())
    return {"epsilon_mse": float(np.mean(losses)), "action_mse": float(np.mean(action_mse))}


@torch.no_grad()
def evaluate(model, loader, stats, diffusion, device, prediction_type: str = "epsilon", vision_cache=None) -> dict[str, float]:
    model.eval(); losses=[]; action_mse=[]
    for batch in loader:
        rgb, state, actions, mask = _prepare(batch, stats, device); t = torch.full((len(rgb),), diffusion.timesteps // 2, device=device, dtype=torch.long)
        features = vision_features_for(batch, vision_cache, device) if vision_cache is not None else None
        noisy, noise = diffusion.q_sample(actions, t); output = model(rgb, state, noisy, t, mask, features)
        target = noise if prediction_type == "epsilon" else actions
        losses.append(masked_mse(output, target, mask).item())
        x0 = diffusion.model_x0(output, noisy, t, prediction_type)
        action_mse.append(masked_mse(stats.denormalize_action(x0), stats.denormalize_action(actions), mask).item())
    return {"epsilon_mse": float(np.mean(losses)), "action_mse": float(np.mean(action_mse))}


def train(dataset_root: str, output: str, seed: int = 17, steps: int = 1000, batch_size: int = 8, learning_rate: float = 3e-4, subset_episodes: int | None = None, resume: str | None = None, pretrained_vision: bool = True, overfit_validation: bool = False, schedule: str = "linear", prediction_type: str = "epsilon", validation_interval: int = 50) -> Path:
    """Train a tiny-overfit or complete experiment; returns best checkpoint path."""
    set_seed(seed); root = Path(dataset_root); out = Path(output); out.mkdir(parents=True, exist_ok=True)
    splits = make_episode_splits(root, seed); splits.save(out / "splits.json")
    train_ids = splits.train[:subset_episodes] if subset_episodes else splits.train
    if not train_ids: raise ValueError("empty training subset")
    stats = compute_normalization(root, train_ids); stats.save(out / "normalization.json")
    horizon = 16
    # The deliberate 10-demo overfit gate is only tens of MB.  Keeping it in
    # RAM eliminates memmap page-fault noise while retaining byte-identical
    # recorded arrays.  Full-data training remains memory-mapped.
    cache_subset = subset_episodes is not None
    train_ds = ActionWindowDataset(root, train_ids, horizon, cache_in_memory=cache_subset)
    val_ds = ActionWindowDataset(root, train_ids if overfit_validation else splits.validation, horizon, cache_in_memory=cache_subset)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    if prediction_type not in ("epsilon", "x0"): raise ValueError("prediction_type must be epsilon or x0")
    device = device_for_training(); config = TinyRDTConfig(pretrained_vision=pretrained_vision); model = TinyRDT(config).to(device); diffusion = ActionDiffusion(schedule=schedule, device=device)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate, weight_decay=1e-4)
    use_amp = device.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_step = 0; best = float("inf")
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False); model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"]); start_step = checkpoint["step"] + 1; best = checkpoint.get("best_validation", best)
    diffusion_config = {"timesteps": 100, "schedule": schedule, "beta_start": 1e-4, "beta_end": .02, "prediction": prediction_type, "sampling_steps": 10}
    (out / "config.json").write_text(json.dumps({"seed": seed, "dataset_root": str(root), "steps": steps, "batch_size": batch_size, "learning_rate": learning_rate, "subset_episodes": subset_episodes, "overfit_validation": overfit_validation, "validation_interval": validation_interval, "model": asdict(config), "diffusion": diffusion_config, "parameter_count": model.parameter_counts()}, indent=2) + "\n")
    cache_path = out / "frozen_vision_features.pt"
    vision_cache = torch.load(cache_path, map_location="cpu", weights_only=False) if cache_path.exists() else build_vision_cache(model, (train_ds, val_ds), device)
    if not cache_path.exists(): torch.save(vision_cache, cache_path)
    train_bank = build_preloaded_bank(train_ds, stats, vision_cache, device) if cache_subset else None
    val_bank = build_preloaded_bank(val_ds, stats, vision_cache, device) if cache_subset else None
    writer = SummaryWriter(out / "tensorboard"); iterator = iter(train_loader); started = time.monotonic(); best_path = out / "tiny_rdt_best.pt"
    for step in range(start_step, steps):
        if train_bank is None:
            try: batch = next(iterator)
            except StopIteration: iterator = iter(train_loader); batch = next(iterator)
            rgb, state, actions, mask = _prepare(batch, stats, device); features = vision_features_for(batch, vision_cache, device)
        else:
            indices = torch.randint(len(train_bank["state"]), (batch_size,), device=device)
            state, actions, mask, features = (train_bank[key][indices] for key in ("state", "actions", "mask", "features")); rgb = None
        model.train(); t = torch.randint(0, diffusion.timesteps, (len(state),), device=device); noisy, noise = diffusion.q_sample(actions, t)
        target = noise if prediction_type == "epsilon" else actions
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp): loss = masked_mse(model(rgb, state, noisy, t, mask, features), target, mask)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer); clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update()
        writer.add_scalar("train/epsilon_mse", loss.item(), step)
        if step % validation_interval == 0 or step == steps - 1:
            metrics = evaluate_bank(model, val_bank, stats, diffusion, device, prediction_type) if val_bank is not None else evaluate(model, val_loader, stats, diffusion, device, prediction_type, vision_cache); writer.add_scalars("validation", metrics, step)
            payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "best_validation": best, "model_config": asdict(config), "normalization": asdict(stats), "splits": asdict(splits), "dataset_schema": "mini-rdt-so101-phase1/v2", "action_horizon": horizon, "action_representation": "five radians + normalized gripper opening", "vision_encoder": "torchvision MobileNetV3-Small ImageNet frozen", "diffusion_config": diffusion_config, "seed": seed}
            torch.save(payload, out / "last.pt")
            if metrics["epsilon_mse"] < best:
                best = metrics["epsilon_mse"]; payload["best_validation"] = best; torch.save(payload, best_path)
            print(json.dumps({"step": step, "train": loss.item(), **metrics}))
    writer.close(); (out / "run_summary.json").write_text(json.dumps({"device": str(device), "elapsed_s": time.monotonic() - started, "best_validation_epsilon_mse": best, "parameter_count": model.parameter_counts()}, indent=2) + "\n")
    return best_path


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--dataset", default="artifacts/pickcube_smoke100_rgb160"); p.add_argument("--output", required=True); p.add_argument("--seed", type=int, default=17); p.add_argument("--steps", type=int, default=1000); p.add_argument("--batch-size", type=int, default=8); p.add_argument("--learning-rate", type=float, default=3e-4); p.add_argument("--subset-episodes", type=int); p.add_argument("--resume"); p.add_argument("--no-pretrained-vision", action="store_true"); p.add_argument("--overfit-validation", action="store_true"); p.add_argument("--schedule", choices=("linear", "cosine"), default="linear"); p.add_argument("--prediction-type", choices=("epsilon", "x0"), default="epsilon"); p.add_argument("--validation-interval", type=int, default=50)
    a = p.parse_args(); print(train(a.dataset, a.output, a.seed, a.steps, a.batch_size, a.learning_rate, a.subset_episodes, a.resume, not a.no_pretrained_vision, a.overfit_validation, a.schedule, a.prediction_type, a.validation_interval))
if __name__ == "__main__": main()
