"""Same-window deterministic BC overfit diagnostic; never used for Phase 2 policy execution."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data.ml_dataset import ActionWindowDataset, compute_normalization, make_episode_splits
from models.tiny_bc import TinyActionBC
from training.trainer import masked_mse, set_seed


def _prepare(batch, stats, device):
    return batch["rgb"].to(device), stats.normalize_state(batch["state"].to(device)), stats.normalize_action(batch["actions"].to(device)), batch["actions"].to(device), batch["valid_mask"].to(device)


@torch.no_grad()
def _evaluate(model, loader, stats, device):
    model.eval(); normalized = []; raw = []
    for batch in loader:
        rgb, state, target, raw_target, mask = _prepare(batch, stats, device)
        prediction = model(rgb, state); normalized.append(masked_mse(prediction, target, mask).item())
        valid = mask.bool()[..., None].expand_as(prediction); error = (stats.denormalize_action(prediction) - raw_target)[valid].reshape(-1, 6)
        raw.append(error)
    error = torch.cat(raw)
    return {"normalized_mse": float(np.mean(normalized)), "raw_action_mse": float(error.square().mean()), "raw_action_mae": float(error.abs().mean()), "per_action_mae": error.abs().mean(0).tolist()}


def train(dataset: str, output: str, steps: int = 5000, seed: int = 17, batch_size: int = 8, learning_rate: float = .001):
    set_seed(seed); out = Path(output); out.mkdir(parents=True, exist_ok=True); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = make_episode_splits(dataset, seed); ids = split.train[:10]; stats = compute_normalization(dataset, ids)
    train_ds = ActionWindowDataset(dataset, ids, cache_in_memory=True); loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    evaluation_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    model = TinyActionBC().to(device); optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate, weight_decay=1e-4)
    iterator = iter(loader); initial = _evaluate(model, evaluation_loader, stats, device); started = time.monotonic()
    for step in range(steps):
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        model.train(); rgb, state, target, _, mask = _prepare(batch, stats, device); optimizer.zero_grad(set_to_none=True)
        loss = masked_mse(model(rgb, state), target, mask); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step % 500 == 0 or step == steps - 1: print(json.dumps({"step": step, "loss": loss.item()}))
    final = _evaluate(model, evaluation_loader, stats, device)
    payload = {"model": model.state_dict(), "normalization": stats.__dict__, "split_seed": seed, "episode_ids": ids, "parameter_count": model.parameter_counts()}
    torch.save(payload, out / "tiny_bc_overfit.pt")
    report = {"initial": initial, "final": final, "steps": steps, "episodes": ids, "parameter_count": model.parameter_counts(), "device": str(device), "elapsed_s": time.monotonic() - started}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n"); return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", default="artifacts/pickcube_smoke100_rgb160"); parser.add_argument("--output", required=True); parser.add_argument("--steps", type=int, default=5000); parser.add_argument("--seed", type=int, default=17); parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--learning-rate", type=float, default=.001)
    args = parser.parse_args(); print(json.dumps(train(args.dataset, args.output, args.steps, args.seed, args.batch_size, args.learning_rate), indent=2))


if __name__ == "__main__": main()
