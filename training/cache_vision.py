"""Incrementally persist frozen MobileNet features for CPU-only experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.ml_dataset import ActionWindowDataset, make_episode_splits
from models.tiny_rdt import TinyRDT, TinyRDTConfig


@torch.no_grad()
def cache(dataset_root: str, output: str, subset_episodes: int = 10, max_batches: int = 1) -> int:
    out = Path(output); path = out / "frozen_vision_features.pt"; device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(path, map_location="cpu", weights_only=False) if path.exists() else {}
    ids = make_episode_splits(dataset_root, 17).train[:subset_episodes]
    dataset = ActionWindowDataset(dataset_root, ids)
    model = TinyRDT(TinyRDTConfig()).to(device).eval(); processed = 0
    # Eight images is deliberately small: this diagnostic workstation is
    # CPU-only and its command runner has a short foreground timeout.
    for batch in DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0):
        keys = [(int(e), int(t)) for e, t in zip(batch["episode_id"].tolist(), batch["t"].tolist())]
        if all(key in cache for key in keys):
            continue
        features = model.vision(batch["rgb"].to(device)).cpu()
        cache.update(dict(zip(keys, features))); torch.save(cache, path); processed += 1
        if processed >= max_batches: break
    print({"cached": len(cache), "total": len(dataset), "processed_batches": processed, "complete": len(cache) == len(dataset)})
    return len(cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", default="artifacts/pickcube_smoke100_rgb160"); parser.add_argument("--output", required=True); parser.add_argument("--subset-episodes", type=int, default=10); parser.add_argument("--max-batches", type=int, default=1)
    args = parser.parse_args(); cache(args.dataset, args.output, args.subset_episodes, args.max_batches)
