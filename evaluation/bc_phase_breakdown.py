"""Per-quarter / episode-start error of the deterministic BC diagnostics.

Answers: with the SAME pooled MobileNet features, can a deterministic model
memorize the episode-start chunk? If yes, pooled features are not the floor
for the diffusion episode-start error on the ten memorized demos.
"""
from __future__ import annotations
import argparse
import json

import torch
from torch import nn

from data.ml_dataset import NormalizationStats
from evaluation.research_audit import load, make_bank, metrics


@torch.no_grad()
def run(checkpoint, dataset, encoder_checkpoint, device):
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    _, vision_model, _, _, _, _ = load(encoder_checkpoint, device)
    stats = NormalizationStats(**ck["normalization"])
    ids = ck["train_episode_ids"]
    _, b = make_bank(dataset, ids, stats, vision_model, device, ck["run_config"].get("padding", "masked"))
    kind = ck["baseline"]
    if kind == "rgb":
        inputs = torch.cat((b["features"], b["state"]), 1)
    elif kind == "state":
        inputs = b["state"]
    else:
        inputs = torch.cat((b["state"], (b["cube"] - ck["cube_mean"]) / ck["cube_std"]), 1)
    model = nn.Sequential(nn.Linear(inputs.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 96)).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    pred = stats.denormalize_action(model(inputs).reshape(-1, 16, 6))
    first = b["t"] == 0
    return {"baseline": kind, "step": ck["step"],
            "all": metrics(pred, b["raw"], b["mask"]),
            "quarters": {str(q): metrics(pred[b["phase"] == q], b["raw"][b["phase"] == q], b["mask"][b["phase"] == q])["action_mae"] for q in range(4)},
            "episode_start": metrics(pred[first], b["raw"][first], b["mask"][first])}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--dataset", default="artifacts/pickcube_smoke100_rgb160")
    p.add_argument("--encoder-checkpoint", default="artifacts/tiny_rdt_overfit10/tiny_rdt_best.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output")
    a = p.parse_args()
    results = [run(c, a.dataset, a.encoder_checkpoint, a.device) for c in a.checkpoints]
    text = json.dumps(results, indent=2)
    print(text)
    if a.output:
        open(a.output, "w").write(text + "\n")
