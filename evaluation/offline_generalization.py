"""Offline action error of a TinyRDT checkpoint on any set of episodes (physics-v2 generalization study).

Method:
- DDIM-10 sampling with seed 4100 over EVERY window;
- no clipping;
- error on valid (unpadded) action tokens only.

Reports:
- action MAE;
- per-joint MAE;
- MAE by episode quarter and by expert phase;
- episode-start MAE;
- per-episode MAE, used for the novelty analysis.

It never runs a simulation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.ml_dataset import episode_ids
from evaluation.research_audit import load, make_bank, metrics, sample, save_json, sha256


@torch.no_grad()
def evaluate(checkpoint, dataset, ids, device="cuda", seed=4100):
    ck, model, stats, d, prediction, _ = load(checkpoint, device)
    ds, b = make_bank(dataset, ids, stats, model, device, ck.get("run_config", {}).get("padding", "masked"))
    pred = stats.denormalize_action(sample(model, d, prediction, b, seed=seed)); raw, mask = b["raw"], b["mask"]
    phases = {ep: np.load(Path(dataset) / "episodes" / f"episode_{ep:06d}" / "expert_state.npy", allow_pickle=True) for ep in ids}
    ep_phase = np.array([str(phases[int(e)][int(t)]) for e, t in zip(b["episode"].cpu(), b["t"].cpu())])
    sel = lambda m: metrics(pred[m], raw[m], mask[m])
    out = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "step": ck["step"], "dataset": str(dataset), "episodes": list(ids),
           "windows": len(ds), "sampler": "ddim_10", "sampler_seed": seed, "all": sel(torch.ones_like(b["t"], dtype=torch.bool)),
           "quarters": {str(q): sel(b["phase"] == q) for q in range(4)}, "episode_start": sel(b["t"] == 0),
           "expert_phase": {ph: sel(torch.as_tensor(ep_phase == ph, device=raw.device)) for ph in sorted(set(ep_phase))},
           "per_episode": {int(ep): {"action_mae": sel(b["episode"] == ep)["action_mae"], "arm_mae_rad": sel(b["episode"] == ep)["arm_mae_rad"],
                                     "gripper_mae": sel(b["episode"] == ep)["gripper_mae"], "start_mae": sel((b["episode"] == ep) & (b["t"] == 0))["action_mae"]} for ep in ids}}
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); p.add_argument("--dataset", required=True)
    p.add_argument("--episodes", type=int, nargs="*", help="default: every episode in --dataset"); p.add_argument("--train-ids", action="store_true", help="use the checkpoint's own training episodes")
    p.add_argument("--output", required=True); p.add_argument("--device", default="cuda")
    a = p.parse_args(); out = Path(a.output)
    if out.exists(): print(f"exists: {out}"); return
    ids = torch.load(a.checkpoint, map_location="cpu", weights_only=False)["train_episode_ids"] if a.train_ids else (a.episodes or episode_ids(a.dataset))
    r = evaluate(a.checkpoint, a.dataset, ids, a.device); out.parent.mkdir(parents=True, exist_ok=True); save_json(out, r)
    print(json.dumps({"output": str(out), "windows": r["windows"], "all_mae": r["all"]["action_mae"], "start_mae": r["episode_start"]["action_mae"], "per_joint": [round(x, 4) for x in r["all"]["per_joint_mae"]]}))


if __name__ == "__main__":
    main()
