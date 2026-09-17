"""Offline sampler-based action-chunk evaluation and trajectory plots."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.ml_dataset import ActionWindowDataset, NormalizationStats
from models.tiny_rdt import TinyRDT, TinyRDTConfig
from training.diffusion import ActionDiffusion


def evaluate(checkpoint_path: str, dataset_root: str, output: str, split: str = "train", samples: int = 8, sampling_steps: int = 10) -> dict:
    out = Path(output); out.mkdir(parents=True, exist_ok=True); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False); stats = NormalizationStats(**ckpt["normalization"])
    model = TinyRDT(TinyRDTConfig(**ckpt["model_config"])).to(device); model.load_state_dict(ckpt["model"]); model.eval()
    diffusion_config = ckpt.get("diffusion_config", {"schedule": "linear", "prediction": "epsilon"})
    diffusion = ActionDiffusion.from_config(diffusion_config, device=device); prediction_type = diffusion_config["prediction"]
    ids = ckpt["splits"][split]
    # Tiny-overfit checkpoints use the first configured subset of train IDs.
    cfg_path = Path(checkpoint_path).with_name("config.json")
    if split == "train" and cfg_path.exists():
        subset = json.loads(cfg_path.read_text()).get("subset_episodes")
        if subset: ids = ids[:subset]
    loader = DataLoader(ActionWindowDataset(dataset_root, ids), batch_size=1, shuffle=False); rows=[]; start=time.monotonic()
    for index, batch in enumerate(loader):
        if index >= samples: break
        rgb = batch["rgb"].to(device); state = stats.normalize_state(batch["state"].to(device)); target = batch["actions"].to(device); mask = batch["valid_mask"].to(device)
        generator = torch.Generator(device=device).manual_seed(10_000 + index)
        prediction = stats.denormalize_action(diffusion.sample(model, rgb, state, 16, 6, sampling_steps, generator, prediction_type))
        valid = mask[0].bool(); error = prediction[0, valid] - target[0, valid]
        rows.append({"episode": int(batch["episode_id"][0]), "t": int(batch["t"][0]), "mse": float(error.square().mean()), "mae": float(error.abs().mean()), "per_action_mae": error.abs().mean(0).cpu().tolist()})
        if index < 3:
            labels = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
            fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
            for dim, ax in enumerate(axes.flat):
                ax.plot(target[0, :, dim].cpu(), label="ground truth"); ax.plot(prediction[0, :, dim].cpu(), label="sampled")
                ax.set_title(labels[dim]); ax.grid(True, alpha=.3)
            axes[0, 0].legend(); fig.suptitle(f"episode {rows[-1]['episode']}, t={rows[-1]['t']}"); fig.tight_layout(); fig.savefig(out / f"trajectory_{index}.png", dpi=120); plt.close(fig)
    report = {"split": split, "samples": len(rows), "sampling_steps": sampling_steps, "action_mse": float(np.mean([r["mse"] for r in rows])), "action_mae": float(np.mean([r["mae"] for r in rows])), "per_action_mae": np.mean([r["per_action_mae"] for r in rows], axis=0).tolist(), "device": str(device), "elapsed_s": time.monotonic()-start, "rows": rows}
    (out / "offline_metrics.json").write_text(json.dumps(report, indent=2) + "\n"); return report


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); p.add_argument("--dataset", default="artifacts/pickcube_smoke100_rgb160"); p.add_argument("--output", required=True); p.add_argument("--split", choices=("train", "validation", "test"), default="train"); p.add_argument("--samples", type=int, default=8); p.add_argument("--sampling-steps", type=int, default=10); a=p.parse_args(); print(json.dumps(evaluate(a.checkpoint,a.dataset,a.output,a.split,a.samples,a.sampling_steps),indent=2))
if __name__ == "__main__": main()
