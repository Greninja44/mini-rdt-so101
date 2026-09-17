"""Inference-only TinyRDT policy returning the Phase-1 action representation."""
from __future__ import annotations
import torch
from data.ml_dataset import NormalizationStats
from models.tiny_rdt import TinyRDT, TinyRDTConfig
from .diffusion import ActionDiffusion


class TinyRDTPolicy:
    def __init__(self, checkpoint_path: str, device: str | None = None, sampling_steps: int = 10, seed: int = 0):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")); ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = TinyRDT(TinyRDTConfig(**ckpt["model_config"])).to(self.device); self.model.load_state_dict(ckpt["model"]); self.model.eval()
        self.stats = NormalizationStats(**ckpt["normalization"])
        diffusion_config = ckpt.get("diffusion_config", {"schedule": "linear", "prediction": "epsilon"})
        self.diffusion = ActionDiffusion.from_config(diffusion_config, device=self.device); self.prediction_type = diffusion_config["prediction"]
        self.sampling_steps = sampling_steps; self.seed = seed; self._calls = 0
    def reset(self) -> None: self._calls = 0
    @torch.no_grad()
    def predict(self, rgb, robot_state) -> torch.Tensor:
        image = torch.as_tensor(rgb, device=self.device).permute(2, 0, 1).unsqueeze(0); state = torch.as_tensor(robot_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        generator = torch.Generator(device=self.device).manual_seed(self.seed + self._calls); self._calls += 1
        actions = self.diffusion.sample(self.model, image, self.stats.normalize_state(state), self.model.config.horizon, self.model.config.action_dim, self.sampling_steps, generator, self.prediction_type)
        return self.stats.denormalize_action(actions)[0].clamp(torch.tensor([-1.919862,-1.745329,-1.69,-1.658063,-2.743847,0],device=self.device), torch.tensor([1.919862,1.745329,1.69,1.658063,2.841206,1],device=self.device)).cpu()
