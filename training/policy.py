"""Inference-only TinyRDT policy returning the Phase-1 action representation."""
from __future__ import annotations
import torch
from data.ml_dataset import NormalizationStats
from models.tiny_rdt import TinyRDT, TinyRDTConfig
from .diffusion import ActionDiffusion


_ACTION_LO = (-1.919862, -1.745329, -1.69, -1.658063, -2.743847, 0.)
_ACTION_HI = (1.919862, 1.745329, 1.69, 1.658063, 2.841206, 1.)


class BCPolicy:
    """Deterministic diagnostic BC chunk policy saved by training.audit_overfit.

    ``rgb``: pooled frozen MobileNet features + normalized state.
    ``privileged``: normalized state + ground-truth cube pose (diagnostic only).
    Inputs are built exactly as in audit_overfit; output is the same clamped
    16x6 raw action chunk as TinyRDTPolicy.
    """
    needs_cube_pose = False

    def __init__(self, checkpoint_path: str, device: str | None = None):
        from torch import nn
        from models.tiny_rdt import FrozenMobileNet
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.kind = ck["baseline"]
        if self.kind not in ("rgb", "privileged"): raise ValueError(f"unsupported closed-loop baseline: {self.kind}")
        self.needs_cube_pose = self.kind == "privileged"
        self.stats = NormalizationStats(**ck["normalization"]); self.horizon = ck["action_horizon"]
        self.model = nn.Sequential(nn.Linear(ck["input_dim"], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 96)).to(self.device)
        self.model.load_state_dict(ck["model"]); self.model.eval()
        if self.kind == "rgb":
            self.vision = FrozenMobileNet(pretrained=False).to(self.device); self.vision.load_state_dict(ck["vision"]); self.vision.eval()
        else:
            self.cube_mean = ck["cube_mean"].to(self.device, torch.float32); self.cube_std = ck["cube_std"].to(self.device, torch.float32)
        self.lo = torch.tensor(_ACTION_LO, device=self.device); self.hi = torch.tensor(_ACTION_HI, device=self.device)

    def reset(self) -> None: pass

    @torch.no_grad()
    def predict(self, rgb, robot_state, cube_pose=None) -> torch.Tensor:
        state = self.stats.normalize_state(torch.as_tensor(robot_state, dtype=torch.float32, device=self.device).unsqueeze(0))
        if self.kind == "rgb":
            image = torch.as_tensor(rgb, device=self.device).permute(2, 0, 1).unsqueeze(0)
            inputs = torch.cat((self.vision(image), state), 1)
        else:
            cube = torch.as_tensor(cube_pose, dtype=torch.float32, device=self.device).unsqueeze(0)
            inputs = torch.cat((state, (cube - self.cube_mean) / self.cube_std), 1)
        actions = self.stats.denormalize_action(self.model(inputs).reshape(1, self.horizon, -1))
        return actions[0].clamp(self.lo, self.hi).cpu()


class TinyRDTPolicy:
    needs_cube_pose = False

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
