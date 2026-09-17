"""Deliberately small non-diffusion action-chunk baseline for diagnostics."""
from __future__ import annotations

import torch
from torch import nn

from .tiny_rdt import FrozenMobileNet


class TinyActionBC(nn.Module):
    """Frozen MobileNet image feature + state -> normalized 16x6 action chunk.

    This model is not a replacement policy.  It is an overfit sanity check for
    the observation, target-window, and normalization contracts independent of
    reverse diffusion.
    """
    def __init__(self, horizon: int = 16, action_dim: int = 6, state_dim: int = 6, hidden_dim: int = 256, pretrained_vision: bool = True):
        super().__init__()
        self.horizon = horizon; self.action_dim = action_dim
        self.vision = FrozenMobileNet(pretrained_vision)
        self.net = nn.Sequential(
            nn.Linear(self.vision.output_dim + state_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, horizon * action_dim),
        )

    def forward(self, rgb: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        features = torch.cat((self.vision(rgb), state), dim=-1)
        return self.net(features).reshape(-1, self.horizon, self.action_dim)

    def parameter_counts(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters()); trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
