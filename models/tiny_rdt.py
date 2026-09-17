"""Small vision-conditioned diffusion Transformer for action chunks."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


@dataclass(frozen=True)
class TinyRDTConfig:
    horizon: int = 16
    action_dim: int = 6
    state_dim: int = 6
    hidden_dim: int = 192
    layers: int = 4
    heads: int = 6
    dropout: float = 0.0
    pretrained_vision: bool = True


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int): super().__init__(); self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2; scale = math.log(10000) / max(half - 1, 1)
        freq = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -scale)
        emb = t.float().unsqueeze(1) * freq.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=1)


class FrozenMobileNet(nn.Module):
    """Small ImageNet MobileNetV3 encoder; only the projection trains."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = mobilenet_v3_small(weights=weights)
        self.features = base.features; self.pool = nn.AdaptiveAvgPool2d(1); self.output_dim = 576
        self.features.requires_grad_(False); self.pool.requires_grad_(False)
        self.features.eval()
    def train(self, mode: bool = True):
        super().train(mode); self.features.eval(); return self
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # ImageNet normalization belongs to the frozen ImageNet encoder only.
        x = rgb.float() / 255.0
        mean = x.new_tensor((0.485, .456, .406))[None, :, None, None]
        std = x.new_tensor((.229, .224, .225))[None, :, None, None]
        with torch.no_grad(): x = self.features((x - mean) / std); x = self.pool(x).flatten(1)
        return x


class TinyRDT(nn.Module):
    """Predicts epsilon for every noisy future action token.

    The token sequence is [vision, state, diffusion-time, noisy_action_1..H].
    It is deliberately compact and uses no language/world-model inputs.
    """
    def __init__(self, config: TinyRDTConfig = TinyRDTConfig()):
        super().__init__(); self.config = config
        d = config.hidden_dim; self.vision = FrozenMobileNet(config.pretrained_vision)
        self.vision_proj = nn.Linear(self.vision.output_dim, d); self.state_proj = nn.Sequential(nn.Linear(config.state_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.time_proj = nn.Sequential(SinusoidalTimeEmbedding(d), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.action_proj = nn.Linear(config.action_dim, d); self.position = nn.Parameter(torch.zeros(1, 3 + config.horizon, d))
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=config.heads, dim_feedforward=d * 4, dropout=config.dropout, batch_first=True, activation="gelu", norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.layers, norm=nn.LayerNorm(d))
        self.output = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, config.action_dim))
        nn.init.normal_(self.position, std=.02)

    def forward(
        self,
        rgb: torch.Tensor,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        action_valid_mask: torch.Tensor | None = None,
        vision_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict epsilon for ``noisy_actions``.

        ``action_valid_mask`` is a [B, H] boolean mask for training windows
        which are padded at an episode boundary.  It is passed to attention,
        not merely the loss: a padded future token must not become context for
        a valid action token.  All H tokens are valid during inference.
        """
        # Frozen encoder features may be cached by the trainer because Phase 2
        # has no image augmentation.  This is exactly the same feature tensor,
        # not a learned replacement for the vision backbone.
        vision_features = self.vision(rgb) if vision_features is None else vision_features
        cond = torch.stack((self.vision_proj(vision_features), self.state_proj(state), self.time_proj(timestep)), dim=1)
        tokens = torch.cat((cond, self.action_proj(noisy_actions)), dim=1) + self.position[:, :3 + noisy_actions.shape[1]]
        padding_mask = None
        if action_valid_mask is not None:
            if action_valid_mask.shape != noisy_actions.shape[:2]:
                raise ValueError("action_valid_mask must have shape [batch, horizon]")
            condition_valid = torch.ones((noisy_actions.shape[0], 3), dtype=torch.bool, device=noisy_actions.device)
            # TransformerEncoder expects True where a token is ignored.
            padding_mask = ~torch.cat((condition_valid, action_valid_mask.bool()), dim=1)
        return self.output(self.transformer(tokens, src_key_padding_mask=padding_mask)[:, 3:])

    def parameter_counts(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters()); trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
