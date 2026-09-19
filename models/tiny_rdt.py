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
    # "pooled": one global-average token (original). "spatial": the 576x4x5
    # frozen feature map as 20 tokens (same per-token projection).
    vision_tokens: str = "pooled"
    vision_grid: tuple[int, int] = (4, 5)
    # Training-only regulariser: probability of replacing the state token by a
    # learned null embedding (the parameter exists only when > 0).
    state_dropout: float = 0.0
    # Phase 10: fine-tune the MobileNet weights (BatchNorm statistics stay frozen).
    train_vision: bool = False


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int): super().__init__(); self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2; scale = math.log(10000) / max(half - 1, 1)
        freq = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -scale)
        emb = t.float().unsqueeze(1) * freq.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=1)


class FrozenMobileNet(nn.Module):
    """Small ImageNet MobileNetV3 encoder; only the projection trains."""
    def __init__(self, pretrained: bool = True, spatial: bool = False, trainable: bool = False):
        super().__init__(); self.spatial = spatial; self.trainable = trainable
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = mobilenet_v3_small(weights=weights)
        self.features = base.features; self.pool = nn.AdaptiveAvgPool2d(1); self.output_dim = 576
        self.features.requires_grad_(trainable); self.pool.requires_grad_(False)
        self.features.eval()
    def train(self, mode: bool = True):
        super().train(mode); self.features.eval(); return self
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # ImageNet normalization belongs to the frozen ImageNet encoder only.
        x = rgb.float() / 255.0
        mean = x.new_tensor((0.485, .456, .406))[None, :, None, None]
        std = x.new_tensor((.229, .224, .225))[None, :, None, None]
        with torch.set_grad_enabled(self.trainable and torch.is_grad_enabled()):
            x = self.features((x - mean) / std)
            return x.flatten(2).transpose(1, 2) if self.spatial else self.pool(x).flatten(1)


class TinyRDT(nn.Module):
    """Predicts epsilon for every noisy future action token.

    The token sequence is [vision, state, diffusion-time, noisy_action_1..H].
    It is deliberately compact and uses no language/world-model inputs.
    """
    def __init__(self, config: TinyRDTConfig = TinyRDTConfig()):
        super().__init__(); self.config = config
        if config.vision_tokens not in ("pooled", "spatial"): raise ValueError(config.vision_tokens)
        d = config.hidden_dim; self.vision = FrozenMobileNet(config.pretrained_vision, config.vision_tokens == "spatial", config.train_vision)
        self.n_vision = 1 if config.vision_tokens == "pooled" else config.vision_grid[0] * config.vision_grid[1]; self.n_cond = self.n_vision + 2
        self.vision_proj = nn.Linear(self.vision.output_dim, d); self.state_proj = nn.Sequential(nn.Linear(config.state_dim, d), nn.SiLU(), nn.Linear(d, d))
        if config.state_dropout > 0: self.state_null = nn.Parameter(torch.zeros(d))
        self.time_proj = nn.Sequential(SinusoidalTimeEmbedding(d), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.action_proj = nn.Linear(config.action_dim, d); self.position = nn.Parameter(torch.zeros(1, self.n_cond + config.horizon, d))
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
        state_drop: torch.Tensor | None = None,
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
        if vision_features.ndim == 2: vision_features = vision_features[:, None]
        if vision_features.shape[1] != self.n_vision: raise ValueError("vision feature tokens do not match config.vision_tokens")
        state_token = self.state_proj(state)
        if self.config.state_dropout >= 1: state_drop = torch.ones(state.shape[0], dtype=torch.bool, device=state.device)  # image-only policy
        if state_drop is not None: state_token = torch.where(state_drop[:, None], self.state_null.expand_as(state_token), state_token)
        cond = torch.cat((self.vision_proj(vision_features), state_token[:, None], self.time_proj(timestep)[:, None]), dim=1)
        tokens = torch.cat((cond, self.action_proj(noisy_actions)), dim=1) + self.position[:, :self.n_cond + noisy_actions.shape[1]]
        padding_mask = None
        if action_valid_mask is not None:
            if action_valid_mask.shape != noisy_actions.shape[:2]:
                raise ValueError("action_valid_mask must have shape [batch, horizon]")
            condition_valid = torch.ones((noisy_actions.shape[0], self.n_cond), dtype=torch.bool, device=noisy_actions.device)
            # TransformerEncoder expects True where a token is ignored.
            padding_mask = ~torch.cat((condition_valid, action_valid_mask.bool()), dim=1)
        return self.output(self.transformer(tokens, src_key_padding_mask=padding_mask)[:, self.n_cond:])

    def parameter_counts(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters()); trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
