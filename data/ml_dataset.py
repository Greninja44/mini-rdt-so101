"""Episode-safe windows and train-only normalization for Phase-2."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class EpisodeSplits:
    seed: int
    train: list[int]
    validation: list[int]
    test: list[int]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")


def episode_ids(root: str | Path) -> list[int]:
    result = []
    for path in sorted((Path(root) / "episodes").glob("episode_*")):
        meta = json.loads((path / "episode.json").read_text())
        if meta.get("success"):
            result.append(int(meta["episode_index"]))
    if not result:
        raise ValueError("no successful episodes found")
    return result


def make_episode_splits(root: str | Path, seed: int = 17, train_fraction: float = .8, validation_fraction: float = .1) -> EpisodeSplits:
    """Deterministic episode—not frame—split. Test receives the remainder."""
    ids = np.asarray(episode_ids(root), dtype=int)
    rng = np.random.default_rng(seed); rng.shuffle(ids)
    n_train = int(len(ids) * train_fraction); n_val = int(len(ids) * validation_fraction)
    return EpisodeSplits(seed, sorted(ids[:n_train].tolist()), sorted(ids[n_train:n_train+n_val].tolist()), sorted(ids[n_train+n_val:].tolist()))


@dataclass(frozen=True)
class NormalizationStats:
    state_mean: list[float]
    state_std: list[float]
    action_mean: list[float]
    action_std: list[float]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "NormalizationStats":
        return cls(**json.loads(Path(path).read_text()))

    def _tensor(self, value: list[float], x: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(value, dtype=x.dtype, device=x.device)

    def normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._tensor(self.state_mean, x)) / self._tensor(self.state_std, x)

    def denormalize_state(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._tensor(self.state_std, x) + self._tensor(self.state_mean, x)

    def normalize_action(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._tensor(self.action_mean, x)) / self._tensor(self.action_std, x)

    def denormalize_action(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._tensor(self.action_std, x) + self._tensor(self.action_mean, x)


def compute_normalization(root: str | Path, train_ids: Iterable[int], epsilon: float = 1e-6) -> NormalizationStats:
    """Compute statistics exclusively from raw observations/actions in train episodes."""
    root = Path(root); states = []; actions = []
    for idx in train_ids:
        ep = root / "episodes" / f"episode_{idx:06d}"
        states.append(np.concatenate((np.load(ep / "joint_pos.npy"), np.load(ep / "gripper.npy")), axis=1))
        actions.append(np.load(ep / "action.npy"))
    state = np.concatenate(states, axis=0); action = np.concatenate(actions, axis=0)
    return NormalizationStats(state.mean(0).tolist(), np.maximum(state.std(0), epsilon).tolist(), action.mean(0).tolist(), np.maximum(action.std(0), epsilon).tolist())


def hold_terminal_actions(actions: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Explicit absorbing-command extension; never changes recorded arrays.

    This is an optional TRAINING convention, not a claim that post-terminal
    actions were recorded. Evaluation must still use the original valid mask.
    """
    lengths = valid_mask.long().sum(-1)
    if actions.ndim != 3 or valid_mask.shape != actions.shape[:2] or torch.any(lengths < 1):
        raise ValueError("need nonempty [B,H,D] action windows and [B,H] mask")
    expected = torch.arange(actions.shape[1], device=actions.device)[None] < lengths[:, None]
    if not torch.equal(expected, valid_mask.bool()):
        raise ValueError("valid actions must be a contiguous prefix")
    last = actions[torch.arange(len(actions), device=actions.device), lengths-1]
    return torch.where(valid_mask.bool()[..., None], actions, last[:, None])


class ActionWindowDataset(Dataset):
    """Returns RGB_t/state_t and action[t:t+H], padded only within one episode.

    Padding values are zero in raw action space; they are generally NONZERO
    after normalization. The caller must mask both loss and attention. ``valid_mask`` is false for padded
    actions; no frame ever reads into another episode.
    """
    def __init__(self, root: str | Path, ids: Iterable[int], horizon: int = 16, cache_in_memory: bool = False):
        self.root = Path(root); self.horizon = horizon; self.entries: list[tuple[int, int]] = []
        self._episodes: dict[int, dict[str, np.ndarray]] = {}
        for idx in ids:
            path = self.root / "episodes" / f"episode_{idx:06d}"
            arrays = {n: np.load(path / f"{n}.npy", mmap_mode=None if cache_in_memory else "r") for n in ("rgb", "joint_pos", "gripper", "action")}
            self._episodes[idx] = arrays
            self.entries.extend((idx, t) for t in range(len(arrays["action"])))

    def __len__(self) -> int: return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_id, t = self.entries[index]; ep = self._episodes[episode_id]; n = len(ep["action"])
        available = min(self.horizon, n - t)
        actions = np.zeros((self.horizon, 6), dtype=np.float32); actions[:available] = ep["action"][t:t+available]
        mask = np.zeros(self.horizon, dtype=np.float32); mask[:available] = 1.0
        state = np.concatenate((ep["joint_pos"][t], ep["gripper"][t])).astype(np.float32)
        return {"rgb": torch.from_numpy(np.asarray(ep["rgb"][t]).copy()).permute(2, 0, 1), "state": torch.from_numpy(state), "actions": torch.from_numpy(actions), "valid_mask": torch.from_numpy(mask), "episode_id": torch.tensor(episode_id), "t": torch.tensor(t)}
