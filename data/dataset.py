"""Transparent Phase-1 episodes with explicit observation_t -> action_t alignment."""
from __future__ import annotations
import json
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any
import numpy as np


SCHEMA_VERSION = "mini-rdt-so101-phase1/v2"


class EpisodeRecorder:
    """Accumulates current-observation transitions and writes one episode."""
    def __init__(self, root: str | Path, episode_index: int, metadata: dict[str, Any]):
        self.root = Path(root); self.episode_index = episode_index; self.metadata = metadata
        self.frames: list[dict[str, Any]] = []

    def append(self, observation: dict[str, np.ndarray], action: np.ndarray, info: dict[str, Any], expert_state: str, success: bool = False) -> None:
        """Append observation_t and the action issued immediately after it."""
        self.frames.append({"rgb": observation["rgb"].copy(), "joint_pos": observation["joint_pos"].copy(), "gripper": observation["gripper"].copy(), "action": np.asarray(action, dtype=np.float32).copy(), "cube_pose": np.asarray(info["cube_pose"], dtype=np.float32).copy(), "end_effector_pose": np.asarray(info["end_effector_pose"], dtype=np.float32).copy(), "expert_state": expert_state, "success": bool(success)})

    def set_last_transition_outcome(self, *, success: bool) -> None:
        """Annotate whether the most recently issued action reached success."""
        if not self.frames:
            raise RuntimeError("cannot annotate an episode without a transition")
        self.frames[-1]["success"] = bool(success)

    def save(self, *, success: bool, failure_reason: str | None = None) -> Path:
        """Write a complete episode after its terminal outcome is known."""
        if not self.frames: raise ValueError("cannot save an empty episode")
        path = self.root / "episodes" / f"episode_{self.episode_index:06d}"; path.mkdir(parents=True, exist_ok=True)
        names = ("rgb", "joint_pos", "gripper", "action", "cube_pose", "end_effector_pose")
        for name in names: np.save(path / f"{name}.npy", np.stack([f[name] for f in self.frames]))
        n, freq = len(self.frames), float(self.metadata["control_frequency"])
        # Frame 0 is the reset observation, before the first policy command.
        np.save(path / "timestamp.npy", np.arange(n, dtype=np.float64) / freq)
        np.save(path / "expert_state.npy", np.asarray([f["expert_state"] for f in self.frames]))
        np.save(path / "success.npy", np.asarray([f["success"] for f in self.frames], dtype=np.bool_))
        episode_metadata = {
            **self.metadata,
            "schema_version": SCHEMA_VERSION,
            "episode_index": self.episode_index,
            "length": n,
            "success": bool(success),
            "failure_reason": failure_reason,
            "temporal_alignment": (
                "observation_t -> action_t; every recorded action is issued "
                "immediately after its same-index observation; success[t] is "
                "the terminal outcome after applying action[t]"
            ),
        }
        (path / "episode.json").write_text(json.dumps(episode_metadata, indent=2) + "\n")
        return path


def make_episode_metadata(env: Any, seed: int, action_representation: str, expert: Any | None = None) -> dict[str, Any]:
    lo, hi = env.joint_limits
    source_root = Path(__file__).resolve().parents[1]
    provenance_files = ("simulation/env.py", "simulation/expert.py", "simulation/controllers.py",
                        "simulation/scene.py", "simulation/assets/so101_new_calib.xml", "data/collect.py", "data/dataset.py")
    return {
        "environment_config": asdict(env.config),
        "expert_config": asdict(expert.config) if expert is not None else None,
        "source_sha256": {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest() for name in provenance_files},
        "initial_cube_pose": env.cube_pose.tolist(),
        "task": "PickCube",
        "seed": int(seed),
        "control_frequency": env.config.control_frequency,
        "simulation_timestep": env.config.simulation_timestep,
        "camera": {
            "name": "external_rgb",
            "width": env.config.camera_width,
            "height": env.config.camera_height,
            "fixed": True,
            "rgb_dtype": "uint8",
            "rgb_range": [0, 255],
        },
        "action_representation": action_representation,
        "joint_order": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
        "joint_limits_radians": {"lower": lo.tolist(), "upper": hi.tolist()},
        "gripper_representation": "normalized opening: 0 closed, 1 open",
        "units": {"joint_pos": "radians", "cube_pose": "metres + WXYZ quaternion", "end_effector_pose": "metres + row-major rotation matrix"},
    }


def load_episode(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load one transparent episode without hiding its on-disk fields."""
    path = Path(path)
    metadata = json.loads((path / "episode.json").read_text())
    arrays = {
        name: np.load(path / f"{name}.npy")
        for name in ("rgb", "joint_pos", "gripper", "action", "cube_pose", "end_effector_pose", "timestamp", "expert_state", "success")
    }
    return metadata, arrays
