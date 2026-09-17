"""Validate the documented intermediate format before future LeRobot conversion."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

REQUIRED = ("rgb", "joint_pos", "gripper", "action", "cube_pose", "end_effector_pose", "timestamp", "expert_state", "success")

def validate_dataset(root: str | Path) -> list[str]:
    root = Path(root); failures: list[str] = []
    for ep in sorted((root / "episodes").glob("episode_*")):
        try:
            meta = json.loads((ep / "episode.json").read_text())
            if not {"task", "seed", "control_frequency", "simulation_timestep", "camera", "action_representation", "joint_order", "joint_limits_radians", "success", "failure_reason", "temporal_alignment"} <= meta.keys(): raise ValueError("missing required metadata")
            arrays = {n: np.load(ep / f"{n}.npy") for n in REQUIRED}
            lengths = {n: len(v) for n, v in arrays.items()}
            if len(set(lengths.values())) != 1: raise ValueError(f"length mismatch: {lengths}")
            if arrays["rgb"].ndim != 4 or arrays["rgb"].shape[-1] != 3: raise ValueError("RGB must be TxHxWx3")
            if arrays["rgb"].dtype != np.uint8: raise ValueError("RGB dtype must be uint8")
            if tuple(arrays["rgb"].shape[1:3]) != (meta["camera"]["height"], meta["camera"]["width"]): raise ValueError("RGB dimensions differ from metadata")
            if arrays["joint_pos"].shape[1:] != (5,) or arrays["action"].shape[1:] != (6,) or arrays["gripper"].shape[1:] != (1,): raise ValueError("unexpected state/action dimensions")
            for n, a in arrays.items():
                if n not in ("success", "expert_state") and not np.isfinite(a).all(): raise ValueError(f"non-finite values in {n}")
            if not np.all(np.diff(arrays["timestamp"]) > 0): raise ValueError("timestamps are not strictly monotonic")
            if not np.logical_or(arrays["success"] == 0, arrays["success"] == 1).all(): raise ValueError("invalid success label")
            if not np.all((arrays["gripper"] >= 0) & (arrays["gripper"] <= 1)): raise ValueError("gripper states outside [0,1]")
            if not (np.all((arrays["action"][:, 5] >= 0) & (arrays["action"][:, 5] <= 1))): raise ValueError("gripper actions outside [0,1]")
            if arrays["expert_state"].ndim != 1 or np.any(arrays["expert_state"] == ""): raise ValueError("invalid expert state sequence")
            lo = np.asarray(meta["joint_limits_radians"]["lower"][:5]); hi = np.asarray(meta["joint_limits_radians"]["upper"][:5])
            if np.any(arrays["action"][:, :5] < lo - 1e-5) or np.any(arrays["action"][:, :5] > hi + 1e-5): raise ValueError("arm actions outside joint limits")
            # MuJoCo's finite-stiffness position servos may transiently exceed a
            # hard hinge range by a few milliradians while resolving contact.
            # This remains a state-integrity check, distinct from strict action
            # clipping above.
            state_limit_tolerance = 1e-2
            if np.any(arrays["joint_pos"] < lo - state_limit_tolerance) or np.any(arrays["joint_pos"] > hi + state_limit_tolerance): raise ValueError("joint positions outside joint limits")
        except Exception as exc: failures.append(f"{ep}: {exc}")
    if not list((root / "episodes").glob("episode_*")): failures.append("no episodes found")
    return failures

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("dataset"); a = p.parse_args(); errors = validate_dataset(a.dataset)
    print(json.dumps({"valid": not errors, "errors": errors}, indent=2)); raise SystemExit(bool(errors))
if __name__ == "__main__": main()
