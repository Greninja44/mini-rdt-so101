"""Validate a physics-v2 dataset: base integrity (data.validate) plus physical-validity checks per episode.

Every episode must:
- be a physics-v2 success with a valid side pinch in each of its final hold steps;
- have no vertical pad contact;
- keep robot-table <= 1.0 mm, cube-table <= 1.0 mm and pad-cube <= 1.5 mm at every step;
- keep joint states and actions within limits;
- have aligned array lengths;
- carry consistent physics/expert/model-fingerprint metadata.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from data.validate import validate_dataset

TOL = {"robot_table_penetration": 1.0e-3, "cube_table_penetration": 1.0e-3, "pad_cube_penetration": 1.5e-3}
HOLD = 5


def validate_v2(root: str | Path) -> dict:
    root = Path(root); errors = list(validate_dataset(root)); stats = {k: [] for k in TOL}; fingerprints = set(); experts = set(); n = 0
    for ep in sorted(p for p in (root / "episodes").glob("episode_*") if p.is_dir()):
        n += 1; meta = json.loads((ep / "episode.json").read_text()); tag = ep.name
        if meta.get("physics_version") != "v2": errors.append(f"{tag}: physics_version {meta.get('physics_version')!r} != 'v2'")
        fingerprints.add(meta.get("model_fingerprint", {}).get("effective_model_sha256")); experts.add(meta.get("expert_version"))
        if not meta.get("success"): errors.append(f"{tag}: not a success")
        arrays = {k: np.load(ep / f"{k}.npy") for k in ("action", "joint_pos", "gripper", "rgb")}
        lengths = {k: len(v) for k, v in arrays.items()}
        diag = {k: np.load(ep / f"contact_{k}.npy") for k in ("side_pinch", "robot_table_penetration", "cube_table_penetration", "pad_cube_penetration", "pad_normals")}
        if len(set(lengths.values()) | {len(diag["side_pinch"])}) != 1: errors.append(f"{tag}: misaligned lengths {lengths} vs diag {len(diag['side_pinch'])}")
        if not diag["side_pinch"][-HOLD:].all(): errors.append(f"{tag}: final {HOLD} steps are not all valid side pinches")
        normals = diag["pad_normals"]; vertical = np.nanmax(np.abs(normals[..., 2])) if np.isfinite(normals).any() else 0.0
        if vertical > 0.7: errors.append(f"{tag}: vertical pad contact (|n_z| = {vertical:.2f})")
        for k, tol in TOL.items():
            worst = float(diag[k].max()); stats[k].append(worst)
            if worst > tol: errors.append(f"{tag}: {k} {1000 * worst:.2f} mm > {1000 * tol:.1f} mm")
        lo, hi = np.array(meta["joint_limits_radians"]["lower"][:5]), np.array(meta["joint_limits_radians"]["upper"][:5])
        if (arrays["joint_pos"] < lo - 1e-3).any() or (arrays["joint_pos"] > hi + 1e-3).any(): errors.append(f"{tag}: joint state outside limits")
        if (arrays["action"][:, :5] < lo - 1e-6).any() or (arrays["action"][:, :5] > hi + 1e-6).any(): errors.append(f"{tag}: action outside limits")
    if len(fingerprints) != 1: errors.append(f"model fingerprints differ across episodes: {fingerprints}")
    if len(experts) != 1: errors.append(f"expert versions differ across episodes: {experts}")
    return {"episodes": n, "errors": errors, "valid": not errors, "expert_versions": sorted(map(str, experts)), "model_fingerprints": sorted(map(str, fingerprints)),
            "max_mm": {k: 1000 * max(v) if v else None for k, v in stats.items()}}


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("dataset"); a = p.parse_args()
    r = validate_v2(a.dataset); print(json.dumps({k: v if k != "errors" else v[:20] for k, v in r.items()}, indent=1))
    raise SystemExit(0 if r["valid"] else 1)
