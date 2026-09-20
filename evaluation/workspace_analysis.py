"""Inspect the cube XY positions of the physics-v2 dataset (step 2 of the generalization study). This never trains or evaluates a policy.

It reports:
- workspace bounds;
- nearest-neighbour distances;
- local density;
- grid coverage for all 100 demonstrations and for CLEAN10.

It also plots the workspace.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE_X, WORKSPACE_Y = (0.235, 0.28), (-0.07, 0.07)
CLEAN10 = [0, 2, 3, 4, 5, 6, 7, 8, 9, 11]


def dataset_positions(root):
    eps = sorted(p for p in (Path(root) / "episodes").glob("episode_*") if p.is_dir())
    return {int(p.name.split("_")[1]): json.loads((p / "episode.json").read_text())["initial_cube_pose"][:2] for p in eps}


def nn_distances(query, reference, exclude_self=False):
    """Nearest-neighbour distance (m) from each query point to the reference set."""
    q, r = np.atleast_2d(query), np.atleast_2d(reference)
    d = np.linalg.norm(q[:, None] - r[None], axis=2)
    if exclude_self: d[d == 0] = np.inf
    return d.min(1)


def coverage(points, radius, x=WORKSPACE_X, y=WORKSPACE_Y, n=200):
    """Fraction of the workspace rectangle within `radius` of at least one point (1 mm-ish grid)."""
    gx, gy = np.meshgrid(np.linspace(*x, n), np.linspace(*y, 3 * n))
    grid = np.c_[gx.ravel(), gy.ravel()]
    return float((nn_distances(grid, points) <= radius).mean())


def describe(points):
    nn = nn_distances(points, points, exclude_self=True)
    density = [(np.linalg.norm(points - p, axis=1) <= 0.01).sum() - 1 for p in points]
    return {"n": len(points), "x_range_m": [float(points[:, 0].min()), float(points[:, 0].max())], "y_range_m": [float(points[:, 1].min()), float(points[:, 1].max())],
            "centroid_m": points.mean(0).tolist(),
            "nn_distance_mm": {q: float(np.percentile(1000 * nn, p)) for q, p in (("min", 0), ("p25", 25), ("median", 50), ("p75", 75), ("max", 100))},
            "neighbours_within_10mm": {"mean": float(np.mean(density)), "min": int(np.min(density)), "max": int(np.max(density))},
            "coverage_fraction": {f"within_{r}mm": coverage(points, r / 1000) for r in (5, 10, 20)}}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--dataset", default="artifacts/pickcube_physics_v2_rgb160")
    p.add_argument("--output", default="artifacts/physics_v2_generalization/workspace"); p.add_argument("--figure", default="docs/assets/v2_workspace_positions.png")
    a = p.parse_args(); out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    pos = dataset_positions(a.dataset); ids = sorted(pos); xy = np.array([pos[i] for i in ids]); c10 = np.array([pos[i] for i in CLEAN10])
    area_mm2 = 1e6 * (WORKSPACE_X[1] - WORKSPACE_X[0]) * (WORKSPACE_Y[1] - WORKSPACE_Y[0])
    report = {"dataset": a.dataset, "workspace_x_m": WORKSPACE_X, "workspace_y_m": WORKSPACE_Y, "workspace_area_mm2": area_mm2, "cube_size_mm": 20, "cube_yaw": 0.0,
              "sampling": "uniform in the workspace rectangle, seed 3000+index", "all100": describe(xy), "clean10": describe(c10),
              "clean10_ids": CLEAN10, "clean10_y_mm_sorted": sorted(round(1000 * v, 1) for v in c10[:, 1]),
              "all100_nn_to_clean10_mm": {q: float(np.percentile(1000 * nn_distances(xy, c10), p)) for q, p in (("median", 50), ("p90", 90), ("max", 100))},
              "histogram_3x7": np.histogram2d(xy[:, 0], xy[:, 1], bins=[3, 7], range=[WORKSPACE_X, WORKSPACE_Y])[0].astype(int).tolist(),
              "positions": {i: pos[i] for i in ids}}
    (out / "workspace_analysis.json").write_text(json.dumps(report, indent=1) + "\n")
    fig, ax = plt.subplots(figsize=(4.2, 9))
    ax.add_patch(plt.Rectangle((1000 * WORKSPACE_X[0], 1000 * WORKSPACE_Y[0]), 1000 * np.ptp(WORKSPACE_X), 1000 * np.ptp(WORKSPACE_Y), fill=False, ls="--", color="0.5", label="sampling workspace"))
    ax.scatter(1000 * xy[:, 0], 1000 * xy[:, 1], s=18, color="0.6", label="all 100 demos")
    ax.scatter(1000 * c10[:, 0], 1000 * c10[:, 1], s=60, facecolor="none", edgecolor="C3", lw=1.6, label="CLEAN10")
    ax.set_xlabel("cube x (mm)"); ax.set_ylabel("cube y (mm)"); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_title(f"physics-v2 cube positions\nmedian NN {report['all100']['nn_distance_mm']['median']:.1f} mm (100) / {report['clean10']['nn_distance_mm']['median']:.1f} mm (CLEAN10)", fontsize=9)
    ax.legend(fontsize=7, loc="lower right"); fig.tight_layout(); Path(a.figure).parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.figure, dpi=130); fig.savefig(out / "workspace.png", dpi=130)
    print(json.dumps({k: v for k, v in report.items() if k != "positions"}, indent=1))


if __name__ == "__main__":
    main()
