"""Deterministic spatial split for the physics-v2 generalization study (docs/research/physics_v2_generalization_spec.md).

It uses only cube positions and never looks at policy results.

- HOLE20 (test category B, sparse interpolation): the 20 dataset positions whose y is nearest to the centre of CLEAN10's largest
  interior y-gap. The band spans the full x range, so every hole position lies between training positions in y.
- TRAIN80: the other 80 dataset positions. CLEAN10 ⊂ TRAIN20 ⊂ TRAIN40 ⊂ TRAIN80 are nested by farthest-point sampling from CLEAN10.
- VAL10 and TEST-A20 (dense interpolation): fresh positions, uniform in the workspace outside the hole band.
- TEST-C20 (extrapolation): fresh positions 3-15 mm outside the sampling workspace rectangle.

All fresh positions are ≥ 2 mm from every dataset position and from each other. Demonstrations for VAL and TEST are collected
afterwards by the unchanged physics-v2 expert (data/collect_v2.py --positions). Positions the expert cannot solve are
marked outside the benchmark; they are never replaced.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.workspace_analysis import CLEAN10, WORKSPACE_X, WORKSPACE_Y, dataset_positions, nn_distances

HOLE_SIZE = 20
MIN_SEPARATION = 0.002
EXTRAPOLATION_MARGIN = (0.003, 0.015)
SEEDS = {"val": 7100, "test_a": 7200, "test_c": 7300}


def outside_distance(p):
    """Distance (m) from p to the workspace rectangle; 0 inside."""
    dx = max(WORKSPACE_X[0] - p[0], 0.0, p[0] - WORKSPACE_X[1]); dy = max(WORKSPACE_Y[0] - p[1], 0.0, p[1] - WORKSPACE_Y[1])
    return float(np.hypot(dx, dy))


def largest_interior_gap(values):
    v = np.sort(values); g = np.diff(v); i = int(np.argmax(g)); return float(v[i]), float(v[i + 1])


def farthest_point_order(start, pool, xy):
    """Add pool ids one at a time, each time taking the id farthest from the current set (ties → lowest id)."""
    chosen, order = list(start), []; rest = sorted(set(pool) - set(start))
    while rest:
        d = nn_distances(np.array([xy[i] for i in rest]), np.array([xy[i] for i in chosen]))
        j = rest[int(np.argmax(d))]; chosen.append(j); order.append(j); rest.remove(j)
    return order


def sample_fresh(n, seed, accept, existing):
    rng = np.random.default_rng(seed); out = []
    lo = (WORKSPACE_X[0] - EXTRAPOLATION_MARGIN[1], WORKSPACE_Y[0] - EXTRAPOLATION_MARGIN[1]); hi = (WORKSPACE_X[1] + EXTRAPOLATION_MARGIN[1], WORKSPACE_Y[1] + EXTRAPOLATION_MARGIN[1])
    while len(out) < n:
        p = rng.uniform(lo, hi)
        if not accept(p): continue
        if nn_distances(p, np.array(existing + out)).min() < MIN_SEPARATION: continue
        out.append(p)
    return [p.tolist() for p in out]


def build(dataset):
    pos = dataset_positions(dataset); ids = sorted(pos); xy = {i: np.array(pos[i]) for i in ids}
    gap = largest_interior_gap([xy[i][1] for i in CLEAN10]); y_c = 0.5 * (gap[0] + gap[1])
    hole = sorted(sorted(ids, key=lambda i: (abs(xy[i][1] - y_c), i))[:HOLE_SIZE])
    if set(hole) & set(CLEAN10): raise RuntimeError("hole band would remove a CLEAN10 position")
    train80 = [i for i in ids if i not in hole]
    hy = [xy[i][1] for i in hole]; ty = np.array([xy[i][1] for i in train80])
    band = (0.5 * (min(hy) + ty[ty < min(hy)].max()), 0.5 * (max(hy) + ty[ty > max(hy)].min()))
    order = farthest_point_order(CLEAN10, train80, xy)
    subsets = {"CLEAN10": sorted(CLEAN10), "TRAIN20": sorted(CLEAN10 + order[:10]), "TRAIN40": sorted(CLEAN10 + order[:30]), "TRAIN80": train80}
    inside = lambda p: WORKSPACE_X[0] <= p[0] <= WORKSPACE_X[1] and WORKSPACE_Y[0] <= p[1] <= WORKSPACE_Y[1]
    dense = lambda p: inside(p) and not (band[0] <= p[1] <= band[1])
    ring = lambda p: EXTRAPOLATION_MARGIN[0] <= outside_distance(p) <= EXTRAPOLATION_MARGIN[1]
    existing = [xy[i].tolist() for i in ids]
    val = sample_fresh(10, SEEDS["val"], dense, existing); existing += val
    test_a = sample_fresh(20, SEEDS["test_a"], dense, existing); existing += test_a
    test_c = sample_fresh(20, SEEDS["test_c"], ring, existing)
    test = ([{"category": "A_interpolation", "cube_xy": p, "source": "fresh"} for p in test_a]
            + [{"category": "B_sparse_interpolation", "cube_xy": xy[i].tolist(), "source": f"dataset_episode_{i}", "dataset_seed": 3000 + i} for i in hole]
            + [{"category": "C_extrapolation", "cube_xy": p, "source": "fresh"} for p in test_c])
    val = [{"category": "validation_interpolation", "cube_xy": p, "source": "fresh"} for p in val]
    train_xy = {name: np.array([xy[i] for i in s]) for name, s in subsets.items()}; centroid = train_xy["TRAIN80"].mean(0)
    def novelty(p):
        p = np.array(p)
        return {**{f"nn_{name.lower()}_mm": float(1000 * nn_distances(p, t)[0]) for name, t in train_xy.items()},
                "dist_to_train80_centroid_mm": float(1000 * np.linalg.norm(p - centroid)),
                "train80_within_10mm": int((np.linalg.norm(train_xy["TRAIN80"] - p, axis=1) <= 0.01).sum()),
                "train80_within_20mm": int((np.linalg.norm(train_xy["TRAIN80"] - p, axis=1) <= 0.02).sum()),
                "outside_workspace_mm": 1000 * outside_distance(p)}
    for i, r in enumerate(val): r.update(index=i, seed=5000 + i, **novelty(r["cube_xy"]))
    for i, r in enumerate(test): r.update(index=i, seed=6000 + i, **novelty(r["cube_xy"]))
    return {"version": "physics-v2-generalization-split-1", "dataset": dataset, "workspace_x_m": WORKSPACE_X, "workspace_y_m": WORKSPACE_Y,
            "rule": {"hole": f"{HOLE_SIZE} dataset positions nearest (in y) to the centre of CLEAN10's largest interior y-gap",
                     "clean10_largest_interior_y_gap_m": gap, "hole_centre_y_m": y_c, "hole_band_y_m": band,
                     "nested_subsets": "farthest-point sampling over TRAIN80 starting from CLEAN10 (ties -> lowest id)",
                     "fresh_sampling_seeds": SEEDS, "min_separation_m": MIN_SEPARATION, "extrapolation_margin_m": EXTRAPOLATION_MARGIN,
                     "reset": "env.reset(seed, options={'cube_xy': ...}); seed is metadata only (cube yaw fixed at 0)"},
            "train_subsets": subsets, "farthest_point_order": order, "validation": val, "test": test}


def plot(split, path):
    xy = {int(k): v for k, v in dataset_positions(split["dataset"]).items()}
    fig, ax = plt.subplots(figsize=(4.6, 9)); W = 1000 * np.array([WORKSPACE_X, WORKSPACE_Y])
    ax.add_patch(plt.Rectangle((W[0, 0], W[1, 0]), np.ptp(W[0]), np.ptp(W[1]), fill=False, ls="--", color="0.5", label="training workspace"))
    band = 1000 * np.array(split["rule"]["hole_band_y_m"]); ax.axhspan(*band, color="C1", alpha=.12, label="held-out hole band")
    t = 1000 * np.array([xy[i] for i in split["train_subsets"]["TRAIN80"]]); c = 1000 * np.array([xy[i] for i in split["train_subsets"]["CLEAN10"]])
    ax.scatter(*t.T, s=16, color="0.35", label="TRAIN80"); ax.scatter(*c.T, s=60, facecolor="none", edgecolor="C3", lw=1.5, label="CLEAN10 (⊂ TRAIN80)")
    v = 1000 * np.array([r["cube_xy"] for r in split["validation"]]); ax.scatter(*v.T, s=26, marker="s", facecolor="none", edgecolor="C2", label="validation (10)")
    for cat, m, col in (("A_interpolation", "^", "C0"), ("B_sparse_interpolation", "D", "C1"), ("C_extrapolation", "*", "C4")):
        p = 1000 * np.array([r["cube_xy"] for r in split["test"] if r["category"] == cat]); ax.scatter(*p.T, s=40 if m != "*" else 70, marker=m, color=col, label=f"test {cat[0]} ({len(p)})")
    ax.set_xlabel("cube x (mm)"); ax.set_ylabel("cube y (mm)"); ax.set_aspect("equal"); ax.grid(alpha=.3); ax.set_title("physics-v2 spatial generalization split", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1)); fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--dataset", default="artifacts/pickcube_physics_v2_rgb160")
    p.add_argument("--output", default="docs/research/physics_v2_generalization_split.json"); p.add_argument("--figure", default="docs/assets/v2_generalization_split.png")
    a = p.parse_args(); split = build(a.dataset)
    out = Path(a.output)
    if out.exists() and json.loads(out.read_text()) != json.loads(json.dumps(split)): raise SystemExit(f"{out} exists and differs: the split is frozen")
    out.write_text(json.dumps(split, indent=1) + "\n"); plot(split, a.figure)
    summ = lambda rows, k: {q: round(float(np.percentile([r[k] for r in rows], p)), 1) for q, p in (("min", 0), ("median", 50), ("max", 100))}
    print(json.dumps({"rule": split["rule"], "sizes": {k: len(v) for k, v in split["train_subsets"].items()},
                      "val_nn_train80_mm": summ(split["validation"], "nn_train80_mm"),
                      **{f"test_{c}": {"nn_train80_mm": summ(rows, "nn_train80_mm"), "nn_clean10_mm": summ(rows, "nn_clean10_mm"), "outside_mm": summ(rows, "outside_workspace_mm")}
                         for c in ("A", "B", "C") for rows in [[r for r in split["test"] if r["category"][0] == c]]}}, indent=1))


if __name__ == "__main__":
    main()
