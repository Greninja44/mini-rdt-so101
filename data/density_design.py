"""Construct the controlled demonstration-density datasets (docs/research/controlled_density_sweep_spec.md).

Each condition is a SEPARATE 80-demonstration training set built so that its evaluation positions have a deliberately chosen distance to
their nearest demonstration:

- every evaluation position sits at the centre of a hole of radius r that contains no demonstration;
- a ring of demonstrations sits exactly at radius r (surrounded geometry, or a one-sided arc for the secondary condition);
- the remaining demonstrations fill the rest of the workspace by farthest-point sampling, never entering a hole.

Demonstration count is identical in every condition, so density conditions are not a disguised dataset-size sweep. Only the local geometry
around evaluation positions changes. All geometry is then re-measured from the constructed positions; nominal labels are never used in the
analysis.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.workspace_analysis import WORKSPACE_X, WORKSPACE_Y, coverage, nn_distances

DEMOS_PER_CONDITION = 80
MARGIN = 0.001          # keep every position >= 1 mm inside the workspace
RING_CLEARANCE = 0.0005  # filler stays this far beyond the hole radius
MIN_SEPARATION = 0.002   # no two cube positions closer than 2 mm
ONE_SIDED_CLEAR = 2.5    # one-sided condition: the unsupported side is empty out to 2.5 r
# nominal radius (m) -> (target number of evaluation positions, ring angles in degrees)
SURROUNDED = [round(np.degrees(t)) for t in np.linspace(0, 2 * np.pi, 7)[:6]]
ONE_SIDED = [-45, -15, 15, 45]
CONDITIONS = {
    "r2.5": {"radius": 0.0025, "targets": 12, "angles": SURROUNDED, "geometry": "surrounded"},
    "r5.0": {"radius": 0.0050, "targets": 12, "angles": SURROUNDED, "geometry": "surrounded"},
    "r7.5": {"radius": 0.0075, "targets": 10, "angles": SURROUNDED, "geometry": "surrounded"},
    "r10.0": {"radius": 0.0100, "targets": 8, "angles": SURROUNDED, "geometry": "surrounded"},
    "r15.0": {"radius": 0.0150, "targets": 5, "angles": SURROUNDED, "geometry": "surrounded"},
    "r20.0": {"radius": 0.0200, "targets": 3, "angles": SURROUNDED, "geometry": "surrounded"},
    "r7.5_onesided": {"radius": 0.0075, "targets": 10, "angles": ONE_SIDED, "geometry": "one_sided"},
}


def grid(step=0.0005, inset=0.0):
    xs = np.arange(WORKSPACE_X[0] + inset, WORKSPACE_X[1] - inset + 1e-9, step)
    ys = np.arange(WORKSPACE_Y[0] + inset, WORKSPACE_Y[1] - inset + 1e-9, step)
    gx, gy = np.meshgrid(xs, ys)
    return np.c_[gx.ravel(), gy.ravel()]


def place_centers(radius, n_target, seed=11):
    """Deterministic packing of evaluation centres inside the inset workspace.

    Centres stay >= 2r + 0.5 mm apart, so no demonstration ever falls inside another centre's hole (a ring demo exactly between two
    centres is legal: it lies on both hole boundaries, not inside either). The workspace is only 45 mm wide, so when the usable x span is
    narrower than the required separation the centres zig-zag between the two x edges, which buys more rows than a plain column.
    """
    inset = radius + MARGIN; sep = 2 * radius + 0.0005
    x_lo, x_hi = WORKSPACE_X[0] + inset, WORKSPACE_X[1] - inset
    y_lo, y_hi = WORKSPACE_Y[0] + inset, WORKSPACE_Y[1] - inset
    if x_hi < x_lo or y_hi < y_lo: return np.empty((0, 2))
    span = x_hi - x_lo
    pts = []
    if span >= sep:                                   # hexagonal rows
        dy = sep * np.sqrt(3) / 2
        for j in range(int(np.floor((y_hi - y_lo) / dy)) + 1):
            y = y_lo + j * dy; off = 0.0 if j % 2 == 0 else sep / 2
            for i in range(int(np.floor((span - off) / sep)) + 1):
                pts.append((x_lo + off + i * sep, y))
    else:                                             # two-column zig-zag: dx uses the full span, dy closes the separation
        dx = span; dy = float(np.sqrt(max(sep ** 2 - dx ** 2, 1e-12)))
        for j in range(int(np.floor((y_hi - y_lo) / dy)) + 1):
            pts.append((x_lo if j % 2 == 0 else x_hi, y_lo + j * dy))
    pts = np.array(pts)
    if not len(pts): return np.empty((0, 2))
    pts = pts + np.array([0.0, (y_hi + y_lo) / 2 - (pts[:, 1].min() + pts[:, 1].max()) / 2])   # centre the pattern in y only
    pts = pts[(pts[:, 1] >= y_lo - 1e-9) & (pts[:, 1] <= y_hi + 1e-9)]
    if len(pts) <= n_target: return pts
    chosen = [pts[np.lexsort((pts[:, 0], pts[:, 1]))[0]]]                                      # deterministic, spread-out subset
    while len(chosen) < n_target:
        d = nn_distances(pts, np.array(chosen)); chosen.append(pts[int(np.argmax(d))])
    return np.array(chosen)


def build(name, cfg, seed=11):
    radius, angles = cfg["radius"], np.radians(cfg["angles"])
    centers = place_centers(radius, cfg["targets"])
    demos, ring_of = [], []
    for i, c in enumerate(centers):
        for a in angles:
            p = c + radius * np.array([np.cos(a), np.sin(a)])
            if not (WORKSPACE_X[0] <= p[0] <= WORKSPACE_X[1] and WORKSPACE_Y[0] <= p[1] <= WORKSPACE_Y[1]): continue
            if demos and nn_distances(p, np.array(demos)).min() < MIN_SEPARATION: continue
            if nn_distances(p, centers).min() < radius - 1e-9: continue   # never inside another hole
            demos.append(p); ring_of.append(i)
    # filler: farthest-point over the workspace, outside every hole, never closer than MIN_SEPARATION to another demo
    cand = grid(0.001, MARGIN)
    keep = nn_distances(cand, centers) >= radius + RING_CLEARANCE
    if cfg["geometry"] == "one_sided":  # the unsupported side must stay empty, otherwise filler restores surround
        arc = np.radians(np.array(cfg["angles"])); lo, hi = arc.min() - np.radians(30), arc.max() + np.radians(30)
        for c in centers:
            v = cand - c; ang = np.arctan2(v[:, 1], v[:, 0]); dist = np.linalg.norm(v, axis=1)
            outside_arc = ~((ang >= lo) & (ang <= hi))
            keep &= ~(outside_arc & (dist < ONE_SIDED_CLEAR * radius))
    cand = cand[keep]
    while len(demos) < DEMOS_PER_CONDITION and len(cand):
        d = nn_distances(cand, np.array(demos)) if demos else np.full(len(cand), np.inf)
        j = int(np.argmax(d))
        if d[j] < MIN_SEPARATION: break
        demos.append(cand[j]); ring_of.append(-1)
        cand = np.delete(cand, j, axis=0)
    demos = np.array(demos)
    return centers, demos, np.array(ring_of)


def measure(centers, demos, radius):
    """Actual geometry per evaluation position; nominal labels are never used downstream."""
    rows = []
    for i, c in enumerate(centers):
        d = np.sort(np.linalg.norm(demos - c, axis=1))
        vec = demos[np.linalg.norm(demos - c, axis=1) <= 2.5 * radius] - c  # fixed window, comparable across conditions
        ang = np.sort(np.degrees(np.arctan2(vec[:, 1], vec[:, 0])) % 360) if len(vec) else np.array([])
        gap = float(np.max(np.diff(np.r_[ang, ang[:1] + 360]))) if len(ang) > 1 else 360.0
        rows.append({"index": i, "cube_xy": c.tolist(),
                     "d1_mm": float(1000 * d[0]), "d2_mm": float(1000 * d[1]), "d3_mm": float(1000 * d[2]),
                     "mean_d1_d6_mm": float(1000 * d[:6].mean()),
                     "n_within_5mm": int((d <= 0.005).sum()), "n_within_10mm": int((d <= 0.010).sum()), "n_within_15mm": int((d <= 0.015).sum()),
                     "local_support_points_within_2p5r": int(len(vec)), "largest_angular_gap_deg": gap,
                     "surrounded": bool(gap <= 180.0), "nominal_radius_mm": 1000 * radius})
    return rows


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", default="docs/research/controlled_density_design.json")
    p.add_argument("--figure", default="docs/assets/v2_density_design.png"); a = p.parse_args()
    design = {"demos_per_condition": DEMOS_PER_CONDITION, "workspace_x_m": WORKSPACE_X, "workspace_y_m": WORKSPACE_Y,
              "construction": "farthest-point evaluation centres (>= 2r + 2 mm apart); a ring of demonstrations at exactly r; "
                              "farthest-point filler outside every hole; identical demonstration count in every condition",
              "min_separation_m": MIN_SEPARATION, "conditions": {}}
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(3.0 * len(CONDITIONS), 8.6), sharey=True)
    for ax, (name, cfg) in zip(np.atleast_1d(axes), CONDITIONS.items()):
        centers, demos, ring_of = build(name, cfg)
        rows = measure(centers, demos, cfg["radius"])
        own = 1000 * nn_distances(demos, demos, exclude_self=True)
        design["conditions"][name] = {
            **{k: v for k, v in cfg.items() if k != "angles"}, "ring_angles_deg": cfg["angles"],
            "evaluation_positions": rows, "n_evaluation_positions": len(rows), "n_demos": len(demos),
            "demo_xy": demos.tolist(), "demo_is_ring": (ring_of >= 0).tolist(),
            "geometry_summary": {"d1_mm_median": float(np.median([r["d1_mm"] for r in rows])),
                                 "d1_mm_min": float(min(r["d1_mm"] for r in rows)), "d1_mm_max": float(max(r["d1_mm"] for r in rows)),
                                 "largest_angular_gap_deg_median": float(np.median([r["largest_angular_gap_deg"] for r in rows]))},
            "dataset_similarity": {"demos": len(demos), "centroid_m": demos.mean(0).tolist(),
                                   "x_range_mm": [float(1000 * demos[:, 0].min()), float(1000 * demos[:, 0].max())],
                                   "y_range_mm": [float(1000 * demos[:, 1].min()), float(1000 * demos[:, 1].max())],
                                   "own_nn_mm": {"median": float(np.median(own)), "mean": float(own.mean()), "max": float(own.max())},
                                   "mean_pairwise_spacing_mm": float(1000 * np.mean([np.linalg.norm(demos[i] - demos[j]) for i in range(len(demos)) for j in range(i + 1, len(demos))])),
                                   "workspace_coverage": {f"within_{q}mm": coverage(demos, q / 1000) for q in (5, 10, 20)}}}
        W = 1000 * np.array([WORKSPACE_X, WORKSPACE_Y])
        ax.add_patch(plt.Rectangle((W[0, 0], W[1, 0]), np.ptp(W[0]), np.ptp(W[1]), fill=False, ls="--", color="0.6"))
        ax.scatter(*(1000 * demos[ring_of >= 0]).T, s=14, color="C0", label="ring demos")
        if (ring_of < 0).any(): ax.scatter(*(1000 * demos[ring_of < 0]).T, s=14, color="0.55", label="filler demos")
        ax.scatter(*(1000 * centers).T, s=52, marker="*", color="C3", label="evaluation positions")
        for c in centers: ax.add_patch(plt.Circle(1000 * c, 1000 * cfg["radius"], fill=False, color="C3", lw=.6, alpha=.6))
        ax.set_title(f"{name}\n{len(rows)} eval pos, {len(demos)} demos\nmedian d1 {design['conditions'][name]['geometry_summary']['d1_mm_median']:.1f} mm", fontsize=8.5)
        ax.set_xlabel("x (mm)"); ax.set_aspect("equal"); ax.grid(alpha=.25)
        if name == list(CONDITIONS)[0]: ax.set_ylabel("y (mm)"); ax.legend(fontsize=6, loc="lower left")
    fig.tight_layout(); Path(a.figure).parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.figure, dpi=120); plt.close(fig)
    out = Path(a.output)
    if out.exists() and json.loads(out.read_text()) != json.loads(json.dumps(design)):
        raise SystemExit(f"{out} exists and differs: the design is frozen")
    out.write_text(json.dumps(design, indent=1) + "\n")
    print(json.dumps({n: {"eval_positions": c["n_evaluation_positions"], "demos": c["n_demos"], **c["geometry_summary"],
                          "own_nn_median_mm": round(c["dataset_similarity"]["own_nn_mm"]["median"], 2),
                          "coverage_10mm": round(c["dataset_similarity"]["workspace_coverage"]["within_10mm"], 3)}
                      for n, c in design["conditions"].items()}, indent=1))


if __name__ == "__main__":
    main()
