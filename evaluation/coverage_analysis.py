"""Spatial coverage of the nested training subsets against the frozen 56-position held-out benchmark.

It answers "how close is the nearest demonstration?" for every test position and every data scale, before any model is trained,
so the later data-scaling curve can be separated into "more data" and "denser local coverage".

Outputs a coverage figure plus per-subset distance statistics and fractions within 2.5 / 5 / 7.5 / 10 / 15 mm.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.workspace_analysis import WORKSPACE_X, WORKSPACE_Y, coverage, dataset_positions, nn_distances

SUBSETS = ("CLEAN10", "TRAIN20", "TRAIN40", "TRAIN80")
CATS = ("A_interpolation", "B_sparse_interpolation", "C_extrapolation")
THRESHOLDS_MM = (2.5, 5.0, 7.5, 10.0, 15.0)


def benchmark_positions(split, control):
    inb = {r["index"] for r in json.loads(Path(control).read_text())["test_rows"] if r["in_benchmark"]}
    return [r for r in split["test"] if r["index"] in inb]


def stats(d_mm, cats):
    d = np.asarray(d_mm)
    out = {"n": len(d), "median_mm": float(np.median(d)), "mean_mm": float(d.mean()), "min_mm": float(d.min()), "max_mm": float(d.max()),
           "fraction_within": {f"{t}mm": float((d <= t).mean()) for t in THRESHOLDS_MM},
           "count_within": {f"{t}mm": int((d <= t).sum()) for t in THRESHOLDS_MM}}
    out["by_category"] = {c: {"n": int((np.array(cats) == c).sum()), "median_mm": float(np.median(d[np.array(cats) == c])),
                              "max_mm": float(d[np.array(cats) == c].max())} for c in CATS}
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--split", default="docs/research/physics_v2_generalization_split.json")
    p.add_argument("--control", default="artifacts/physics_v2_generalization/expert_control.json")
    p.add_argument("--dataset", default="artifacts/pickcube_physics_v2_rgb160")
    p.add_argument("--output", default="artifacts/exposure_matched_scaling/coverage.json"); p.add_argument("--figure", default="docs/assets/v2_scaling_coverage.png")
    a = p.parse_args(); split = json.loads(Path(a.split).read_text()); pos = dataset_positions(a.dataset)
    rows = benchmark_positions(split, a.control); xy = np.array([r["cube_xy"] for r in rows]); cats = [r["category"] for r in rows]
    train = {s: np.array([pos[i] for i in split["train_subsets"][s]]) for s in SUBSETS}
    windows = {s: int(sum(len(np.load(Path(a.dataset) / "episodes" / f"episode_{i:06d}" / "action.npy")) for i in split["train_subsets"][s])) for s in SUBSETS}
    out = {"benchmark_positions": len(rows), "subsets": {}}
    for s in SUBSETS:
        d = 1000 * nn_distances(xy, train[s]); own = 1000 * nn_distances(train[s], train[s], exclude_self=True)
        out["subsets"][s] = {"episodes": len(train[s]), "windows": windows[s],
                             "nearest_test_to_train_mm": stats(d, cats),
                             "own_nn_spacing_mm": {"median": float(np.median(own)), "max": float(own.max())},
                             "workspace_coverage_fraction": {f"within_{r}mm": coverage(train[s], r / 1000) for r in (5, 10, 20)},
                             "per_position_mm": {int(r["index"]): float(v) for r, v in zip(rows, d)}}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(json.dumps(out, indent=1) + "\n")

    fig, axes = plt.subplots(1, 5, figsize=(19, 8.4), sharey=True)
    W = 1000 * np.array([WORKSPACE_X, WORKSPACE_Y]); band = 1000 * np.array(split["rule"]["hole_band_y_m"])
    marks = {"A_interpolation": ("^", "C0"), "B_sparse_interpolation": ("D", "C1"), "C_extrapolation": ("*", "C4")}
    for ax, s in zip(axes, SUBSETS):
        ax.add_patch(plt.Rectangle((W[0, 0], W[1, 0]), np.ptp(W[0]), np.ptp(W[1]), fill=False, ls="--", color="0.6")); ax.axhspan(*band, color="C1", alpha=.08)
        t = 1000 * train[s]; ax.scatter(*t.T, s=26, color="0.35", label=f"{s} ({len(t)} demos, {windows[s]} windows)")
        for c, (m, col) in marks.items():
            q = 1000 * xy[np.array(cats) == c]; ax.scatter(*q.T, s=34 if m != "*" else 64, marker=m, color=col, alpha=.85, label=f"test {c[0]}" if s == SUBSETS[0] else None)
        st = out["subsets"][s]["nearest_test_to_train_mm"]
        ax.set_title(f"{s}\nnearest demo: median {st['median_mm']:.1f} mm, max {st['max_mm']:.1f} mm\n{100 * st['fraction_within']['5.0mm']:.0f}% of test within 5 mm", fontsize=9)
        ax.set_xlabel("cube x (mm)"); ax.set_aspect("equal"); ax.grid(alpha=.25); ax.legend(fontsize=6.5, loc="lower right")
    ax = axes[4]
    for s in SUBSETS:
        d = np.sort([out["subsets"][s]["per_position_mm"][r["index"]] for r in rows])
        ax.step(d, 100 * np.arange(1, len(d) + 1) / len(d), where="post", label=f"{s} ({len(train[s])} demos)")
    for t in THRESHOLDS_MM: ax.axvline(t, color="0.85", lw=.8, zorder=0)
    ax.set_xlabel("distance to nearest training cube (mm)"); ax.set_ylabel("% of the 56 test positions"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    ax.set_title("coverage of the held-out benchmark", fontsize=9); ax.set_aspect("auto")
    fig.tight_layout(); Path(a.figure).parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.figure, dpi=120); plt.close(fig)
    print(json.dumps({s: {k: v for k, v in d.items() if k != "per_position_mm"} for s, d in out["subsets"].items()}, indent=1))


if __name__ == "__main__":
    main()
