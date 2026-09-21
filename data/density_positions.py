"""Emit per-condition position files for the controlled density sweep, in the format data/collect_v2.py consumes.

Each condition gets a "train" list (its 80 demonstration positions) and an "eval" list (its evaluation positions, at the centre of each
hole). Episode seeds are deterministic and disjoint between conditions and between the two lists.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(); p.add_argument("--design", default="docs/research/controlled_density_design.json")
    p.add_argument("--output", default="artifacts/density_sweep/positions"); a = p.parse_args()
    design = json.loads(Path(a.design).read_text()); out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    for ci, (name, cond) in enumerate(design["conditions"].items()):
        rows = {"condition": name, "nominal_radius_mm": 1000 * cond["radius"], "geometry": cond["geometry"],
                "train": [{"index": i, "seed": 20000 + 1000 * ci + i, "cube_xy": xy, "is_ring": bool(r), "condition": name}
                          for i, (xy, r) in enumerate(zip(cond["demo_xy"], cond["demo_is_ring"]))],
                "eval": [{"index": i, "seed": 30000 + 1000 * ci + i, "cube_xy": r["cube_xy"], "condition": name,
                          **{k: r[k] for k in r if k not in ("cube_xy", "index")}} for i, r in enumerate(cond["evaluation_positions"])]}
        (out / f"{name}.json").write_text(json.dumps(rows, indent=1) + "\n")
        print(name, "train", len(rows["train"]), "eval", len(rows["eval"]))


if __name__ == "__main__":
    main()
