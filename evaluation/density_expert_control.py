"""Expert control for the controlled density sweep: which evaluation positions can the unchanged physics-v2 expert solve?

Positions the expert cannot solve are excluded from every policy metric, before any policy is evaluated. Training-set shortfalls
(expert failures at demonstration positions) are reported too, since they change a condition's demonstration count.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.generalization_expert_control import episode_stats


def main():
    p = argparse.ArgumentParser(); p.add_argument("--design", default="docs/research/controlled_density_design.json")
    p.add_argument("--data", default="artifacts/density_sweep/data"); p.add_argument("--output", required=True)
    a = p.parse_args(); design = json.loads(Path(a.design).read_text()); out = {"conditions": {}}
    for name, cond in design["conditions"].items():
        root = Path(a.data) / f"{name}_eval"; train_root = Path(a.data) / f"{name}_train"
        failed = {json.loads(l)["index"]: json.loads(l) for l in (root / "failed.jsonl").read_text().splitlines()} if (root / "failed.jsonl").exists() else {}
        rows = []
        for i, r in enumerate(cond["evaluation_positions"]):
            ep = root / "episodes" / f"episode_{i:06d}"
            base = {k: r[k] for k in ("index", "cube_xy", "d1_mm", "d2_mm", "d3_mm", "mean_d1_d6_mm", "n_within_5mm", "n_within_10mm",
                                      "n_within_15mm", "local_support_points_within_2p5r", "largest_angular_gap_deg", "surrounded", "nominal_radius_mm")}
            rows.append({**base, **(episode_stats(ep) if ep.exists() else {"expert_success": False}), "in_benchmark": ep.exists(),
                         "expert_failure": None if ep.exists() else failed.get(i)})
        n_train = len(list((train_root / "episodes").glob("episode_*"))) if train_root.exists() else 0
        ok = [r for r in rows if r["expert_success"]]
        out["conditions"][name] = {
            "nominal_radius_mm": 1000 * cond["radius"], "geometry": cond["geometry"],
            "evaluation_positions": len(rows), "expert_success": len(ok), "excluded": [r["index"] for r in rows if not r["expert_success"]],
            "demonstrations_collected": n_train, "demonstrations_requested": cond["n_demos"],
            "all_valid_side_pinch_hold": all(r["final_hold_side_pinch"] for r in ok),
            "max_penetration_mm": {k: max((r[f"max_{k}_mm"] for r in ok), default=None) for k in ("robot_table_penetration", "cube_table_penetration", "pad_cube_penetration")},
            "time_to_success_s_median": float(np.median([r["time_to_success_s"] for r in ok])) if ok else None,
            "measured_d1_mm": {"median": float(np.median([r["d1_mm"] for r in ok])) if ok else None},
            "rows": rows}
    Path(a.output).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({n: {k: v for k, v in c.items() if k != "rows"} for n, c in out["conditions"].items()}, indent=1))


if __name__ == "__main__":
    main()
