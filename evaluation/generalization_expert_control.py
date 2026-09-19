"""Expert control for the generalization split. Can the unchanged physics-v2 expert solve every VAL/TEST position, and how?

For each position it reports:
- success;
- side-pinch validity through the final hold;
- vertical pad contact;
- max penetrations;
- time to grasp and time to success.

Positions the expert fails are marked outside the benchmark. It also checks that the re-collected hole (B) demonstrations reproduce the
original dataset episodes bit-exactly.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

HZ = 20


def episode_stats(ep):
    sp = np.load(ep / "contact_side_pinch.npy"); n = np.load(ep / "contact_pad_normals.npy"); L = len(sp)
    first = int(np.argmax(sp)) if sp.any() else None
    return {"expert_success": True, "length": L, "time_to_grasp_s": None if first is None else (first + 1) / HZ, "time_to_success_s": L / HZ,
            "final_hold_side_pinch": bool(sp[-5:].all()), "any_vertical_pad_contact": bool(np.nanmax(np.abs(n[..., 2])) > 0.7) if np.isfinite(n).any() else False,
            **{f"max_{k}_mm": float(1000 * np.load(ep / f"contact_{k}.npy").max()) for k in ("robot_table_penetration", "cube_table_penetration", "pad_cube_penetration")}}


def control(rows, root):
    failed = {json.loads(l)["index"]: json.loads(l) for l in (root / "failed.jsonl").read_text().splitlines()} if (root / "failed.jsonl").exists() else {}
    out = []
    for r in rows:
        ep = root / "episodes" / f"episode_{r['index']:06d}"
        base = {k: r[k] for k in ("index", "category", "cube_xy", "source", "nn_train80_mm", "nn_clean10_mm", "outside_workspace_mm")}
        if ep.exists(): out.append({**base, **episode_stats(ep), "in_benchmark": True})
        else: out.append({**base, "expert_success": False, "in_benchmark": False, "expert_failure": failed.get(r["index"])})
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--split", required=True); p.add_argument("--test", required=True); p.add_argument("--val", required=True)
    p.add_argument("--original", required=True); p.add_argument("--output", required=True)
    a = p.parse_args(); split = json.loads(Path(a.split).read_text())
    test = control(split["test"], Path(a.test)); val = control(split["validation"], Path(a.val))
    determinism = []
    for r in test:
        if not r["source"].startswith("dataset_episode_") or not r["in_benchmark"]: continue
        i = int(r["source"].split("_")[-1]); new = Path(a.test) / "episodes" / f"episode_{r['index']:06d}"; old = Path(a.original) / "episodes" / f"episode_{i:06d}"
        same = {k: bool(np.array_equal(np.load(new / f"{k}.npy"), np.load(old / f"{k}.npy"))) for k in ("action", "joint_pos", "gripper", "cube_pose", "rgb")}
        determinism.append({"test_index": r["index"], "dataset_episode": i, **same})
    summ = lambda rows: {"n": len(rows), "expert_success": sum(r["expert_success"] for r in rows),
                         "all_valid_side_pinch_hold": all(r["final_hold_side_pinch"] for r in rows if r["expert_success"]),
                         "any_vertical_pad_contact": sum(r.get("any_vertical_pad_contact", False) for r in rows),
                         "max_robot_table_penetration_mm": max((r["max_robot_table_penetration_mm"] for r in rows if r["expert_success"]), default=None),
                         "max_cube_table_penetration_mm": max((r["max_cube_table_penetration_mm"] for r in rows if r["expert_success"]), default=None),
                         "max_pad_cube_penetration_mm": max((r["max_pad_cube_penetration_mm"] for r in rows if r["expert_success"]), default=None),
                         "time_to_grasp_s_median": float(np.median([r["time_to_grasp_s"] for r in rows if r["expert_success"]])) if any(r["expert_success"] for r in rows) else None,
                         "time_to_success_s_median": float(np.median([r["time_to_success_s"] for r in rows if r["expert_success"]])) if any(r["expert_success"] for r in rows) else None,
                         "outside_benchmark": [r["index"] for r in rows if not r["expert_success"]]}
    result = {"validation": summ(val), "test": summ(test), "test_by_category": {c: summ([r for r in test if r["category"] == c]) for c in sorted({r["category"] for r in test})},
              "hole_recollection_bit_exact": {"n": len(determinism), "all_identical": all(all(v for k, v in d.items() if k not in ("test_index", "dataset_episode")) for d in determinism), "rows": determinism},
              "test_rows": test, "validation_rows": val}
    Path(a.output).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("test_rows", "validation_rows")}, indent=1))


if __name__ == "__main__":
    main()
