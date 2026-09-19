"""Summarise physics-v2 closed-loop rollouts (evaluation/closed_loop.py output): success, validity and failure phases per K."""
from __future__ import annotations
import argparse
import glob
import json
from math import comb
from pathlib import Path

import numpy as np


def failure_phase(r, rec):
    if r["success"]: return "success"
    if r.get("invalid_reason"): return "invalid_" + r["invalid_reason"]
    closed = rec["executed"][:, 5] < 0.5
    if not closed.any(): return "never_closed"
    if not r.get("ever_side_pinch"): return "closed_without_valid_side_pinch"
    z = rec["cube"][:, 2]
    if z.max() >= 0.03 and z[-1] < 0.02: return "drop"
    return "lift_or_hold"


def main():
    p = argparse.ArgumentParser(); p.add_argument("--rollouts", default="artifacts/physics_v2/closed_loop"); p.add_argument("--output")
    a = p.parse_args(); rows = []
    for f in sorted(glob.glob(f"{a.rollouts}/ep*_k*.json")):
        r = json.loads(Path(f).read_text()); rec = np.load(f.replace(".json", ".npz")); ph = failure_phase(r, rec)
        closed = np.where(rec["executed"][:, 5] < 0.5)[0]; t = int(closed[0]) if len(closed) else None
        close = None
        if t is not None:
            gc, cube = rec["grasp_center"][t], rec["cube"][t]
            close = {"t": t, "grasp_center_height_above_cube_center_mm": 1000 * float(gc[2] - cube[2]), "lateral_offset_mm": 1000 * float(np.linalg.norm(gc[:2] - cube[:2]))}
        rows.append({"episode": r["episode"], "k": r["k"], "success": r["success"], "phase": ph, "early_close": bool(close and close["grasp_center_height_above_cube_center_mm"] > 20.0),
                     "close": close, "any_vertical_pad_contact": r.get("any_vertical_pad_contact"), "ever_side_pinch": r.get("ever_side_pinch"),
                     **{k: r.get(k) for k in ("max_robot_table_penetration_mm", "max_cube_table_penetration_mm", "max_pad_cube_penetration_mm", "invalid_reason", "steps", "replans")}})
    replay = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(f"{a.rollouts}/ep*_expert_replay.json"))]
    ks = sorted({r["k"] for r in rows}); out = {"n": len(rows), "success": sum(r["success"] for r in rows),
        "expert_replay": {"n": len(replay), "success": sum(r["success"] for r in replay)},
        "by_k": {k: {"n": len(s := [r for r in rows if r["k"] == k]), "success": sum(r["success"] for r in s),
                     "phases": {ph: sum(r["phase"] == ph for r in s) for ph in sorted({r["phase"] for r in s})}} for k in ks},
        "per_episode": {e: sum(r["success"] for r in rows if r["episode"] == e) for e in sorted({r["episode"] for r in rows})},
        "all_successes_physically_valid": all(r["invalid_reason"] is None and not r["any_vertical_pad_contact"] for r in rows if r["success"]),
        "max_robot_table_penetration_mm": max(r["max_robot_table_penetration_mm"] or 0 for r in rows),
        "max_cube_table_penetration_mm": max(r["max_cube_table_penetration_mm"] or 0 for r in rows),
        "max_pad_cube_penetration_mm": max(r["max_pad_cube_penetration_mm"] or 0 for r in rows),
        "early_close": sum(r["early_close"] for r in rows), "vertical_pad_contact_rollouts": sum(bool(r["any_vertical_pad_contact"]) for r in rows),
        "close_lateral_offset_mm_median": {"success": float(np.median([r["close"]["lateral_offset_mm"] for r in rows if r["success"] and r["close"]] or [np.nan])),
                                           "failure": float(np.median([r["close"]["lateral_offset_mm"] for r in rows if not r["success"] and r["close"]] or [np.nan]))},
        "rows": rows}
    text = json.dumps(out, indent=1, default=float)
    if a.output: Path(a.output).write_text(text + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1, default=float))


if __name__ == "__main__":
    main()
