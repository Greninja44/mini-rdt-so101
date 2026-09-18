"""Mechanical check of a research_audit diagnostics.json against gate_definition.json."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def check(diagnostics: dict, gate: dict) -> dict:
    rows = {r["seed"]: r for r in diagnostics["full_training_windows"]}
    per_seed = {}
    for seed in gate["sampling_seeds"]:
        r = rows[seed]
        c = {"all_window_mae": (r["all"]["action_mae"], gate["all_window_mae"]),
             "worst_arm_joint_mae_rad": (max(r["all"]["per_joint_mae"][:5]), gate["each_arm_joint_mae_rad"]),
             "gripper_mae": (r["all"]["gripper_mae"], gate["gripper_mae"]),
             "worst_quarter_mae": (max(q["action_mae"] for q in r["quarters"].values()), gate["each_quarter_mae"]),
             "episode_start_mae": (r["episode_start"]["action_mae"], gate["episode_start_mae"])}
        per_seed[seed] = {k: {"value": v, "limit": lim, "pass": v <= lim} for k, (v, lim) in c.items()}
    return {"pass": all(x["pass"] for s in per_seed.values() for x in s.values()), "per_seed": per_seed}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--diagnostics", required=True)
    p.add_argument("--gate", required=True)
    a = p.parse_args()
    result = check(json.loads(Path(a.diagnostics).read_text()), json.loads(Path(a.gate).read_text()))
    print(json.dumps(result, indent=2))
