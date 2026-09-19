"""CLEAN (A) vs PERTURB (B) vs DAGGER (C) vs size-matched PERTURB (Bm) comparison.

Per-rollout metrics come from evaluation/closed_loop_analysis.analyze_rollout
(definitions fixed before any Phase-4/5 result). Fisher exact tests compare
success counts on identical seeds/conditions.
"""
from __future__ import annotations
from math import comb
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.closed_loop_analysis import ExpertLabeler, analyze_rollout, load_rollouts
from evaluation.research_audit import save_json

CLEAN10 = [0, 2, 3, 4, 5, 6, 7, 8, 9, 11]
VARIANTS = {"A": "artifacts/closed_loop_v2/tinyrdt_ema", "B": "artifacts/closed_loop_v3/B", "C": "artifacts/closed_loop_v3/C", "Bm": "artifacts/closed_loop_v3/Bm", "C40k": "artifacts/closed_loop_v3/C40k", "S": "artifacts/closed_loop_v3/S", "SD": "artifacts/closed_loop_v3/SD", "IO": "artifacts/closed_loop_v3/IO",
            "gripper_oracle": "artifacts/phase6/oracle/gripper_oracle", "arm_oracle": "artifacts/phase6/oracle/arm_oracle"}
COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "Bm": "#eda100", "C40k": "#4a3aa7", "S": "#e87ba4", "SD": "#e34948", "IO": "#199e70", "gripper_oracle": "#52514e", "arm_oracle": "#008300"}
LABELS = {"A": "A CLEAN10", "B": "B +PERTURB", "C": "C +DAGGER", "Bm": "Bm +PERTURB (size-matched to C)", "C40k": "C40k +DAGGER, 40k steps (follow-up)", "S": "S spatial tokens, CLEAN10", "SD": "SD state dropout 0.3, CLEAN10", "IO": "IO image-only, CLEAN10",
          "gripper_oracle": "policy arm + expert gripper", "arm_oracle": "expert arm + policy gripper"}


def med(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def fisher(s1, n1, s2, n2):
    a, b, c, d = s1, n1 - s1, s2, n2 - s2; k = a + c; N = n1 + n2
    p = lambda x: comb(n1, x) * comb(n2, k - x) / comb(N, k)
    p0 = p(a); return sum(p(x) for x in range(max(0, k - n2), min(k, n1) + 1) if p(x) <= p0 * (1 + 1e-9))


def dataset_stats():
    out = {"A": {"episodes": 10, "frames": 483, "action_windows": 483, "corrective_states": 0, "unique_cube_seeds": 10}}
    for name, root in (("perturb", "artifacts/corrective/perturb"), ("dagger", "artifacts/corrective/dagger")):
        eps = sorted(p for p in Path(root, "episodes").glob("episode_*") if p.is_dir() and not p.name.endswith(".tmp"))
        metas = [json.loads((p / "episode.json").read_text()) for p in eps]
        disc = [json.loads(l) for l in Path(root, "discarded.jsonl").read_text().splitlines()] if Path(root, "discarded.jsonl").exists() else []
        ctrl = np.concatenate([np.load(p / "controller.npy") for p in eps]) if eps else np.array([])
        out[name] = {"episodes_kept": len(eps), "episodes_discarded": len(disc), "discarded_seeds": sorted({d["source_seed"] for d in disc}),
                     "corrective_frames": int(sum(m["frames"] for m in metas)), "frames_by_controller": {c: int((ctrl == c).sum()) for c in np.unique(ctrl)},
                     "unique_cube_seeds": len({m["source_seed"] for m in metas})}
        if name == "perturb":
            out[name]["magnitude_counts"] = {str(m): sum(x["config"]["magnitude_rad"] == m for x in metas) for m in (.01, .02, .03, .04)}
            out[name]["peak_time_free_joint_dist_rad_median"] = float(np.median([x["outcome"]["peak_time_free_joint_dist_rad"] for x in metas]))
            out[name]["peak_grasp_center_path_mm_median"] = float(np.median([x["outcome"]["peak_grasp_center_path_dev_mm"] for x in metas]))
        else:
            out[name]["takeover_reasons"] = {r: sum((x["outcome"]["takeover_reason"] or "none").startswith(r) for x in metas) for r in ("joint_dist", "policy close", "none")}
            out[name]["policy_driven_frames"] = int((ctrl == "policy").sum()) if len(ctrl) else 0
    for v in ("B", "C", "Bm", "C40k"):
        cfg = Path("artifacts/corrective_train", v, "config.json")
        if cfg.exists():
            c = json.loads(cfg.read_text()); out[v] = {"clean_windows": c["windows"], "corrective_frames": c["corrective_frames"], "corrective_episodes": len(c["corrective_episodes"]),
                                                      "corrective_fraction_per_batch": c["corrective_fraction"], "steps": c["steps"]}
    return out


def main():
    refs = {}
    for ep in CLEAN10:
        rec = dict(np.load(f"artifacts/closed_loop_v2/tinyrdt_ema/ep{ep:03d}_expert_replay.npz"))
        rec["expert_state"] = np.load(f"artifacts/pickcube_smoke100_rgb160/episodes/episode_{ep:06d}/expert_state.npy"); refs[ep] = rec
    demo_states = np.concatenate([refs[ep]["state"][:, :5] for ep in CLEAN10]); labeler = ExpertLabeler()
    report = {"datasets": dataset_stats(), "normal": {}, "recovery": {}, "tests": {}}
    per_variant = {}
    for v, d in VARIANTS.items():
        rows = load_rollouts(Path(d))
        if not rows: continue
        an = [analyze_rollout(r, refs[r["episode"]], demo_states, labeler) for r in rows]; per_variant[v] = an
        g = [x["grasp"] for x in an if x["grasp"]]
        report["normal"][v] = {
            "by_k": {k: {"n": len(s := [x for x in an if x["k"] == k]), "success": sum(x["success"] for x in s)} for k in (1, 2, 4, 8)},
            "total": {"n": len(an), "success": sum(x["success"] for x in an)},
            "grasped": sum(x["grasped"] for x in an), "lifted": sum(x["lifted"] for x in an), "dropped": sum(x["dropped"] for x in an),
            "failure_phases": {ph: sum(x["failure_phase"] == ph for x in an) for ph in ("approach", "descent", "grasp", "lift", "drop", "other")},
            "onset_0.03rad_median": med([x["onset_0.03rad"] for x in an]),
            "onset_0.03rad_never": sum(x["onset_0.03rad"] is None for x in an),
            "close": {"n": len(g), "early_close": sum(x["close_timing_vs_expert_steps"] < 0 for x in g),
                      "xy_mm_median": float(np.median([x["xy_offset_mm"] for x in g])), "xy_mm_p90": float(np.percentile([x["xy_offset_mm"] for x in g], 90)),
                      "z_mm_median": float(np.median([x["z_offset_mm"] for x in g])),
                      "max_joint_dev_median_rad": float(np.median([max(map(abs, x["joint_dev_vs_expert_close_rad"])) for x in g])),
                      "within_7mm_lateral": sum(x["xy_offset_mm"] <= 7 for x in g)},
            "mean_path_dev_mm_first_30": float(np.mean([np.mean(x["_dev"]["grasp_center_path_mm"][:30]) for x in an])),
            "replans_median": float(np.median([x["replans"] for x in an])),
            "per_episode_success": {ep: sum(x["success"] for x in an if x["episode"] == ep) for ep in CLEAN10}}
    for v in ("B", "C", "Bm", "C40k", "S", "SD", "IO", "gripper_oracle", "arm_oracle"):
        if v in report["normal"] and "A" in report["normal"]:
            a, b = report["normal"]["A"]["total"], report["normal"][v]["total"]
            report["tests"][f"normal A vs {v}"] = fisher(a["success"], a["n"], b["success"], b["n"])
    for v in VARIANTS:
        f = Path("artifacts/recovery", v, "recovery_summary.json")
        if f.exists():
            s = json.loads(f.read_text()); rows = s["rows"]
            report["recovery"][v] = {"by_magnitude": s["table"], "total": {"n": len(rows), "success": sum(r["success"] for r in rows)},
                                     "by_joint": {j: {"n": len(r := [x for x in rows if x["joint"] == j]), "success": sum(x["success"] for x in r)} for j in ("shoulder_pan", "shoulder_lift")},
                                     "handover_joint_dist_median_by_magnitude": {m: med([r["handover"]["joint_dist_rad"] for r in rows if f"{r['magnitude_rad']:.2f}" == m]) for m in s["table"]},
                                     "handover_path_mm_median_by_magnitude": {m: med([r["handover"]["grasp_center_path_mm"] for r in rows if f"{r['magnitude_rad']:.2f}" == m]) for m in s["table"]}}
    for v in ("B", "C", "Bm", "C40k", "S", "SD", "IO"):
        if v in report["recovery"] and "A" in report["recovery"]:
            a, b = report["recovery"]["A"]["total"], report["recovery"][v]["total"]
            report["tests"][f"recovery A vs {v}"] = fisher(a["success"], a["n"], b["success"], b["n"])
    if "B" in report["recovery"] and "C" in report["recovery"]:
        report["tests"]["recovery B vs C"] = fisher(report["recovery"]["B"]["total"]["success"], report["recovery"]["B"]["total"]["n"], report["recovery"]["C"]["total"]["success"], report["recovery"]["C"]["total"]["n"])
    if "B" in report["normal"] and "C" in report["normal"]:
        report["tests"]["normal B vs C"] = fisher(report["normal"]["B"]["total"]["success"], report["normal"]["B"]["total"]["n"], report["normal"]["C"]["total"]["success"], report["normal"]["C"]["total"]["n"])
    out = Path("artifacts/phase5_analysis"); out.mkdir(exist_ok=True)
    save_json(out / "phase5.json", report)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for v, r in report["normal"].items():
        axes[0].plot([1, 2, 4, 8], [r["by_k"][k]["success"] / max(1, r["by_k"][k]["n"]) for k in (1, 2, 4, 8)], marker="o", lw=2, color=COLORS[v], label=LABELS[v])
    axes[0].set(xscale="log", xticks=[1, 2, 4, 8], xticklabels=["1", "2", "4", "8"], xlabel="K (actions executed per replan)", ylabel="success rate (10 memorised seeds)", ylim=(-.05, 1.05))
    for v, r in report["recovery"].items():
        ms = sorted(r["by_magnitude"]); ms = [m for m in ms if r["by_magnitude"][m]["n"]]; axes[1].plot([float(m) for m in ms], [r["by_magnitude"][m]["success"] / r["by_magnitude"][m]["n"] for m in ms], marker="o", lw=2, color=COLORS[v], label=LABELS[v])
    axes[1].set(xlabel="initial perturbation (rad, 3 steps at t=6)", ylabel="recovery success rate (K=4)", ylim=(-.05, 1.05))
    for ax in axes: ax.grid(color="#d9d8d4", lw=.5); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "phase5_success.png", dpi=120); plt.close(fig)
    print(json.dumps({k: report[k] for k in ("normal", "recovery", "tests")}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
