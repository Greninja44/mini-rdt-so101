"""Corrected capacity x density analysis after the EMA-resume recovery (docs/research/capacity_density_recovery_spec.md).

Builds the protocol-valid 45-run matrix programmatically: a cell uses its recovered run when the committed integrity audit flagged it, and
its original run otherwise. The contaminated versions of flagged runs appear only in the integrity comparison, never as extra observations.

Produces the original-vs-recovered table, the corrected matrix, per-capacity distance curves, the capacity x distance interaction under the
corrected cluster structure, the support-geometry gap, mechanism and position-level analyses, offline-vs-closed-loop, compute and validity.
"""
from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.density_analysis import curve_estimates
from evaluation.generalization_analysis import classify, wilson
from evaluation.recovery_bootstrap import bootstrap_logistic, bootstrap_rate
from scripts.artifact_manifest import sha256

ORIG = Path("artifacts/capacity_scaling"); REC = Path("artifacts/capacity_recovery_v1"); DSW = Path("artifacts/density_sweep")
OUT = REC
SIZES = ["m2.0", "m4.3", "m9.1", "m19.5"]
PARAMS = {"m2.0": 2_009_670, "m4.3": 4_304_902, "m9.1": 9_137_286, "m19.5": 19_517_062}
LABEL = {"m2.0": "2.0M", "m4.3": "4.3M", "m9.1": "9.1M", "m19.5": "19.5M"}
COLORS = {"m2.0": "C3", "m4.3": "C1", "m9.1": "C2", "m19.5": "C0"}
CONDS = ["r10.0", "r15.0", "r20.0", "r7.5_onesided", "r7.5"]
SURROUNDED = ["r7.5", "r10.0", "r15.0", "r20.0"]
SEEDS = (0, 1, 2)
D1 = {"r7.5": 7.5, "r10.0": 10.0, "r15.0": 15.0, "r20.0": 20.0, "r7.5_onesided": 7.5}


def run_name(size, cond, seed):
    return f"{cond}_seed{seed}" if size == "m2.0" else f"{size}_{cond}_seed{seed}"


def roots(size, cond, seed, flagged):
    """Which tree owns this cell: the 2M baseline, the recovered run if the audit flagged it, otherwise the original."""
    name = run_name(size, cond, seed)
    if size == "m2.0": return "baseline", DSW, name
    if name in flagged and (REC / "models" / name / "result.json").exists(): return "recovered", REC, name
    return "original", ORIG, name


def load_rollouts(root, kind, name, positions=None):
    rows = []
    for f in sorted(glob.glob(str(root / kind / name / "ep*_k8.json"))):
        r = json.loads(Path(f).read_text())
        if positions is not None and r["episode"] not in positions: continue
        rec = np.load(f.replace(".json", ".npz")); cat, flags, close, geo = classify(r, rec)
        ex = rec["executed"]; closed = np.where(ex[:, 5] < 0.1)[0]
        rows.append({"episode": r["episode"], "success": bool(r["success"]), "outcome": cat, "close": close, "steps": r["steps"],
                     "close_step": int(closed[0]) if len(closed) else None, **geo, "invalid_reason": r.get("invalid_reason"),
                     "any_vertical_pad_contact": r.get("any_vertical_pad_contact"),
                     **{k: r.get(k) for k in ("max_robot_table_penetration_mm", "max_cube_table_penetration_mm", "max_pad_cube_penetration_mm")}})
    return rows


def cell_record(root, name, positions):
    res = root / "models" / name / "result.json"
    if not res.exists(): return None
    r = json.loads(res.read_text())
    test = load_rollouts(root, "closed_loop", name, positions)
    train = load_rollouts(root, "train_scenes", name)
    offline = {}
    for part, key in (("eval", "eval"), ("train", "train")):
        for stem in (f"{name}_{part}.json", f"{name}_{'eval' if part == 'eval' else 'train'}.json"):
            f = root / "offline" / stem
            if f.exists(): offline[key] = json.loads(f.read_text())["all"]["action_mae"]; break
    return {"train_mae": r["sampled"]["action_mae"], "loss": r["loss"], "wall_min": r.get("elapsed_s", np.nan) / 60,
            "vram_mb": r["peak_vram_bytes"] / 1e6, "params": r["policy_parameters"], "resume": json.loads((root / "models" / name / "config.json").read_text()).get("resume"),
            "ema_sha256": sha256(root / "models" / name / "ema_last.pt"),
            "test": test, "train_scenes": train, "offline": offline,
            "successes": sum(x["success"] for x in test), "trials": len(test),
            "train_scene_successes": sum(x["success"] for x in train), "train_scene_trials": len(train)}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--figures", default="docs/assets"); p.add_argument("--audit", default="docs/research/capacity_integrity_audit.json")
    a = p.parse_args()
    audit = json.loads(Path(a.audit).read_text())["runs"]
    flagged = {k for k, v in audit.items() if v.get("resume") or not v["closed_loop"]["checkpoint_hashes_match"] or not v["train_scenes"]["checkpoint_hashes_match"]}
    control = json.loads((DSW / "expert_control.json").read_text())["conditions"]
    out = {"flagged_runs": sorted(flagged), "n_flagged": len(flagged), "sources": {}, "missing": [], "cells": {}, "integrity_comparison": []}
    rows = []
    for size in SIZES:
        for cond in CONDS:
            positions = {r["index"]: r for r in control[cond]["rows"] if r["in_benchmark"]}
            cell = {"per_seed": [], "train_scene_per_seed": [], "train_mae": [], "offline_eval": [], "offline_train": [],
                    "wall_min": [], "vram_mb": [], "sources": [], "ema_sha256": []}
            for seed in SEEDS:
                source, root, name = roots(size, cond, seed, flagged)
                rec = cell_record(root, name, positions)
                if rec is None or len(rec["test"]) < len(positions):
                    out["missing"].append(f"{size}|{cond}|seed{seed} ({source}): {0 if rec is None else len(rec['test'])}/{len(positions)}")
                    if rec is None: continue
                out["sources"][f"{size}|{cond}|seed{seed}"] = {"source": source, "run": name, "ema_sha256": rec["ema_sha256"], "resume_in_config": bool(rec["resume"])}
                cell["per_seed"].append(rec["successes"]); cell["train_scene_per_seed"].append(rec["train_scene_successes"])
                cell["train_mae"].append(rec["train_mae"]); cell["sources"].append(source); cell["ema_sha256"].append(rec["ema_sha256"])
                cell["wall_min"].append(rec["wall_min"]); cell["vram_mb"].append(rec["vram_mb"])
                for k2, key in (("eval", "offline_eval"), ("train", "offline_train")):
                    if k2 in rec["offline"]: cell[key].append(rec["offline"][k2])
                for r in rec["test"]:
                    q = positions[r["episode"]]
                    rows.append({**r, "capacity": size, "condition": cond, "seed": seed, "params": PARAMS[size],
                                 "position": f"{cond}#{r['episode']}", "d1_mm": q["d1_mm"], "surrounded": cond in SURROUNDED,
                                 "log2_params": float(np.log2(PARAMS[size] / PARAMS["m2.0"])), "distance": q["d1_mm"] / 10.0,
                                 "source": source, "success": float(r["success"])})
                # integrity comparison for flagged cells: original vs recovered, same cell
                if source == "recovered":
                    o = cell_record(ORIG, name, positions)
                    if o: out["integrity_comparison"].append({
                        "run": name, "capacity": size, "condition": cond, "seed": seed,
                        "original": {"successes": o["successes"], "trials": o["trials"], "train_mae": o["train_mae"],
                                     "train_scenes": [o["train_scene_successes"], o["train_scene_trials"]], "ema_sha256": o["ema_sha256"],
                                     "offline_eval": o["offline"].get("eval")},
                        "recovered": {"successes": rec["successes"], "trials": rec["trials"], "train_mae": rec["train_mae"],
                                      "train_scenes": [rec["train_scene_successes"], rec["train_scene_trials"]], "ema_sha256": rec["ema_sha256"],
                                      "offline_eval": rec["offline"].get("eval")},
                        "delta_successes": rec["successes"] - o["successes"],
                        "delta_train_mae": rec["train_mae"] - o["train_mae"],
                        "delta_train_scenes": rec["train_scene_successes"] - o["train_scene_successes"],
                        "ema_checkpoint_changed": rec["ema_sha256"] != o["ema_sha256"]})
            k = sum(cell["per_seed"]); n = len(positions) * len(cell["per_seed"])
            cell.update({"successes": k, "trials": n, "rate": k / n if n else None, "wilson95": wilson(k, n), "positions": len(positions)})
            out["cells"][f"{size}|{cond}"] = cell
    counts = {s: sum(1 for v in out["sources"].values() if v["source"] == s) for s in ("baseline", "recovered", "original")}
    out["source_counts"] = counts
    out["matrix"] = {"cells_total": len(out["sources"]), "new_runs_in_protocol_matrix": counts["recovered"] + counts["original"],
                     "recovered": counts["recovered"], "reused_original": counts["original"], "reused_2M_baseline_cells": counts["baseline"],
                     "note": "the protocol's 45-run matrix is the new runs (3 capacities x 5 conditions x 3 seeds); the 15 2M baseline "
                             "cells are reused unchanged from the density sweep, giving 60 cells in total"}
    # ---- per-capacity distance curves on the surrounded conditions
    out["curves"] = {}
    for size in SIZES:
        sub = [r for r in rows if r["capacity"] == size and r["surrounded"]]
        if len({r["success"] for r in sub}) == 2:
            out["curves"][size] = curve_estimates([r["d1_mm"] for r in sub], [r["success"] for r in sub], clusters=[r["position"] for r in sub])
    # ---- interaction under the corrected cluster structure
    sub = [r for r in rows if r["surrounded"]]
    for r in sub: r["distance_x_log2"] = r["distance"] * r["log2_params"]
    out["interaction"] = {
        "main_effects": bootstrap_logistic(sub, ["distance", "log2_params"]),
        "with_interaction": bootstrap_logistic(sub, ["distance", "log2_params", "distance_x_log2"]),
        "note": "distance in units of 10 mm; log2_params centred at the 2M baseline; positions condition-stratified and shared across "
                "capacities, seeds resampled independently within capacity"}
    one = [r for r in rows if r["condition"] == "r7.5_onesided"]
    if one: out["one_sided_capacity"] = bootstrap_logistic(one, ["log2_params"])
    # ---- support-geometry gap per capacity
    out["support_gap"] = {}
    for size in SIZES:
        s_ = [r for r in rows if r["capacity"] == size and r["condition"] == "r7.5"]
        o_ = [r for r in rows if r["capacity"] == size and r["condition"] == "r7.5_onesided"]
        if s_ and o_:
            out["support_gap"][size] = {"surrounded": bootstrap_rate(s_), "one_sided": bootstrap_rate(o_),
                                        "gap": float(np.mean([r["success"] for r in s_]) - np.mean([r["success"] for r in o_]))}
    # ---- mechanism
    out["mechanism"] = {}
    for size in SIZES:
        for cond in CONDS:
            sub2 = [r for r in rows if r["capacity"] == size and r["condition"] == cond]
            if not sub2: continue
            med = lambda xs: float(np.median(xs)) if xs else None
            out["mechanism"][f"{size}|{cond}"] = {
                "lateral_at_close_mm": med([r["close"]["lateral_mm"] for r in sub2 if r["close"]]),
                "height_at_close_mm": med([r["close"]["height_mm"] for r in sub2 if r["close"]]),
                "closest_lateral_approach_mm": med([r["min_lateral_mm"] for r in sub2]),
                "close_step": med([r["close_step"] for r in sub2 if r["close_step"] is not None]),
                "pre_pinch_cube_displacement_mm": med([r["max_disp_before_close_mm"] for r in sub2]),
                "failures": {o: sum(1 for r in sub2 if not r["success"] and r["outcome"] == o) for o in sorted({r["outcome"] for r in sub2 if not r["success"]})}}
    # ---- position level vs 2M
    base = {}
    for r in rows:
        if r["capacity"] == "m2.0": base[r["position"]] = base.get(r["position"], 0) + r["success"]
    out["position_change_vs_2M"] = {}
    for size in SIZES[1:]:
        cnt = {}
        for r in rows:
            if r["capacity"] == size: cnt[r["position"]] = cnt.get(r["position"], 0) + r["success"]
        common = sorted(set(cnt) & set(base)); diff = {q: cnt[q] - base[q] for q in common}
        out["position_change_vs_2M"][size] = {
            "positions": len(common), "improved": sum(v > 0 for v in diff.values()), "unchanged": sum(v == 0 for v in diff.values()),
            "regressed": sum(v < 0 for v in diff.values()),
            "rescued_0of3_to_ge2of3": sum(1 for q in common if base[q] == 0 and cnt[q] >= 2),
            "lost_3of3_to_le1of3": sum(1 for q in common if base[q] == 3 and cnt[q] <= 1)}
    # ---- offline vs closed loop, cost, validity
    pts = [(s, c, v, out["cells"][f"{s}|{c}"]["per_seed"][i] / out["cells"][f"{s}|{c}"]["positions"])
           for s in SIZES for c in CONDS for i, v in enumerate(out["cells"][f"{s}|{c}"]["offline_eval"])
           if i < len(out["cells"][f"{s}|{c}"]["per_seed"])]
    if pts:
        x = np.array([q[2] for q in pts]); y = np.array([q[3] for q in pts])
        out["offline_vs_closed_loop"] = {"pearson_all_models": float(np.corrcoef(x, y)[0, 1]), "n": len(pts),
            "mean_offline_eval_by_capacity": {s: float(np.mean([q[2] for q in pts if q[0] == s])) for s in SIZES if any(q[0] == s for q in pts)},
            "within_capacity_pearson": {s: (float(np.corrcoef([q[2] for q in pts if q[0] == s], [q[3] for q in pts if q[0] == s])[0, 1])
                                            if np.std([q[3] for q in pts if q[0] == s]) > 0 else None) for s in SIZES}}
    out["cost"] = {s: {"train_mae_mean": float(np.mean([v for c in CONDS for v in out["cells"][f"{s}|{c}"]["train_mae"]])),
                       "train_scene_successes": sum(sum(out["cells"][f"{s}|{c}"]["train_scene_per_seed"]) for c in CONDS),
                       "train_scene_trials": sum(len(out["cells"][f"{s}|{c}"]["train_scene_per_seed"]) * 10 for c in CONDS),
                       "wall_min_mean": float(np.nanmean([v for c in CONDS for v in out["cells"][f"{s}|{c}"]["wall_min"]])),
                       "peak_vram_mb": float(np.nanmax([v for c in CONDS for v in out["cells"][f"{s}|{c}"]["vram_mb"]]))} for s in SIZES}
    out["validity"] = {"rollouts": len(rows), "successes_with_invalid_reason": sum(r["success"] and r["invalid_reason"] is not None for r in rows),
                       "invalid_physics_rollouts": sum(r["invalid_reason"] is not None for r in rows),
                       "max_robot_table_penetration_mm": max((r["max_robot_table_penetration_mm"] or 0) for r in rows),
                       "max_cube_table_penetration_mm": max((r["max_cube_table_penetration_mm"] or 0) for r in rows),
                       "max_pad_cube_penetration_mm": max((r["max_pad_cube_penetration_mm"] or 0) for r in rows)}
    out["rows"] = rows
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    figures(out, Path(a.figures))
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "cells", "mechanism", "sources")}, indent=1, default=float)[:4500])


def figures(out, adir):
    present = [s for s in SIZES if any(out["cells"][f"{s}|{c}"]["per_seed"] for c in CONDS)]
    P = np.array([PARAMS[s] for s in present]) / 1e6
    # corrected capacity curves per condition
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.0), sharey=True)
    for ax, cond in zip(axes, CONDS):
        for s in present:
            cell = out["cells"][f"{s}|{cond}"]
            if not cell["per_seed"]: continue
            ax.scatter([PARAMS[s] / 1e6] * len(cell["per_seed"]), [100 * v / cell["positions"] for v in cell["per_seed"]],
                       color=COLORS[s], s=40, zorder=3)
        rates = [100 * out["cells"][f"{s}|{cond}"]["rate"] if out["cells"][f"{s}|{cond}"]["rate"] is not None else np.nan for s in present]
        ax.plot(P, rates, "k-", lw=1.2)
        for s, r_ in zip(present, rates):
            c = out["cells"][f"{s}|{cond}"]; ax.annotate(f"{c['successes']}/{c['trials']}", (PARAMS[s] / 1e6, r_), textcoords="offset points", xytext=(4, 5), fontsize=7)
        ax.set_xscale("log"); ax.set_xticks(P, [LABEL[s] for s in present]); ax.set_xlabel("trainable policy parameters")
        ax.grid(alpha=.3); ax.set_ylim(-5, 108); ax.set_title(cond + (" (easy control)" if cond == "r7.5" else ""), fontsize=9)
    axes[0].set_ylabel("success (%) — corrected matrix"); fig.tight_layout()
    fig.savefig(adir / "v2_capacity_corrected_by_condition.png", dpi=125); plt.close(fig)

    # integrity figure: original vs recovered for the flagged runs
    comp = out["integrity_comparison"]
    if comp:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        order = sorted(range(len(comp)), key=lambda i: (comp[i]["capacity"], comp[i]["condition"], comp[i]["seed"]))
        names = [f"{comp[i]['capacity']}\n{comp[i]['condition']}·s{comp[i]['seed']}" for i in order]
        for ax, (key, getter, lab) in zip(axes, (
                ("train_mae", lambda c, w: c[w]["train_mae"], "training MAE"),
                ("train_scenes", lambda c, w: c[w]["train_scenes"][0] / max(c[w]["train_scenes"][1], 1) * 100, "training-scene success (%)"),
                ("heldout", lambda c, w: c[w]["successes"] / max(c[w]["trials"], 1) * 100, "held-out success (%)"))):
            xs = np.arange(len(order))
            ax.bar(xs - .2, [getter(comp[i], "original") for i in order], .4, label="original (defect)", color="0.6")
            ax.bar(xs + .2, [getter(comp[i], "recovered") for i in order], .4, label="recovered", color="C0")
            ax.set_xticks(xs, names, fontsize=5.5, rotation=90); ax.set_ylabel(lab, fontsize=9); ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=7)
        fig.suptitle("EMA-resume defect: original vs recovered, audit-flagged runs only", fontsize=10)
        fig.tight_layout(); fig.savefig(adir / "v2_capacity_recovery_integrity.png", dpi=125); plt.close(fig)

    # corrected distance curves + support gap + mechanism
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]; xs = np.linspace(0, 25, 200)
    for s in present:
        w = out["curves"].get(s)
        if w: ax.plot(xs, 100 / (1 + np.exp(-(w["intercept"] + w["slope_per_10mm"] * xs / 10))), color=COLORS[s], lw=2, label=LABEL[s])
        for cond in SURROUNDED:
            c = out["cells"][f"{s}|{cond}"]
            if c["rate"] is not None: ax.scatter([D1[cond]], [100 * c["rate"]], color=COLORS[s], s=28, zorder=3)
    ax.set_xlabel("nearest demonstration (mm), surrounded"); ax.set_ylabel("success (%)"); ax.set_ylim(-5, 108); ax.grid(alpha=.3)
    ax.legend(fontsize=8); ax.set_title("corrected distance response by capacity", fontsize=9)
    ax = axes[1]
    for i, s in enumerate(present):
        g = out["support_gap"].get(s)
        if not g: continue
        ax.bar(i - .18, 100 * g["surrounded"]["rate"], .36, color="C0"); ax.bar(i + .18, 100 * g["one_sided"]["rate"], .36, color="C4")
        ax.annotate(f"gap {100 * g['gap']:.0f}", (i, 103), ha="center", fontsize=7)
    ax.set_xticks(range(len(present)), [LABEL[s] for s in present]); ax.set_ylim(0, 112); ax.set_ylabel("success (%) at d1 = 7.5 mm")
    ax.bar([], [], color="C0", label="surrounded"); ax.bar([], [], color="C4", label="one-sided"); ax.legend(fontsize=7)
    ax.grid(alpha=.3, axis="y"); ax.set_title("support-geometry gap (corrected)", fontsize=9)
    ax = axes[2]
    for s in present:
        pts = [(D1[c], out["mechanism"][f"{s}|{c}"]["lateral_at_close_mm"]) for c in SURROUNDED
               if f"{s}|{c}" in out["mechanism"] and out["mechanism"][f"{s}|{c}"]["lateral_at_close_mm"] is not None]
        if pts: ax.plot([q[0] for q in pts], [q[1] for q in pts], "o-", color=COLORS[s], label=LABEL[s])
    ax.set_xlabel("nearest demonstration (mm)"); ax.set_ylabel("median lateral error at close (mm)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("aiming error vs distance (corrected)", fontsize=9)
    fig.tight_layout(); fig.savefig(adir / "v2_capacity_corrected_analysis.png", dpi=125); plt.close(fig)


if __name__ == "__main__":
    main()
