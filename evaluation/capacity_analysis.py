"""Analysis for the targeted capacity x density scaling study (docs/research/capacity_density_scaling_spec.md).

For each policy size (2.0M baseline reused from the density sweep; 4.3M, 9.1M, 19.5M trained here), it produces:
- per-condition success, with every seed visible;
- per-size distance curves with d90..d10;
- the capacity x distance interaction;
- the surrounded-vs-one-sided gap;
- the failure mechanism;
- position-level rescue and regression relative to 2M;
- offline MAE vs closed loop;
- training and inference cost.

Repeated measurements are respected: positions and training seeds are resampled as clusters.
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
from evaluation.seed_replication_analysis import logistic_fit, two_way_bootstrap

C = Path("artifacts/capacity_scaling"); DSW = Path("artifacts/density_sweep")
SIZES = ["m2.0", "m4.3", "m9.1", "m19.5"]
PARAMS = {"m2.0": 2_009_670, "m4.3": 4_304_902, "m9.1": 9_137_286, "m19.5": 19_517_062}
LABEL = {"m2.0": "2.0M", "m4.3": "4.3M", "m9.1": "9.1M", "m19.5": "19.5M"}
COLORS = {"m2.0": "C3", "m4.3": "C1", "m9.1": "C2", "m19.5": "C0"}
CONDS = ["r7.5", "r10.0", "r15.0", "r20.0", "r7.5_onesided"]
SURROUNDED = ["r7.5", "r10.0", "r15.0", "r20.0"]
SEEDS = (0, 1, 2)
rng = np.random.default_rng(0)


def mdir(size, cond, seed, kind):
    if size == "m2.0":
        return {"model": DSW / f"models/{cond}_seed{seed}", "test": DSW / f"closed_loop/{cond}_seed{seed}",
                "train": DSW / f"train_scenes/{cond}_seed{seed}", "offline": DSW / f"offline/{cond}_seed{seed}"}[kind]
    name = f"{size}_{cond}_seed{seed}"
    return {"model": C / f"models/{name}", "test": C / f"closed_loop/{name}", "train": C / f"train_scenes/{name}",
            "offline": C / f"offline/{name}"}[kind]


def load(directory, positions=None):
    rows = []
    for f in sorted(glob.glob(str(directory / "ep*_k8.json"))):
        r = json.loads(Path(f).read_text())
        if positions is not None and r["episode"] not in positions: continue
        rec = np.load(f.replace(".json", ".npz")); cat, flags, close, geo = classify(r, rec)
        ex = rec["executed"]; closed = np.where(ex[:, 5] < 0.1)[0]
        rows.append({"episode": r["episode"], "success": bool(r["success"]), "outcome": cat, "close": close, "steps": r["steps"],
                     "close_step": int(closed[0]) if len(closed) else None, **geo, "invalid_reason": r.get("invalid_reason"),
                     "any_vertical_pad_contact": r.get("any_vertical_pad_contact"),
                     **{k: r.get(k) for k in ("max_robot_table_penetration_mm", "max_cube_table_penetration_mm", "max_pad_cube_penetration_mm")}})
    return rows


def main():
    p = argparse.ArgumentParser(); p.add_argument("--figures", default="docs/assets"); a = p.parse_args()
    control = json.loads((DSW / "expert_control.json").read_text())["conditions"]
    out = {"sizes": {s: {"params": PARAMS[s]} for s in SIZES}, "cells": {}, "missing": []}
    rows = []
    for size in SIZES:
        for cond in CONDS:
            positions = {r["index"]: r for r in control[cond]["rows"] if r["in_benchmark"]}
            cell = {"per_seed": [], "train_scene_per_seed": [], "offline_eval": [], "offline_train": [], "wall_min": [], "vram_mb": []}
            for seed in SEEDS:
                test = load(mdir(size, cond, seed, "test"), positions)
                if len(test) < len(positions): out["missing"].append(f"{size}_{cond}_seed{seed}: {len(test)}/{len(positions)}")
                if not test: continue
                tr = load(mdir(size, cond, seed, "train"))
                cell["per_seed"].append(sum(r["success"] for r in test)); cell["train_scene_per_seed"].append(sum(r["success"] for r in tr))
                off = mdir(size, cond, seed, "offline")
                for part, key in (("eval", "offline_eval"), ("train", "offline_train")):
                    f = Path(f"{off}_{part}.json")
                    if f.exists(): cell[key].append(json.loads(f.read_text())["all"]["action_mae"])
                res = mdir(size, cond, seed, "model") / "result.json"
                if res.exists():
                    r = json.loads(res.read_text()); cell["wall_min"].append(r.get("elapsed_s", np.nan) / 60); cell["vram_mb"].append(r["peak_vram_bytes"] / 1e6)
                for r in test:
                    q = positions[r["episode"]]
                    rows.append({**r, "size": size, "cond": cond, "seed": seed, "params": PARAMS[size], "d1_mm": q["d1_mm"],
                                 "position": f"{cond}#{r['episode']}", "surrounded": cond in SURROUNDED})
            n_pos = len(positions); k = sum(cell["per_seed"]); n = n_pos * len(cell["per_seed"])
            cell.update({"successes": k, "trials": n, "rate": k / n if n else None, "wilson95": wilson(k, n), "positions": n_pos})
            out["cells"][f"{size}|{cond}"] = cell
    # ---- per-size distance curves (surrounded conditions present for every size)
    out["curves"] = {}
    for size in SIZES:
        sub = [r for r in rows if r["size"] == size and r["surrounded"]]
        if len({r["success"] for r in sub}) == 2:
            out["curves"][size] = curve_estimates([r["d1_mm"] for r in sub], [r["success"] for r in sub], clusters=[r["position"] for r in sub])
    # ---- capacity x distance interaction (surrounded), two-way cluster bootstrap over positions and seed index
    sub = [r for r in rows if r["surrounded"]]
    y = np.array([r["success"] for r in sub], float); pos = np.array([r["position"] for r in sub]); sd = np.array([r["seed"] for r in sub])
    dist = np.array([r["d1_mm"] for r in sub]) / 10.0; lp = np.log2(np.array([r["params"] for r in sub]) / PARAMS["m2.0"])
    out["interaction"] = {"main_effects": two_way_bootstrap([dist, lp], ["distance_per_10mm", "log2_params_doubling"], y, pos, sd),
                          "with_interaction": two_way_bootstrap([dist, lp, dist * lp], ["distance_per_10mm", "log2_params_doubling", "distance_x_log2_params"], y, pos, sd),
                          "note": "log2 params is centred at 2M (0 = baseline); distance in units of 10 mm"}
    # one-sided: capacity effect on its own
    one = [r for r in rows if r["cond"] == "r7.5_onesided"]
    if one:
        out["one_sided_capacity"] = two_way_bootstrap([np.log2(np.array([r["params"] for r in one]) / PARAMS["m2.0"])], ["log2_params_doubling"],
                                                      np.array([r["success"] for r in one], float), np.array([r["position"] for r in one]), np.array([r["seed"] for r in one]))
    # ---- support-geometry gap per size, cluster bootstrap
    out["support_gap"] = {}
    for size in SIZES:
        s_ = [r for r in rows if r["size"] == size and r["cond"] == "r7.5"]; o_ = [r for r in rows if r["size"] == size and r["cond"] == "r7.5_onesided"]
        if not s_ or not o_: continue
        gap = np.mean([r["success"] for r in s_]) - np.mean([r["success"] for r in o_]); bs = []
        sp, op = sorted({r["position"] for r in s_}), sorted({r["position"] for r in o_})
        for _ in range(4000):
            ss = rng.choice(SEEDS, 3); a_ = rng.choice(sp, len(sp)); b_ = rng.choice(op, len(op))
            va = [r["success"] for q in a_ for s in ss for r in s_ if r["position"] == q and r["seed"] == s]
            vb = [r["success"] for q in b_ for s in ss for r in o_ if r["position"] == q and r["seed"] == s]
            if va and vb: bs.append(np.mean(va) - np.mean(vb))
        out["support_gap"][size] = {"surrounded": [sum(r["success"] for r in s_), len(s_)], "one_sided": [sum(r["success"] for r in o_), len(o_)],
                                    "gap": float(gap), "bootstrap95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    # ---- failure mechanism
    out["mechanism"] = {}
    for size in SIZES:
        for cond in CONDS:
            sub = [r for r in rows if r["size"] == size and r["cond"] == cond]
            if not sub: continue
            med = lambda xs: float(np.median(xs)) if xs else None
            out["mechanism"][f"{size}|{cond}"] = {
                "lateral_at_close_mm": med([r["close"]["lateral_mm"] for r in sub if r["close"]]),
                "height_at_close_mm": med([r["close"]["height_mm"] for r in sub if r["close"]]),
                "closest_lateral_approach_mm": med([r["min_lateral_mm"] for r in sub]),
                "close_step": med([r["close_step"] for r in sub if r["close_step"] is not None]),
                "pre_pinch_cube_displacement_mm": med([r["max_disp_before_close_mm"] for r in sub]),
                "failures": {o: sum(1 for r in sub if not r["success"] and r["outcome"] == o) for o in sorted({r["outcome"] for r in sub if not r["success"]})}}
    # ---- position-level rescue / regression vs 2M
    out["position_change_vs_2M"] = {}
    base = {}
    for r in rows:
        if r["size"] == "m2.0": base[r["position"]] = base.get(r["position"], 0) + r["success"]
    for size in SIZES[1:]:
        cnt = {}
        for r in rows:
            if r["size"] == size: cnt[r["position"]] = cnt.get(r["position"], 0) + r["success"]
        common = sorted(set(cnt) & set(base))
        diff = {q: cnt[q] - base[q] for q in common}
        out["position_change_vs_2M"][size] = {
            "positions": len(common), "improved": sum(v > 0 for v in diff.values()), "unchanged": sum(v == 0 for v in diff.values()),
            "regressed": sum(v < 0 for v in diff.values()),
            "rescued_from_0of3_to_ge2of3": sum(1 for q in common if base[q] == 0 and cnt[q] >= 2),
            "lost_from_3of3_to_le1of3": sum(1 for q in common if base[q] == 3 and cnt[q] <= 1),
            "by_condition": {c: {"improved": sum(1 for q in common if q.startswith(c + "#") and diff[q] > 0),
                                 "regressed": sum(1 for q in common if q.startswith(c + "#") and diff[q] < 0),
                                 "positions": sum(1 for q in common if q.startswith(c + "#"))} for c in CONDS}}
    # ---- offline vs closed loop
    ms = []
    for key, cell in out["cells"].items():
        size, cond = key.split("|")
        for i, k in enumerate(cell["per_seed"]):
            if i < len(cell["offline_eval"]): ms.append((size, cond, cell["offline_eval"][i], k / cell["positions"]))
    if ms:
        x = np.array([m[2] for m in ms]); yv = np.array([m[3] for m in ms])
        out["offline_vs_closed_loop"] = {"pearson_all_models": float(np.corrcoef(x, yv)[0, 1]), "n": len(ms),
                                         "mean_offline_eval_mae_by_size": {s: float(np.mean([m[2] for m in ms if m[0] == s])) for s in SIZES if any(m[0] == s for m in ms)},
                                         "mean_offline_train_mae_by_size": {s: float(np.mean([v for k2, c in out["cells"].items() if k2.startswith(s + "|") for v in c["offline_train"]])) for s in SIZES},
                                         "within_condition_pearson": {c: (float(np.corrcoef([m[2] for m in ms if m[1] == c], [m[3] for m in ms if m[1] == c])[0, 1])
                                                                          if np.std([m[3] for m in ms if m[1] == c]) > 0 else None) for c in CONDS}}
    # ---- cost
    lat = json.loads((C / "latency.json").read_text()) if (C / "latency.json").exists() else None
    out["cost"] = {s: {"train_wall_min_mean": float(np.nanmean([v for k2, c in out["cells"].items() if k2.startswith(s + "|") for v in c["wall_min"]] or [np.nan])),
                       "peak_vram_mb_max": float(np.nanmax([v for k2, c in out["cells"].items() if k2.startswith(s + "|") for v in c["vram_mb"]] or [np.nan])),
                       "latency": lat["models"].get(s) if lat else None} for s in SIZES}
    # ---- parameter efficiency
    tot = {s: sum(out["cells"][f"{s}|{c}"]["successes"] for c in ["r10.0", "r15.0", "r20.0", "r7.5_onesided"]) for s in SIZES}
    n_tot = {s: sum(out["cells"][f"{s}|{c}"]["trials"] for c in ["r10.0", "r15.0", "r20.0", "r7.5_onesided"]) for s in SIZES}
    d50 = {s: out["curves"].get(s, {}).get("distance_mm_at_probability", {}).get("0.5", {}).get("estimate") for s in SIZES}
    out["primary_totals"] = {s: [tot[s], n_tot[s]] for s in SIZES}
    out["efficiency"] = {f"{LABEL[a_]}->{LABEL[b_]}": {"delta_primary_successes": tot[b_] - tot[a_], "delta_primary_rate": tot[b_] / max(n_tot[b_], 1) - tot[a_] / max(n_tot[a_], 1),
                                                       "delta_d50_mm": (d50[b_] - d50[a_]) if d50[a_] is not None and d50[b_] is not None else None,
                                                       "param_ratio": PARAMS[b_] / PARAMS[a_]} for a_, b_ in zip(SIZES[:-1], SIZES[1:])}
    out["validity"] = {"rollouts": len(rows), "successes_with_invalid_reason": sum(r["success"] and r["invalid_reason"] is not None for r in rows),
                       "invalid_physics_rollouts": sum(r["invalid_reason"] is not None for r in rows),
                       "max_robot_table_penetration_mm": max((r["max_robot_table_penetration_mm"] or 0) for r in rows),
                       "max_cube_table_penetration_mm": max((r["max_cube_table_penetration_mm"] or 0) for r in rows),
                       "max_pad_cube_penetration_mm": max((r["max_pad_cube_penetration_mm"] or 0) for r in rows)}
    out["rows"] = rows
    (C / "summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    figures(out, Path(a.figures))
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "cells", "mechanism")}, indent=1, default=float)[:5000])


def figures(out, adir):
    present = [s for s in SIZES if any(k.startswith(s + "|") and out["cells"][k]["per_seed"] for k in out["cells"])]
    P = np.array([PARAMS[s] for s in present]) / 1e6
    # 1: success vs params per condition (every seed)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.0), sharey=True)
    for ax, cond in zip(axes, CONDS):
        for s in present:
            cell = out["cells"][f"{s}|{cond}"]
            if not cell["per_seed"]: continue
            ax.scatter([PARAMS[s] / 1e6] * len(cell["per_seed"]), [100 * v / cell["positions"] for v in cell["per_seed"]], color=COLORS[s], s=40, zorder=3)
        rates = [100 * out["cells"][f"{s}|{cond}"]["rate"] if out["cells"][f"{s}|{cond}"]["rate"] is not None else np.nan for s in present]
        ax.plot(P, rates, "k-", lw=1.2)
        for s, r_ in zip(present, rates):
            c = out["cells"][f"{s}|{cond}"]; ax.annotate(f"{c['successes']}/{c['trials']}", (PARAMS[s] / 1e6, r_), textcoords="offset points", xytext=(4, 5), fontsize=7)
        ax.set_xscale("log"); ax.set_xticks(P, [LABEL[s] for s in present]); ax.set_xlabel("trainable policy parameters"); ax.grid(alpha=.3)
        ax.set_title(cond + (" (easy control)" if cond == "r7.5" else ""), fontsize=9); ax.set_ylim(-5, 108)
    axes[0].set_ylabel("success (%)  — dots = seeds, line = aggregate"); fig.tight_layout(); fig.savefig(adir / "v2_capacity_by_condition.png", dpi=125); plt.close(fig)
    # 2-6: distance curves, d-values, gap, lateral error, offline, cost
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2))
    ax = axes[0]; xs = np.linspace(0, 25, 200)
    for s in present:
        w = out["curves"].get(s)
        if not w: continue
        ax.plot(xs, 100 / (1 + np.exp(-(w["intercept"] + w["slope_per_10mm"] * xs / 10))), color=COLORS[s], lw=2, label=LABEL[s])
        for cond in SURROUNDED:
            c = out["cells"][f"{s}|{cond}"]
            if c["rate"] is not None: ax.scatter([float(cond[1:])], [100 * c["rate"]], color=COLORS[s], s=30, zorder=3)
    ax.set_xlabel("nearest demonstration (mm), surrounded"); ax.set_ylabel("success (%)"); ax.set_ylim(-5, 108); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("distance response by capacity", fontsize=9)
    ax = axes[1]
    for pr, mk in (("0.75", "s"), ("0.5", "o"), ("0.25", "^")):
        pts = [(PARAMS[s] / 1e6, out["curves"][s]["distance_mm_at_probability"][pr]) for s in present if s in out["curves"]]
        if pts:
            ax.errorbar([q[0] for q in pts], [q[1]["estimate"] for q in pts],
                        yerr=np.clip(np.array([[q[1]["estimate"] - q[1]["bootstrap95"][0], q[1]["bootstrap95"][1] - q[1]["estimate"]] for q in pts]).T, 0, None),
                        fmt=mk + "-", capsize=3, label=f"d{int(float(pr) * 100)}")
    ax.set_xscale("log"); ax.set_xticks(P, [LABEL[s] for s in present]); ax.set_ylabel("distance (mm)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("d75 / d50 / d25 vs capacity", fontsize=9)
    ax = axes[2]
    g = out["support_gap"]
    for i, s in enumerate(present):
        if s not in g: continue
        v = g[s]; ax.bar(i - .18, 100 * v["surrounded"][0] / v["surrounded"][1], .36, color="C0")
        ax.bar(i + .18, 100 * v["one_sided"][0] / v["one_sided"][1], .36, color="C4")
        ax.annotate(f"gap {100 * v['gap']:.0f}", (i, 102), ha="center", fontsize=7)
    ax.set_xticks(range(len(present)), [LABEL[s] for s in present]); ax.set_ylim(0, 110); ax.set_ylabel("success (%) at d1 = 7.5 mm")
    ax.bar([], [], color="C0", label="surrounded"); ax.bar([], [], color="C4", label="one-sided"); ax.legend(fontsize=7, loc="lower right")
    ax.set_title("support-geometry gap by capacity", fontsize=9); ax.grid(alpha=.3, axis="y")
    ax = axes[3]
    for s in present:
        pts = [(float(c[1:]), out["mechanism"][f"{s}|{c}"]["lateral_at_close_mm"]) for c in SURROUNDED if f"{s}|{c}" in out["mechanism"] and out["mechanism"][f"{s}|{c}"]["lateral_at_close_mm"] is not None]
        if pts: ax.plot([q[0] for q in pts], [q[1] for q in pts], "o-", color=COLORS[s], label=LABEL[s])
    ax.set_xlabel("nearest demonstration (mm)"); ax.set_ylabel("median lateral error at close (mm)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("aiming error vs distance by capacity", fontsize=9)
    ax = axes[4]
    for key, cell in out["cells"].items():
        s, cond = key.split("|")
        for i, k in enumerate(cell["per_seed"]):
            if i < len(cell["offline_eval"]): ax.scatter(cell["offline_eval"][i], 100 * k / cell["positions"], color=COLORS[s], s=18, alpha=.8)
    for s in present: ax.scatter([], [], color=COLORS[s], label=LABEL[s])
    ax.set_xlabel("offline MAE at evaluation positions"); ax.set_ylabel("closed-loop success (%)"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    ax.set_title("offline vs closed loop (every model)", fontsize=9)
    fig.tight_layout(); fig.savefig(adir / "v2_capacity_analysis.png", dpi=125); plt.close(fig)
    # 7: position x capacity x seed
    positions = sorted({r["position"] for r in out["rows"]}, key=lambda q: (CONDS.index(q.split("#")[0]), int(q.split("#")[1])))
    cols = [(s, sd) for s in present for sd in SEEDS]
    M = np.full((len(positions), len(cols)), np.nan)
    for r in out["rows"]:
        if (r["size"], r["seed"]) in cols: M[positions.index(r["position"]), cols.index((r["size"], r["seed"]))] = float(r["success"])
    fig, ax = plt.subplots(figsize=(1.0 + 0.42 * len(cols), 0.22 * len(positions) + 1.6))
    ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), [f"{LABEL[s]}\ns{sd}" for s, sd in cols], fontsize=6.5); ax.set_yticks(range(len(positions)), positions, fontsize=6)
    for i in range(1, len(present)): ax.axvline(3 * i - .5, color="k", lw=1)
    ax.set_title("evaluation position x capacity x training seed", fontsize=9); fig.tight_layout(); fig.savefig(adir / "v2_capacity_positions.png", dpi=125); plt.close(fig)
    # 9: cost
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (key, lab) in zip(axes, (("train_wall_min_mean", "training wall time (min, 2 concurrent jobs)"), ("peak_vram_mb_max", "peak training VRAM (MB)"), ("latency", "predict latency (ms, DDIM-10)"))):
        if key == "latency":
            for dev, mk in (("cuda", "o-"), ("cpu", "s--")):
                pts = [(PARAMS[s] / 1e6, out["cost"][s]["latency"][dev]["median_ms"]) for s in present if out["cost"][s]["latency"] and out["cost"][s]["latency"].get(dev)]
                if pts: ax.plot([q[0] for q in pts], [q[1] for q in pts], mk, label=dev)
            ax.axhline(50, color="0.6", ls=":", label="20 Hz control period"); ax.legend(fontsize=7)
        else:
            ax.plot(P, [out["cost"][s][key] for s in present], "o-")
        ax.set_xscale("log"); ax.set_xticks(P, [LABEL[s] for s in present]); ax.set_ylabel(lab, fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(adir / "v2_capacity_cost.png", dpi=125); plt.close(fig)


if __name__ == "__main__":
    main()
