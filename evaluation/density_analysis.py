"""Analysis for the controlled demonstration-density sweep (docs/research/controlled_density_sweep_spec.md).

Estimates the controlled distance-response curve, compares it with the earlier observational curve, tests whether geometry beyond the
nearest demonstration matters, and separates the density effect from training-seed variance.

Repeated measurements are respected: rollouts share evaluation positions and training seeds, so confidence intervals come from a two-way
cluster bootstrap rather than from treating rollouts as independent.
"""
from __future__ import annotations
import argparse
import glob
import json
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.generalization_analysis import classify, wilson
from evaluation.seed_replication_analysis import logistic_fit, two_way_bootstrap

D = Path("artifacts/density_sweep")
SEEDS = (0, 1, 2)
PREDICTORS = ["d1_mm", "d2_mm", "d3_mm", "mean_d1_d6_mm", "n_within_5mm", "n_within_10mm", "n_within_15mm",
              "local_support_points_within_2p5r", "largest_angular_gap_deg"]
rng = np.random.default_rng(0)


def fisher_exact(a, b, c, d):
    """Two-sided Fisher exact test on [[a, b], [c, d]]."""
    n = a + b + c + d; row1, col1 = a + b, a + c
    def prob(x): return comb(row1, x) * comb(n - row1, col1 - x) / comb(n, col1)
    lo, hi = max(0, col1 - (n - row1)), min(row1, col1)
    p0 = prob(a); tot = sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p0 + 1e-12)
    return {"table": [[a, b], [c, d]], "p_two_sided": float(min(1.0, tot))}


def load(condition, seed, positions, kind="closed_loop"):
    rows = []
    for f in sorted(glob.glob(str(D / kind / f"{condition}_seed{seed}" / "ep*_k8.json"))):
        r = json.loads(Path(f).read_text())
        if kind == "closed_loop" and r["episode"] not in positions: continue
        rec = np.load(f.replace(".json", ".npz")); cat, flags, close, geo = classify(r, rec)
        row = {"condition": condition, "seed": seed, "episode": r["episode"], "success": bool(r["success"]), "steps": r["steps"],
               "outcome": cat, "close": close, "invalid_reason": r.get("invalid_reason"),
               "any_vertical_pad_contact": r.get("any_vertical_pad_contact"), **geo,
               **{k: r.get(k) for k in ("max_robot_table_penetration_mm", "max_cube_table_penetration_mm", "max_pad_cube_penetration_mm")}}
        if kind == "closed_loop": row.update({k: positions[r["episode"]][k] for k in PREDICTORS + ["nominal_radius_mm", "surrounded"]})
        rows.append(row)
    return rows


def curve_estimates(dist_mm, success, B=2000, clusters=None):
    """Logistic fit of success on distance, with position-cluster bootstrap CIs for the dX distances."""
    x = np.asarray(dist_mm, float) / 10.0; y = np.asarray(success, float)
    clusters = np.asarray(clusters if clusters is not None else np.arange(len(y)))
    w = logistic_fit(np.c_[np.ones(len(y)), x], y); uniq = np.unique(clusters); boots = []
    for _ in range(B):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.where(clusters == c)[0] for c in pick]); yy = y[idx]
        if len(set(yy)) < 2: continue
        boots.append(logistic_fit(np.c_[np.ones(len(idx)), x[idx]], yy))
    boots = np.array(boots)
    at = lambda pr, ws: 10.0 * (np.log(pr / (1 - pr)) - ws[..., 0]) / ws[..., 1]
    return {"intercept": float(w[0]), "slope_per_10mm": float(w[1]),
            "distance_mm_at_probability": {str(pr): {"estimate": float(at(pr, w)),
                                                     "bootstrap95": [float(np.percentile(at(pr, boots), 2.5)), float(np.percentile(at(pr, boots), 97.5))]}
                                           for pr in (0.9, 0.75, 0.5, 0.25, 0.1)},
            "n_rollouts": int(len(y)), "n_clusters": int(len(uniq)),
            "caveat": "fitted curve; distances associated with an estimated success probability, not thresholds"}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--design", default="docs/research/controlled_density_design.json")
    p.add_argument("--control", default="artifacts/density_sweep/expert_control.json"); p.add_argument("--figures", default="docs/assets")
    a = p.parse_args(); design = json.loads(Path(a.design).read_text()); control = json.loads(Path(a.control).read_text())
    prev = json.loads(Path("artifacts/seed_replication/summary.json").read_text())["generalization_curve"]
    out = {"observational_curve": prev, "conditions": {}, "excluded_positions": {}}
    allrows, trainrows = [], []
    for name, cond in design["conditions"].items():
        cc = control["conditions"][name]
        positions = {r["index"]: r for r in cc["rows"] if r["in_benchmark"]}
        out["excluded_positions"][name] = cc["excluded"]
        rows = [r for s in SEEDS for r in load(name, s, positions)]
        tr = [r for s in SEEDS for r in load(name, s, positions, kind="train_scenes")]
        allrows += rows; trainrows += tr
        if not rows: continue
        per_seed = [sum(r["success"] for r in rows if r["seed"] == s) for s in SEEDS]
        n_pos = len(positions)
        offline = {}
        for s in SEEDS:
            for part in ("train", "eval"):
                f = D / "offline" / f"{name}_seed{s}_{part}.json"
                if f.exists(): offline.setdefault(part, []).append(json.loads(f.read_text())["all"]["action_mae"])
        k, n = sum(r["success"] for r in rows), len(rows)
        out["conditions"][name] = {
            "nominal_radius_mm": cc["nominal_radius_mm"], "geometry": cc["geometry"],
            "measured_d1_mm": {"median": float(np.median([positions[i]["d1_mm"] for i in positions])),
                               "min": float(min(positions[i]["d1_mm"] for i in positions)), "max": float(max(positions[i]["d1_mm"] for i in positions))},
            "evaluation_positions": n_pos, "expert_excluded": cc["excluded"], "demonstrations": cc["demonstrations_collected"],
            "per_seed_successes": per_seed, "successes": k, "trials": n, "rate": k / n if n else None,
            "wilson95_naive": wilson(k, n),
            "train_scene_success_per_seed": [sum(r["success"] for r in tr if r["seed"] == s) for s in SEEDS],
            "train_scene_trials_per_seed": [sum(1 for r in tr if r["seed"] == s) for s in SEEDS],
            "offline_mae": {q: [round(v, 5) for v in vals] for q, vals in offline.items()},
            "failures": {o: sum(1 for r in rows if not r["success"] and r["outcome"] == o) for o in sorted({r["outcome"] for r in rows if not r["success"]})},
            "close_lateral_mm": {q: float(np.median([r["close"]["lateral_mm"] for r in rows if r["close"] and r["success"] == v] or [np.nan]))
                                 for q, v in (("success", True), ("failure", False))},
            "close_height_mm": {q: float(np.median([r["close"]["height_mm"] for r in rows if r["close"] and r["success"] == v] or [np.nan]))
                                for q, v in (("success", True), ("failure", False))},
            "min_lateral_approach_mm": float(np.median([r["min_lateral_mm"] for r in rows])),
            "validity": {"successes_with_invalid_reason": sum(r["success"] and r["invalid_reason"] is not None for r in rows),
                         "invalid_physics_rollouts": sum(r["invalid_reason"] is not None for r in rows),
                         "max_robot_table_penetration_mm": max((r["max_robot_table_penetration_mm"] or 0) for r in rows),
                         "max_cube_table_penetration_mm": max((r["max_cube_table_penetration_mm"] or 0) for r in rows),
                         "max_pad_cube_penetration_mm": max((r["max_pad_cube_penetration_mm"] or 0) for r in rows)},
            "per_position": {r_i: {"successes": sum(1 for r in rows if r["episode"] == r_i and r["success"]), "seeds": sum(1 for r in rows if r["episode"] == r_i),
                                   "d1_mm": positions[r_i]["d1_mm"]} for r_i in sorted(positions)}}
    # ---- primary: controlled curve on SURROUNDED conditions (the density sweep proper)
    main_rows = [r for r in allrows if r["surrounded"]]
    pos_id = np.array([f"{r['condition']}#{r['episode']}" for r in main_rows]); seed_id = np.array([r["seed"] for r in main_rows])
    y = np.array([r["success"] for r in main_rows], float); d1 = np.array([r["d1_mm"] for r in main_rows])
    out["controlled_curve"] = curve_estimates(d1, y, clusters=pos_id)
    out["controlled_logistic_two_way"] = two_way_bootstrap([d1 / 10.0], ["nearest_demo_per_10mm"], y, pos_id, seed_id)
    # ---- predictors beyond d1
    preds = {}
    for name in PREDICTORS:
        x = np.array([r[name] for r in main_rows], float)
        if np.std(x) == 0: continue
        z = (x - x.mean()) / x.std()
        preds[name] = {"alone": two_way_bootstrap([z], [name], y, pos_id, seed_id)}
        if name != "d1_mm":
            preds[name]["with_d1"] = two_way_bootstrap([d1 / 10.0, z], ["nearest_demo_per_10mm", name], y, pos_id, seed_id)
    out["predictors"] = preds
    # ---- support geometry: surrounded vs one-sided at matched d1 = 7.5 mm
    s_rows = [r for r in allrows if r["condition"] == "r7.5"]; o_rows = [r for r in allrows if r["condition"] == "r7.5_onesided"]
    if s_rows and o_rows:
        ks, ns = sum(r["success"] for r in s_rows), len(s_rows); ko, no = sum(r["success"] for r in o_rows), len(o_rows)
        out["support_geometry"] = {"surrounded": {"successes": ks, "trials": ns, "rate": ks / ns, "wilson95_naive": wilson(ks, ns),
                                                  "per_seed": [sum(r["success"] for r in s_rows if r["seed"] == s) for s in SEEDS]},
                                   "one_sided": {"successes": ko, "trials": no, "rate": ko / no, "wilson95_naive": wilson(ko, no),
                                                 "per_seed": [sum(r["success"] for r in o_rows if r["seed"] == s) for s in SEEDS]},
                                   "fisher_exact_rollouts": fisher_exact(ks, ns - ks, ko, no - ko),
                                   "note": "rollouts are clustered by position and seed, so the Fisher p-value is anti-conservative; per-seed counts are shown"}
    # ---- variance decomposition
    yv = np.array([r["success"] for r in allrows], float)
    cond_id = np.array([r["condition"] for r in allrows]); pid = np.array([f"{r['condition']}#{r['episode']}" for r in allrows]); sid = np.array([r["seed"] for r in allrows])
    grand = yv.mean(); eff = lambda ids: float(np.mean([(yv[ids == u].mean() - grand) ** 2 for u in np.unique(ids)]))
    ce, pe, se = eff(cond_id), eff(pid), eff(sid)
    out["variance_decomposition"] = {"total": float(yv.var()), "between_condition": ce, "between_position": pe, "between_seed": se,
                                     "shares_of_explained": {"condition": ce / (ce + pe + se), "position": pe / (ce + pe + se), "seed": se / (ce + pe + se)}}
    out["train_scene_summary"] = {c: {"per_seed": [sum(r["success"] for r in trainrows if r["condition"] == c and r["seed"] == s) for s in SEEDS],
                                      "trials_per_seed": [sum(1 for r in trainrows if r["condition"] == c and r["seed"] == s) for s in SEEDS]}
                                  for c in design["conditions"]}
    out["rows"] = allrows
    (D / "summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    figures(out, design, Path(a.figures))
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "conditions")}, indent=1, default=float)[:3500])


def figures(out, design, adir):
    conds = [c for c in design["conditions"] if c in out["conditions"]]
    surrounded = [c for c in conds if out["conditions"][c]["geometry"] == "surrounded"]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    # 1: controlled curve with individual positions and seeds
    ax = axes[0]
    for c in surrounded:
        cc = out["conditions"][c]; d = cc["measured_d1_mm"]["median"]
        for i, (pos, v) in enumerate(cc["per_position"].items()):
            ax.scatter(d + rng.uniform(-.25, .25), 100 * v["successes"] / v["seeds"] + rng.uniform(-1.2, 1.2), s=22, color="C0", alpha=.55, zorder=3)
        ax.scatter([d], [100 * cc["rate"]], s=90, marker="_", color="k", zorder=4)
        ax.annotate(f"{cc['successes']}/{cc['trials']}", (d, 100 * cc["rate"]), textcoords="offset points", xytext=(4, 6), fontsize=7)
    xs = np.linspace(0, 23, 200); w = out["controlled_curve"]
    ax.plot(xs, 100 / (1 + np.exp(-(w["intercept"] + w["slope_per_10mm"] * xs / 10))), "C0-", lw=2, label="controlled fit")
    o = out["observational_curve"]
    ax.plot(xs, 100 / (1 + np.exp(-(o["logit_intercept"] + o["slope_per_10mm"] * xs / 10))), "C3--", lw=1.6, label="observational (previous phase)")
    ax.set_xlabel("measured distance to nearest demonstration (mm)"); ax.set_ylabel("success (%)"); ax.set_ylim(-6, 108); ax.grid(alpha=.3)
    ax.legend(fontsize=7.5); ax.set_title("controlled density response\ndots = evaluation positions (3 seeds each)", fontsize=9)
    # 2: per-condition with seeds visible
    ax = axes[1]
    for c in conds:
        cc = out["conditions"][c]; d = cc["measured_d1_mm"]["median"]; col = "C4" if cc["geometry"] == "one_sided" else "C0"
        n_pos = cc["evaluation_positions"]
        ax.scatter([d + (0.6 if cc["geometry"] == "one_sided" else 0)] * len(cc["per_seed_successes"]),
                   [100 * v / n_pos for v in cc["per_seed_successes"]], s=46, color=col, alpha=.85, zorder=3)
        ax.plot([d - .5, d + .5], [100 * cc["rate"]] * 2, color=col, lw=2)
    ax.set_xlabel("measured d1 (mm)"); ax.set_ylabel("success (%) per training seed"); ax.set_ylim(-6, 108); ax.grid(alpha=.3)
    ax.set_title("every training seed (blue = surrounded, purple = one-sided)", fontsize=9)
    # 3: failure modes vs density
    ax = axes[2]
    modes = sorted({m for c in surrounded for m in out["conditions"][c]["failures"]})
    bottom = np.zeros(len(surrounded))
    for m in modes:
        v = np.array([out["conditions"][c]["failures"].get(m, 0) / out["conditions"][c]["trials"] * 100 for c in surrounded])
        ax.bar([out["conditions"][c]["measured_d1_mm"]["median"] for c in surrounded], v, bottom=bottom, width=1.4, label=m)
        bottom += v
    ax.set_xlabel("measured d1 (mm)"); ax.set_ylabel("% of rollouts"); ax.grid(alpha=.3); ax.legend(fontsize=6.5); ax.set_title("failure modes vs density", fontsize=9)
    # 4: surrounded vs one-sided
    ax = axes[3]
    sg = out.get("support_geometry")
    if sg:
        for i, (lab, key, col) in enumerate((("surrounded\n(60° gap)", "surrounded", "C0"), ("one-sided\n(220° gap)", "one_sided", "C4"))):
            v = sg[key]; ax.bar([i], [100 * v["rate"]], color=col, width=.6)
            ax.scatter([i] * len(v["per_seed"]), [100 * s / (v["trials"] / len(SEEDS)) for s in v["per_seed"]], color="k", s=30, zorder=3)
            ax.annotate(f"{v['successes']}/{v['trials']}", (i, 100 * v["rate"]), ha="center", textcoords="offset points", xytext=(0, 5), fontsize=8)
        ax.set_xticks([0, 1], ["surrounded\n(60° gap)", "one-sided\n(220° gap)"]); ax.set_ylabel("success (%)"); ax.set_ylim(0, 108)
        ax.set_title(f"same d1 = 7.5 mm, different support\nFisher p = {sg['fisher_exact_rollouts']['p_two_sided']:.3f}", fontsize=9); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(adir / "v2_density_response.png", dpi=125); plt.close(fig)

    # position x seed heatmap
    fig, axes = plt.subplots(1, len(conds), figsize=(2.3 * len(conds), 5.4))
    for ax, c in zip(np.atleast_1d(axes), conds):
        cc = out["conditions"][c]; pos = sorted(cc["per_position"])
        M = np.full((len(pos), len(SEEDS)), np.nan)
        for i, pp in enumerate(pos):
            for j, s in enumerate(SEEDS):
                r = [q for q in out["rows"] if q["condition"] == c and q["seed"] == s and q["episode"] == pp]
                if r: M[i, j] = float(r[0]["success"])
        ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto"); ax.set_xticks(range(len(SEEDS)), [f"s{s}" for s in SEEDS], fontsize=7)
        ax.set_title(f"{c}\n{cc['successes']}/{cc['trials']}", fontsize=8); ax.set_yticks(range(len(pos)), [str(p) for p in pos], fontsize=6)
    fig.suptitle("evaluation position x training seed", fontsize=10); fig.tight_layout()
    fig.savefig(adir / "v2_density_positions.png", dpi=125); plt.close(fig)


if __name__ == "__main__":
    main()
