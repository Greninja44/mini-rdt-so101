"""Analysis for the training-seed replication study (docs/research/training_seed_replication_spec.md).

Quantifies how much closed-loop behaviour is a property of the data geometry and how much is a property of one training run:
- per-scale training-seed variance, with every seed visible;
- the TRAIN40 category-B anomaly as a position x seed matrix;
- per-position success frequency across independently trained policies;
- the coverage model refitted on all replicates with a two-way cluster bootstrap (positions and seeds);
- distance response and a generalization curve with uncertainty;
- offline MAE and training-scene success as predictors of held-out success.

Repeated measurements are respected: rollouts sharing a position or a seed are never treated as independent.
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

from evaluation.generalization_analysis import classify, wilson

R = Path("artifacts/seed_replication")
S_PREV = Path("artifacts/exposure_matched_scaling")
G = Path("artifacts/physics_v2_generalization")
SCALES = (10, 20, 40, 80)
SEEDS = (17, 1, 2, 3, 4)
SUBSET = {10: "CLEAN10", 20: "TRAIN20", 40: "TRAIN40", 80: "TRAIN80"}
CATS = ("A_interpolation", "B_sparse_interpolation", "C_extrapolation")
BINS = [0, 2.5, 5, 7.5, 10, 15, np.inf]
BIN_LABELS = ["0-2.5", "2.5-5", "5-7.5", "7.5-10", "10-15", ">15"]
COLORS = {10: "C3", 20: "C1", 40: "C2", 80: "C0"}
rng = np.random.default_rng(0)


def paths(scale, seed):
    """Rollout/offline/model locations, including the seed-17 replicate produced by the previous phase."""
    if seed != 17:
        m = R / f"models/train{scale}_seed{seed}"
        return {"model": m, "test": R / f"closed_loop/train{scale}_seed{seed}", "train_scenes": R / f"train_scenes/train{scale}_seed{seed}",
                "offline": lambda part: R / f"offline/train{scale}_seed{seed}_{part}.json"}
    if scale == 80:
        return {"model": G / "models/TRAIN80_80k", "test": G / "closed_loop/TRAIN80_80k", "train_scenes": G / "diagnostic_train_scenes/TRAIN80_80k",
                "offline": lambda part: G / f"offline/TRAIN80_80k_{part}.json"}
    name = {10: "TRAIN10_e", 20: "TRAIN20_e", 40: "TRAIN40_e"}[scale]
    return {"model": S_PREV / f"models/{name}", "test": S_PREV / f"closed_loop/{name}", "train_scenes": S_PREV / f"train_scenes/{name}",
            "offline": lambda part: S_PREV / f"offline/{name}_{part}.json"}


def load_rollouts(directory, positions=None):
    rows = []
    for f in sorted(glob.glob(str(directory / "ep*_k8.json"))):
        r = json.loads(Path(f).read_text())
        if positions is not None and r["episode"] not in positions: continue
        rec = np.load(f.replace(".json", ".npz")); cat, flags, close, geo = classify(r, rec)
        row = {"episode": r["episode"], "success": bool(r["success"]), "outcome": cat, "steps": r["steps"], "close": close,
               "invalid_reason": r.get("invalid_reason"), "any_vertical_pad_contact": r.get("any_vertical_pad_contact"),
               **{k: r.get(k) for k in ("max_robot_table_penetration_mm", "max_cube_table_penetration_mm", "max_pad_cube_penetration_mm")}}
        if positions is not None: row.update(category=positions[r["episode"]]["category"], nn_mm=positions[r["episode"]]["nn_mm"])
        rows.append(row)
    return rows


def logistic_fit(X, y, ridge=1e-3, iters=200):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w)); W = np.clip(p * (1 - p), 1e-9, None)
        H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1]); g = X.T @ (y - p) - ridge * w
        step = np.linalg.solve(H, g); w += step
        if np.abs(step).max() < 1e-9: break
    return w


def two_way_bootstrap(cols, names, y, position, seed_id, B=2000):
    """Cluster bootstrap that resamples held-out positions AND training seeds (both are repeated-measure dimensions)."""
    X = np.c_[np.ones(len(y)), np.column_stack(cols)] if cols else np.ones((len(y), 1))
    w = logistic_fit(X, y); pos_u, seed_u = np.unique(position), np.unique(seed_id); boots = []
    index = {(p, s): np.where((position == p) & (seed_id == s))[0] for p in pos_u for s in seed_u}
    for _ in range(B):
        pp = rng.choice(pos_u, len(pos_u), replace=True); ss = rng.choice(seed_u, len(seed_u), replace=True)
        idx = np.concatenate([index[(p, s)] for p in pp for s in ss if len(index[(p, s)])])
        if len(set(y[idx])) < 2: continue
        boots.append(logistic_fit(X[idx], y[idx]))
    boots = np.array(boots)
    return {n: {"coef": float(w[i]), "bootstrap95": [float(np.percentile(boots[:, i], 2.5)), float(np.percentile(boots[:, i], 97.5))]}
            for i, n in enumerate(["intercept"] + names)} | {"n_rollouts": int(len(y)), "n_positions": int(len(pos_u)), "n_seeds": int(len(seed_u))}


def corr_with_ci(x, y, B=5000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rk = lambda v: np.argsort(np.argsort(v)).astype(float)
    pear = float(np.corrcoef(x, y)[0, 1]); spear = float(np.corrcoef(rk(x), rk(y))[0, 1]); bs = []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x))
        if np.std(x[i]) > 0 and np.std(y[i]) > 0: bs.append(np.corrcoef(x[i], y[i])[0, 1])
    return {"pearson": pear, "spearman": spear, "bootstrap95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": len(x)}


def permutation_paired(a, b, B=20000):
    """Paired permutation test on per-position success frequencies (sign-flip of the paired differences)."""
    d = np.asarray(a, float) - np.asarray(b, float); obs = d.mean()
    flips = rng.choice([-1.0, 1.0], (B, len(d)))
    null = (flips * d).mean(1)
    return {"mean_difference": float(obs), "p_two_sided": float((np.abs(null) >= abs(obs) - 1e-12).mean()), "n_positions": int(len(d))}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--split", default="docs/research/physics_v2_generalization_split.json")
    p.add_argument("--coverage", default="artifacts/exposure_matched_scaling/coverage.json"); p.add_argument("--figures", default="docs/assets")
    a = p.parse_args(); split = json.loads(Path(a.split).read_text()); cov = json.loads(Path(a.coverage).read_text())
    control = json.loads((G / "expert_control.json").read_text()); inb = {r["index"] for r in control["test_rows"] if r["in_benchmark"]}
    base = {r["index"]: r for r in split["test"] if r["index"] in inb}
    out = {"seeds": list(SEEDS), "scales": list(SCALES), "models": {}, "missing_cells": []}
    allrows = []
    for scale in SCALES:
        d = cov["subsets"][SUBSET[scale]]["per_position_mm"]
        positions = {i: {**base[i], "nn_mm": d[str(i)]} for i in base}
        for seed in SEEDS:
            pth = paths(scale, seed); rows = load_rollouts(pth["test"], positions)
            if len(rows) < len(positions):
                out["missing_cells"].append({"scale": scale, "seed": seed, "rollouts": len(rows)})
                if not rows: continue
            ts = load_rollouts(pth["train_scenes"])
            c10 = set(split["train_subsets"]["CLEAN10"]); ts = [r for r in ts if r["episode"] in c10]
            offline = {}
            for part in ("train", "val", "test"):
                f = pth["offline"](part)
                if Path(f).exists():
                    r = json.loads(Path(f).read_text()); offline[part] = r["all"]["action_mae"]; offline[f"{part}_start"] = r["episode_start"]["action_mae"]
            key = f"train{scale}_seed{seed}"
            out["models"][key] = {"scale": scale, "seed": seed, "heldout_total": sum(r["success"] for r in rows), "n": len(rows),
                                  "by_category": {c: [sum(r["success"] for r in rows if r["category"] == c), sum(r["category"] == c for r in rows)] for c in CATS},
                                  "train_scenes": [sum(r["success"] for r in ts), len(ts)], "offline": offline,
                                  "failures": {o: sum(r["outcome"] == o for r in rows if not r["success"]) for o in sorted({r["outcome"] for r in rows if not r["success"]})},
                                  "validity": {"successes_with_invalid_reason": sum(r["success"] and r["invalid_reason"] is not None for r in rows),
                                               "invalid_physics_rollouts": sum(r["invalid_reason"] is not None for r in rows),
                                               "max_robot_table_penetration_mm": max((r["max_robot_table_penetration_mm"] or 0) for r in rows),
                                               "max_cube_table_penetration_mm": max((r["max_cube_table_penetration_mm"] or 0) for r in rows),
                                               "max_pad_cube_penetration_mm": max((r["max_pad_cube_penetration_mm"] or 0) for r in rows)}}
            for r in rows: allrows.append({**r, "scale": scale, "seed": seed})
    # ---- per-scale variance
    out["by_scale"] = {}
    for scale in SCALES:
        vals = [out["models"][f"train{scale}_seed{s}"]["heldout_total"] for s in SEEDS if f"train{scale}_seed{s}" in out["models"]]
        ts = [out["models"][f"train{scale}_seed{s}"]["train_scenes"][0] for s in SEEDS if f"train{scale}_seed{s}" in out["models"]]
        if not vals: continue
        boots = [float(np.mean(rng.choice(vals, len(vals), replace=True))) for _ in range(10000)]
        cat = {c: [out["models"][f"train{scale}_seed{s}"]["by_category"][c][0] for s in SEEDS if f"train{scale}_seed{s}" in out["models"]] for c in CATS}
        out["by_scale"][scale] = {"seeds_present": len(vals), "per_seed": vals, "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                                  "median": float(np.median(vals)), "min": int(min(vals)), "max": int(max(vals)),
                                  "bootstrap95_mean": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                  "per_seed_by_category": cat, "category_mean": {c: float(np.mean(v)) for c, v in cat.items()},
                                  "category_sd": {c: (float(np.std(v, ddof=1)) if len(v) > 1 else None) for c, v in cat.items()},
                                  "train_scenes_per_seed": ts, "train_scenes_range": [int(min(ts)), int(max(ts))] if ts else None}
    # ---- position x seed structure
    freq = {}
    for scale in SCALES:
        m = {}
        for ep in sorted(base):
            v = [r["success"] for r in allrows if r["scale"] == scale and r["episode"] == ep]
            if v: m[ep] = {"successes": int(sum(v)), "seeds": len(v), "nn_mm": cov["subsets"][SUBSET[scale]]["per_position_mm"][str(ep)], "category": base[ep]["category"]}
        freq[scale] = m
    out["position_frequency"] = freq
    out["position_frequency_histogram"] = {scale: {k: sum(1 for v in m.values() if v["successes"] == k) for k in range(0, max((v["seeds"] for v in m.values()), default=0) + 1)} for scale, m in freq.items()}
    out["train40_category_B_matrix"] = {str(ep): {"nn_mm": freq[40][ep]["nn_mm"],
                                                  "per_seed": {str(s): next((r["success"] for r in allrows if r["scale"] == 40 and r["seed"] == s and r["episode"] == ep), None) for s in SEEDS}}
                                        for ep in sorted(base) if base[ep]["category"] == "B_sparse_interpolation" and ep in freq[40]}
    # ---- variance decomposition (share of outcome variance from position vs seed), per scale
    out["variance_decomposition"] = {}
    for scale in SCALES:
        sub = [r for r in allrows if r["scale"] == scale]
        if not sub: continue
        y = np.array([r["success"] for r in sub], float); pos = np.array([r["episode"] for r in sub]); sd = np.array([r["seed"] for r in sub])
        grand = y.mean()
        pos_eff = np.mean([(y[pos == p].mean() - grand) ** 2 for p in np.unique(pos)])
        seed_eff = np.mean([(y[sd == s].mean() - grand) ** 2 for s in np.unique(sd)])
        out["variance_decomposition"][scale] = {"total_variance": float(y.var()), "between_position": float(pos_eff), "between_seed": float(seed_eff),
                                                "position_share_of_explained": float(pos_eff / (pos_eff + seed_eff)) if pos_eff + seed_eff > 0 else None}
    # ---- coverage model on all replicates
    y = np.array([r["success"] for r in allrows], float); pos = np.array([r["episode"] for r in allrows]); sd = np.array([r["seed"] for r in allrows])
    dist = np.array([r["nn_mm"] for r in allrows]) / 10.0; ldem = np.log2(np.array([r["scale"] for r in allrows]) / 10.0)
    out["coverage_model"] = {"distance_only": two_way_bootstrap([dist], ["nearest_demo_per_10mm"], y, pos, sd),
                             "demos_only": two_way_bootstrap([ldem], ["log2_demos_doubling"], y, pos, sd),
                             "both": two_way_bootstrap([dist, ldem], ["nearest_demo_per_10mm", "log2_demos_doubling"], y, pos, sd)}
    # ---- paired scale comparisons on per-position frequencies
    out["paired_scale_tests"] = {}
    for x, z in ((10, 20), (20, 40), (40, 80), (10, 80)):
        common = [ep for ep in base if ep in freq[x] and ep in freq[z]]
        if common:
            fa = [freq[z][ep]["successes"] / freq[z][ep]["seeds"] for ep in common]; fb = [freq[x][ep]["successes"] / freq[x][ep]["seeds"] for ep in common]
            out["paired_scale_tests"][f"{z} vs {x} demos"] = permutation_paired(fa, fb)
    # ---- distance response and generalization curve
    out["distance_bins"] = {}
    for lo, hi, lab in zip(BINS[:-1], BINS[1:], BIN_LABELS):
        sel = [r for r in allrows if lo <= r["nn_mm"] < hi]
        k, n = sum(r["success"] for r in sel), len(sel)
        pos_n = len({r["episode"] for r in sel})
        out["distance_bins"][lab] = {"trials": n, "successes": k, "rate": k / n if n else None, "positions": pos_n,
                                     "wilson95_naive": wilson(k, n), "note": "trials share positions and seeds; the Wilson interval is naive"}
    w = logistic_fit(np.c_[np.ones(len(y)), dist], y)
    curve_boot = []
    for _ in range(2000):
        pp = rng.choice(np.unique(pos), len(np.unique(pos)), replace=True)
        idx = np.concatenate([np.where(pos == q)[0] for q in pp])
        if len(set(y[idx])) == 2: curve_boot.append(logistic_fit(np.c_[np.ones(len(idx)), dist[idx]], y[idx]))
    curve_boot = np.array(curve_boot)
    def dist_at(prob, ws):
        return 10.0 * (np.log(prob / (1 - prob)) - ws[:, 0]) / ws[:, 1]
    out["generalization_curve"] = {"logit_intercept": float(w[0]), "slope_per_10mm": float(w[1]),
                                   "distance_mm_at_probability": {str(pr): {"estimate": float(10.0 * (np.log(pr / (1 - pr)) - w[0]) / w[1]),
                                                                            "bootstrap95": [float(np.percentile(dist_at(pr, curve_boot), 2.5)), float(np.percentile(dist_at(pr, curve_boot), 97.5))]}
                                                                  for pr in (0.9, 0.75, 0.5, 0.25, 0.1)},
                                   "caveat": "a smooth logistic fit; this is not evidence of a hard threshold or generalization radius"}
    # ---- predictors of held-out success across models
    ms = [m for m in out["models"].values() if m["offline"].get("test") is not None]
    if len(ms) > 3:
        succ = [m["heldout_total"] for m in ms]
        out["predictors_of_heldout_success"] = {
            "train_scene_success": corr_with_ci([m["train_scenes"][0] for m in ms], succ),
            "offline_test_mae": corr_with_ci([m["offline"]["test"] for m in ms], succ),
            "offline_val_mae": corr_with_ci([m["offline"]["val"] for m in ms], succ),
            "offline_train_mae": corr_with_ci([m["offline"]["train"] for m in ms], succ),
            "n_models": len(ms)}
    # ---- extrapolation and failure modes
    csucc = [r for r in allrows if r["category"] == "C_extrapolation" and r["success"]]
    out["extrapolation"] = {"successes": len(csucc), "trials": sum(1 for r in allrows if r["category"] == "C_extrapolation"),
                            "nn_mm": {"min": float(min((r["nn_mm"] for r in csucc), default=np.nan)), "median": float(np.median([r["nn_mm"] for r in csucc])) if csucc else None,
                                      "max": float(max((r["nn_mm"] for r in csucc), default=np.nan))},
                            "by_scale": {scale: sum(r["success"] for r in allrows if r["scale"] == scale and r["category"] == "C_extrapolation") for scale in SCALES}}
    out["failure_modes_by_scale"] = {scale: {o: sum(1 for r in allrows if r["scale"] == scale and not r["success"] and r["outcome"] == o)
                                             for o in sorted({r["outcome"] for r in allrows if not r["success"]})} for scale in SCALES}
    same = []
    for ep in sorted(base):
        for scale in SCALES:
            outs = [r["outcome"] for r in allrows if r["episode"] == ep and r["scale"] == scale and not r["success"]]
            if len(outs) >= 2: same.append(len(set(outs)) == 1)
    out["failures_same_mode_across_seeds"] = {"position_scale_cells": len(same), "identical_mode": int(sum(same)), "fraction": float(np.mean(same)) if same else None}
    out["rows"] = allrows
    R.mkdir(parents=True, exist_ok=True); (R / "summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    figures(out, freq, Path(a.figures))
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "position_frequency", "train40_category_B_matrix", "models")}, indent=1, default=float)[:4000])


def figures(out, freq, adir):
    present = [s for s in SCALES if s in out["by_scale"]]
    # 1+2: totals and categories, every seed visible
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for ax, key, title, denom in zip(axes, ["total"] + list(CATS), ["held-out total (56)", "A interpolation (20)", "B sparse (20)", "C extrapolation (16)"], [56, 20, 20, 16]):
        for scale in present:
            v = out["by_scale"][scale]["per_seed"] if key == "total" else out["by_scale"][scale]["per_seed_by_category"][key]
            x = np.full(len(v), scale) * np.exp(rng.uniform(-.06, .06, len(v)))
            ax.scatter(x, [100 * q / denom for q in v], s=42, color=COLORS[scale], alpha=.85, zorder=3)
            m = 100 * np.mean(v) / denom; ax.plot([scale * .82, scale * 1.18], [m, m], color=COLORS[scale], lw=2.4, zorder=2)
            if len(v) > 1:
                sd = 100 * np.std(v, ddof=1) / denom; ax.errorbar([scale], [m], yerr=[sd], color=COLORS[scale], capsize=4, lw=1.2, zorder=1)
        ax.set_xscale("log", base=2); ax.set_xticks(present, [str(s) for s in present]); ax.set_xlabel("training demonstrations")
        ax.set_ylabel("success (%)"); ax.set_ylim(-5, 108); ax.grid(alpha=.3); ax.set_title(title + "\npoints = training seeds, bar = mean, whisker = SD", fontsize=9)
    fig.tight_layout(); fig.savefig(adir / "v2_seed_variance.png", dpi=125); plt.close(fig)

    # 3: TRAIN40 category-B position x seed heatmap (and the other scales for context)
    bpos = sorted([ep for ep, v in freq[40].items() if v["category"] == "B_sparse_interpolation"], key=lambda e: freq[40][e]["nn_mm"])
    if bpos:
        fig, axes = plt.subplots(1, len(present), figsize=(3.1 * len(present), 6.4), sharey=True)
        for ax, scale in zip(np.atleast_1d(axes), present):
            M = np.full((len(bpos), len(SEEDS)), np.nan)
            for i, ep in enumerate(bpos):
                for j, s in enumerate(SEEDS):
                    r = [q for q in out["rows"] if q["scale"] == scale and q["seed"] == s and q["episode"] == ep]
                    if r: M[i, j] = float(r[0]["success"])
            ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(SEEDS)), [f"s{s}" for s in SEEDS], fontsize=7)
            ax.set_title(f"{scale} demos\nB successes {int(np.nansum(M))}/{int(np.isfinite(M).sum())}", fontsize=9)
            if scale == present[0]:
                ax.set_yticks(range(len(bpos)), [f"ep{ep} ({freq[40][ep]['nn_mm']:.0f}mm)" for ep in bpos], fontsize=6.5)
        fig.suptitle("category B (sparse hole): position x training seed", fontsize=10)
        fig.tight_layout(); fig.savefig(adir / "v2_seed_trainB_matrix.png", dpi=125); plt.close(fig)

    # 4-7: distance response, predictors, position stability
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    ax = axes[0]; centers = [1.25, 3.75, 6.25, 8.75, 12.5, 18]
    ks = [out["distance_bins"][l] for l in BIN_LABELS]
    xs = [c for c, b in zip(centers, ks) if b["trials"]]; ys = [100 * b["rate"] for b in ks if b["trials"]]
    err = np.clip(np.array([[100 * (b["rate"] - b["wilson95_naive"][0]), 100 * (b["wilson95_naive"][1] - b["rate"])] for b in ks if b["trials"]]).T, 0, None)
    ax.errorbar(xs, ys, yerr=err, fmt="o-", capsize=3, color="C0")
    for c, b in zip(centers, ks):
        if b["trials"]: ax.annotate(f"{b['successes']}/{b['trials']}", (c, 100 * b["rate"]), textcoords="offset points", xytext=(5, 4), fontsize=7)
    d = np.linspace(0, 22, 100); w0, w1 = out["generalization_curve"]["logit_intercept"], out["generalization_curve"]["slope_per_10mm"]
    ax.plot(d, 100 / (1 + np.exp(-(w0 + w1 * d / 10))), "k--", lw=1, label="logistic fit")
    ax.set_xticks(centers, BIN_LABELS); ax.set_xlabel("nearest training cube (mm)"); ax.set_ylabel("success (%), all replicates")
    ax.set_ylim(-5, 108); ax.grid(alpha=.3); ax.legend(fontsize=7); ax.set_title("distance response across all seeds", fontsize=9)

    ax = axes[1]
    for scale in present:
        ms = [out["models"][f"train{scale}_seed{s}"] for s in SEEDS if f"train{scale}_seed{s}" in out["models"]]
        ms = [m for m in ms if m["offline"].get("test") is not None]
        ax.scatter([m["offline"]["test"] for m in ms], [m["heldout_total"] for m in ms], s=48, color=COLORS[scale], label=f"{scale} demos")
    ax.set_xlabel("offline held-out action MAE"); ax.set_ylabel("held-out successes / 56"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    r = out.get("predictors_of_heldout_success", {}).get("offline_test_mae", {})
    ax.set_title(f"offline MAE vs closed loop\nPearson {r.get('pearson', float('nan')):.2f} [{r.get('bootstrap95', [float('nan')] * 2)[0]:.2f}, {r.get('bootstrap95', [float('nan')] * 2)[1]:.2f}]", fontsize=9)

    ax = axes[2]
    for scale in present:
        ms = [out["models"][f"train{scale}_seed{s}"] for s in SEEDS if f"train{scale}_seed{s}" in out["models"]]
        ax.scatter([m["train_scenes"][0] + rng.uniform(-.12, .12) for m in ms], [m["heldout_total"] for m in ms], s=48, color=COLORS[scale], label=f"{scale} demos")
    ax.set_xlabel("training-scene successes / 10"); ax.set_ylabel("held-out successes / 56"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    r = out.get("predictors_of_heldout_success", {}).get("train_scene_success", {})
    ax.set_title(f"training-scene diagnostic vs closed loop\nPearson {r.get('pearson', float('nan')):.2f} [{r.get('bootstrap95', [float('nan')] * 2)[0]:.2f}, {r.get('bootstrap95', [float('nan')] * 2)[1]:.2f}]", fontsize=9)

    ax = axes[3]
    for scale in present:
        m = freq[scale]; nn = [v["nn_mm"] for v in m.values()]; fr = [100 * v["successes"] / v["seeds"] for v in m.values()]
        ax.scatter(nn, np.array(fr) + rng.uniform(-1.5, 1.5, len(fr)), s=26, color=COLORS[scale], alpha=.7, label=f"{scale} demos")
    ax.set_xlabel("nearest training cube (mm)"); ax.set_ylabel("% of training seeds solving the position"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    ax.set_title("per-position stability across independently trained policies", fontsize=9)
    fig.tight_layout(); fig.savefig(adir / "v2_seed_analysis.png", dpi=125); plt.close(fig)


if __name__ == "__main__":
    main()
