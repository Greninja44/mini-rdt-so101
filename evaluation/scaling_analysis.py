"""Analysis for the exposure-matched data-scaling phase (docs/research/exposure_matched_data_scaling_spec.md).

Produces:
- the exposure-matched scaling curve (overall and per category);
- pre-defined distance-bin tables;
- the coverage-aware logistic models that separate "more data" from "denser local coverage";
- matched-distance comparisons;
- the offline-MAE versus closed-loop-success figure;
- the failure taxonomy and physical-validity summary.

Every statistic follows the pre-registered protocol: Wilson CIs, exact McNemar, and position-cluster bootstraps.
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

from evaluation.generalization_analysis import classify, mcnemar_exact, rate, wilson

S = Path("artifacts/exposure_matched_scaling")
G = Path("artifacts/physics_v2_generalization")
MODELS = ("TRAIN10_e", "TRAIN20_e", "TRAIN40_e", "TRAIN80_e")
DEMOS = {"TRAIN10_e": 10, "TRAIN20_e": 20, "TRAIN40_e": 40, "TRAIN80_e": 80}
SUBSET = {"TRAIN10_e": "CLEAN10", "TRAIN20_e": "TRAIN20", "TRAIN40_e": "TRAIN40", "TRAIN80_e": "TRAIN80"}
CATS = ("A_interpolation", "B_sparse_interpolation", "C_extrapolation")
BINS = [0, 2.5, 5, 7.5, 10, 15, np.inf]  # pre-registered, never adjusted
BIN_LABELS = ["0-2.5", "2.5-5", "5-7.5", "7.5-10", "10-15", ">15"]
rng = np.random.default_rng(0)


def rollout_dir(model, kind):
    if model == "TRAIN80_e":
        return G / ("closed_loop/TRAIN80_80k" if kind == "test" else "diagnostic_train_scenes/TRAIN80_80k")
    return S / ("closed_loop" if kind == "test" else "train_scenes") / model


def logistic_multi(X, y, ridge=1e-3, iters=200):
    """Newton IRLS with a small ridge (reported); X already includes the intercept column."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w)); W = np.clip(p * (1 - p), 1e-9, None)
        H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1]); g = X.T @ (y - p) - ridge * w
        step = np.linalg.solve(H, g); w += step
        if np.abs(step).max() < 1e-9: break
    return w


def fit_with_ci(cols, names, y, clusters, B=2000):
    """Logistic fit with a position-cluster bootstrap CI (each held-out position is one cluster)."""
    X = np.c_[np.ones(len(y)), np.column_stack(cols)] if cols else np.ones((len(y), 1))
    w = logistic_multi(X, y); uniq = np.unique(clusters); boots = []
    for _ in range(B):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.where(clusters == c)[0] for c in pick])
        yy = y[idx]
        if len(set(yy)) < 2: continue
        boots.append(logistic_multi(X[idx], yy))
    boots = np.array(boots)
    return {n: {"coef": float(w[i]), "bootstrap95": [float(np.percentile(boots[:, i], 2.5)), float(np.percentile(boots[:, i], 97.5))]}
            for i, n in enumerate(["intercept"] + names)} | {"n_rollouts": int(len(y)), "n_positions": int(len(uniq)), "ridge": 1e-3}


def load(model, positions, kind="test"):
    rows = []
    for f in sorted(glob.glob(str(rollout_dir(model, kind) / "ep*_k8.json"))):
        r = json.loads(Path(f).read_text())
        if kind == "test" and r["episode"] not in positions: continue
        rec = np.load(f.replace(".json", ".npz")); cat, flags, close, geo = classify(r, rec)
        row = {"model": model, "demos": DEMOS[model], "episode": r["episode"], "success": bool(r["success"]), "steps": r["steps"],
               "outcome": cat, "flags": flags, "close": close, "invalid_reason": r.get("invalid_reason"),
               "any_vertical_pad_contact": r.get("any_vertical_pad_contact"),
               **{k: r.get(k) for k in ("max_robot_table_penetration_mm", "max_cube_table_penetration_mm", "max_pad_cube_penetration_mm")}}
        if kind == "test": row.update(category=positions[r["episode"]]["category"], nn_mm=positions[r["episode"]]["nn_mm"])
        rows.append(row)
    return rows


def fig_curve(summary, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    n = [DEMOS[m] for m in MODELS]
    ax = axes[0]
    for label, key, col in (("overall (56)", "overall", "C0"), ("A interpolation (20)", "A_interpolation", "C1"),
                            ("B sparse (20)", "B_sparse_interpolation", "C2"), ("C extrapolation (16)", "C_extrapolation", "C4")):
        v = [summary["models"][m]["heldout"][key] for m in MODELS]
        ax.errorbar(n, [100 * x["rate"] for x in v], yerr=np.array([[100 * (x["rate"] - x["wilson95"][0]), 100 * (x["wilson95"][1] - x["rate"])] for x in v]).T,
                    fmt="o-", capsize=3, color=col, label=label)
        for xx, x in zip(n, v): ax.annotate(f"{x['k']}/{x['n']}", (xx, 100 * x["rate"]), textcoords="offset points", xytext=(5, 4), fontsize=7, color=col)
    ax.set_xscale("log", base=2); ax.set_xticks(n, [str(x) for x in n]); ax.set_xlabel("training demonstrations (nested, exposure matched)")
    ax.set_ylabel("held-out success at K=8 (%)"); ax.set_ylim(-5, 108); ax.grid(alpha=.3); ax.legend(fontsize=7.5); ax.set_title("exposure-matched data scaling", fontsize=10)

    ax = axes[1]
    centers = [1.25, 3.75, 6.25, 8.75, 12.5, 18]
    for m, col in zip(MODELS, ("C3", "C1", "C2", "C0")):
        b = summary["models"][m]["distance_bins"]
        xs = [c for c, l in zip(centers, BIN_LABELS) if b[l]["n"]]; ys = [100 * b[l]["rate"] for l in BIN_LABELS if b[l]["n"]]
        ax.plot(xs, ys, "o-", color=col, label=f"{DEMOS[m]} demos")
        for c, l in zip(centers, BIN_LABELS):
            if b[l]["n"]: ax.annotate(f"{b[l]['k']}/{b[l]['n']}", (c, 100 * b[l]["rate"]), textcoords="offset points", xytext=(4, 3), fontsize=6, color=col)
    ax.set_xticks(centers, BIN_LABELS); ax.set_xlabel("distance to nearest training cube (mm)"); ax.set_ylabel("success (%)")
    ax.set_ylim(-5, 108); ax.grid(alpha=.3); ax.legend(fontsize=7.5); ax.set_title("success vs local coverage, per data scale", fontsize=10)

    ax = axes[2]
    for m, col in zip(MODELS, ("C3", "C1", "C2", "C0")):
        o = summary["models"][m]["offline"]; h = summary["models"][m]["heldout"]["overall"]
        if o.get("test") is None: continue
        ax.scatter(o["test"], 100 * h["rate"], s=70, color=col, label=f"{DEMOS[m]} demos", zorder=3)
        ax.annotate(f"{DEMOS[m]}", (o["test"], 100 * h["rate"]), textcoords="offset points", xytext=(7, -3), fontsize=8, color=col)
    prev = summary.get("fixed_step_reference", [])
    if prev:
        ax.scatter([p["offline_test_mae"] for p in prev], [100 * p["rate"] for p in prev], s=50, facecolor="none", edgecolor="0.5", label="fixed-step budget (previous phase)")
    ax.set_xlabel("held-out offline action MAE"); ax.set_ylabel("held-out closed-loop success (%)"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    ax.set_title("offline error vs closed-loop success", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "v2_scaling_curve.png", dpi=125); plt.close(fig)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--split", default="docs/research/physics_v2_generalization_split.json")
    p.add_argument("--coverage", default="artifacts/exposure_matched_scaling/coverage.json"); p.add_argument("--figures", default="docs/assets")
    a = p.parse_args(); split = json.loads(Path(a.split).read_text()); cov = json.loads(Path(a.coverage).read_text())
    control = json.loads((G / "expert_control.json").read_text()); inb = {r["index"] for r in control["test_rows"] if r["in_benchmark"]}
    base = {r["index"]: r for r in split["test"] if r["index"] in inb}
    out = {"exposure": json.loads(Path("docs/research/physics_v2_generalization_freeze.json").read_text())["train_dataset"]["subset_windows"],
           "budgets": {"TRAIN10_e": 10055, "TRAIN20_e": 20037, "TRAIN40_e": 40018, "TRAIN80_e": 80000}, "models": {}}
    allrows = []
    for m in MODELS:
        d = cov["subsets"][SUBSET[m]]["per_position_mm"]
        positions = {i: {**base[i], "nn_mm": d[str(i)]} for i in base}
        rows = load(m, positions); allrows += rows
        train_rows = load(m, positions, kind="train")
        c10 = set(split["train_subsets"]["CLEAN10"])
        offline = {}
        for part in ("train", "val", "test"):
            f = (S / "offline" / f"{m}_{part}.json") if m != "TRAIN80_e" else (G / "offline" / f"TRAIN80_80k_{part}.json")
            if f.exists():
                r = json.loads(f.read_text())
                offline[part] = r["all"]["action_mae"]; offline[f"{part}_start"] = r["episode_start"]["action_mae"]
                offline[f"{part}_per_joint"] = [round(x, 5) for x in r["all"]["per_joint_mae"]]
                if part == "test": offline["test_phase"] = {k: round(v["action_mae"], 5) for k, v in r["expert_phase"].items()}
        by_bin = {}
        for lo, hi, lab in zip(BINS[:-1], BINS[1:], BIN_LABELS):
            sel = [r for r in rows if lo <= r["nn_mm"] < hi]; by_bin[lab] = rate([r["success"] for r in sel])
        out["models"][m] = {
            "demos": DEMOS[m], "subset": SUBSET[m], "windows": cov["subsets"][SUBSET[m]]["windows"],
            "steps": out["budgets"][m], "exposure_passes": round(out["budgets"][m] * 8 / cov["subsets"][SUBSET[m]]["windows"], 3),
            "nearest_demo_mm": {k: cov["subsets"][SUBSET[m]]["nearest_test_to_train_mm"][k] for k in ("median_mm", "mean_mm", "max_mm", "count_within")},
            "heldout": {"overall": rate([r["success"] for r in rows]), **{c: rate([r["success"] for r in rows if r["category"] == c]) for c in CATS}},
            "distance_bins": by_bin,
            "train_scenes": {"clean10_scenes": rate([r["success"] for r in train_rows if r["episode"] in c10]),
                             "all_own_diagnostic_scenes": rate([r["success"] for r in train_rows])},
            "offline": offline,
            "failures": {c: {o: sum(r["outcome"] == o for r in rows if r["category"] == c and not r["success"]) for o in sorted({r["outcome"] for r in rows if r["category"] == c and not r["success"]})} for c in CATS},
            "failure_totals": {o: sum(r["outcome"] == o for r in rows if not r["success"]) for o in sorted({r["outcome"] for r in rows if not r["success"]})},
            "validity": {"successes": sum(r["success"] for r in rows), "successes_with_invalid_reason": sum(r["success"] and r["invalid_reason"] is not None for r in rows),
                         "invalid_physics_rollouts": sum(r["invalid_reason"] is not None for r in rows),
                         "successes_with_vertical_pad_contact": sum(r["success"] and bool(r["any_vertical_pad_contact"]) for r in rows),
                         "max_robot_table_penetration_mm": max((r["max_robot_table_penetration_mm"] or 0) for r in rows),
                         "max_cube_table_penetration_mm": max((r["max_cube_table_penetration_mm"] or 0) for r in rows),
                         "max_pad_cube_penetration_mm": max((r["max_pad_cube_penetration_mm"] or 0) for r in rows)},
            "close_lateral_mm": {k: float(np.median([r["close"]["lateral_mm"] for r in rows if r["close"] and r["success"] == v] or [np.nan]))
                                 for k, v in (("success", True), ("failure", False))},
            "steps_to_success_median": float(np.median([r["steps"] for r in rows if r["success"]])) if any(r["success"] for r in rows) else None}
    # paired comparisons on identical positions
    get = lambda m: {r["episode"]: r["success"] for r in allrows if r["model"] == m}
    out["paired_mcnemar"] = {}
    for x, y in (("TRAIN10_e", "TRAIN20_e"), ("TRAIN20_e", "TRAIN40_e"), ("TRAIN40_e", "TRAIN80_e"), ("TRAIN10_e", "TRAIN80_e")):
        A, B = get(x), get(y); common = sorted(set(A) & set(B))
        if common:
            r = mcnemar_exact([B[e] for e in common], [A[e] for e in common])
            out["paired_mcnemar"][f"{DEMOS[y]} vs {DEMOS[x]} demos"] = {"n": len(common), "successes": [sum(A[e] for e in common), sum(B[e] for e in common)],
                                                                       "improved_only_by_larger": r["only_first"], "lost_only_by_larger": r["only_second"], "p_two_sided": r["p_two_sided"]}
    # coverage-aware models (pooled over 4 x 56 rollouts, clustered on position)
    y = np.array([r["success"] for r in allrows], float); clusters = np.array([r["episode"] for r in allrows])
    dist = np.array([r["nn_mm"] for r in allrows]) / 10.0; ldem = np.log2(np.array([r["demos"] for r in allrows]) / 10.0)
    out["coverage_aware_logistic"] = {
        "distance_only": fit_with_ci([dist], ["nearest_demo_per_10mm"], y, clusters),
        "demos_only": fit_with_ci([ldem], ["log2_demos_doubling"], y, clusters),
        "both": fit_with_ci([dist, ldem], ["nearest_demo_per_10mm", "log2_demos_doubling"], y, clusters)}
    # matched-distance comparison: success by (bin, data scale)
    out["matched_distance_table"] = {lab: {m: out["models"][m]["distance_bins"][lab] for m in MODELS} for lab in BIN_LABELS}
    out["rows"] = allrows
    S.mkdir(parents=True, exist_ok=True); (S / "summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    fig_curve(out, Path(a.figures))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
