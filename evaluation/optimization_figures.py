"""Figures for the optimization scaling study that the pre-registered figure list requires beyond
the two training-diagnostic panels already produced by optimization_analysis.py.

Panels whose data does not exist yet (anything needing Stage F) render an explicit placeholder rather
than being silently omitted, so a figure set built mid-study cannot be mistaken for a complete one.
The frozen baseline comparison comes from the corrected capacity matrix of the recovery study; no
previous artifact is modified.
"""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
O = ROOT / "artifacts/optimization_scaling_v1"
ASSETS = ROOT / "docs/assets"
RECOVERY = ROOT / "artifacts/capacity_recovery_v1/summary.json"
CAP_ORDER = ["m2.0", "m4.3", "m9.1", "m19.5"]
PARAMS = {"m2.0": 2_009_670, "m4.3": 4_304_902, "m9.1": 9_137_286, "m19.5": 19_517_062}
COLORS = {"m2.0": "C3", "m4.3": "C1", "m9.1": "C2", "m19.5": "C0"}
CONDITIONS = ["r7.5", "r7.5_onesided", "r10.0", "r15.0", "r20.0"]
N_EVAL = {"r7.5": 10, "r7.5_onesided": 10, "r10.0": 8, "r15.0": 4, "r20.0": 3}
MAE_THRESHOLD = 0.01012


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def placeholder(ax, text):
    ax.text(.5, .5, text, ha="center", va="center", fontsize=8, color="0.35")
    ax.set_axis_off()


def fig_mae_by_capacity_recipe(runs, out):
    """Training MAE by capacity and recipe, against the gate threshold and the frozen per-seed baseline."""
    cells = json.loads(RECOVERY.read_text())["cells"] if RECOVERY.exists() else {}
    fig, ax = plt.subplots(figsize=(9, 4.4))
    recipes = sorted({r["recipe"] for r in runs.values()})
    width = 0.8 / max(1, len(recipes))
    for j, rec in enumerate(recipes):
        xs, ys = [], []
        for i, cap in enumerate(CAP_ORDER):
            m = [r for r in runs.values() if r["capacity"] == cap and r["recipe"] == rec]
            if m:
                xs.append(i + j * width - 0.4 + width / 2)
                ys.append(min(r["train_mae"] for r in m))
        if xs:
            ax.bar(xs, ys, width, label=rec, edgecolor="k", lw=.4)
    for i, cap in enumerate(CAP_ORDER):                      # frozen per-seed baseline spread
        c = cells.get(f"{cap}|r10.0")
        if c:
            ax.scatter([i] * len(c["train_mae"]), c["train_mae"], marker="_", s=260, color="k", zorder=5,
                       label="frozen baseline, per seed" if i == 0 else None)
    ax.axhline(MAE_THRESHOLD, color="crimson", ls=":", lw=1.4, label=f"gate threshold {MAE_THRESHOLD}")
    ax.set_xticks(range(len(CAP_ORDER)), CAP_ORDER)
    ax.set_xlabel("capacity"); ax.set_ylabel("final sampled training MAE")
    ax.set_title("training fit by capacity and optimizer recipe (condition r10, screening cell)", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(ASSETS / "v2_optimization_mae_by_recipe.png", dpi=125); plt.close(fig)


def fig_capacity_response(runs, out):
    """Capacity x optimization response: does the recipe's benefit grow with capacity?"""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    base = {c: min([r["train_mae"] for r in runs.values() if r["capacity"] == c and r["recipe"] == "baseline"] or [np.nan])
            for c in CAP_ORDER}
    for rec in sorted({r["recipe"] for r in runs.values()} - {"baseline"}):
        xs, ys = [], []
        for cap in CAP_ORDER:
            m = [r["train_mae"] for r in runs.values() if r["capacity"] == cap and r["recipe"] == rec]
            if m and np.isfinite(base[cap]):
                xs.append(PARAMS[cap]); ys.append(100 * (min(m) - base[cap]) / base[cap])
        if xs:
            ax.plot(xs, ys, "o-", lw=1.4, label=rec)
    ax.axhline(0, color="k", lw=1, ls="--", label="baseline recipe")
    ax.set_xscale("log"); ax.set_xlabel("trainable parameters"); ax.set_ylabel("change in training MAE vs baseline (%)")
    ax.set_title("capacity x optimization response (negative = recipe fits better)", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(ASSETS / "v2_optimization_capacity_response.png", dpi=125); plt.close(fig)


def fig_lateral(runs, out):
    """Lateral aiming error at first gripper close, by capacity and recipe."""
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    keys = [k for k, r in sorted(runs.items()) if r.get("lateral_at_close_mm")]
    if not keys:
        placeholder(ax, "no rollouts with a measured first close yet")
    else:
        vals = [runs[k]["lateral_at_close_mm"] for k in keys]
        ax.bar(range(len(keys)), vals, color=[COLORS.get(runs[k]["capacity"], "0.5") for k in keys], edgecolor="k", lw=.4)
        ax.set_xticks(range(len(keys)), [f"{runs[k]['capacity']}\n{runs[k]['recipe']}" for k in keys], fontsize=6.5)
        ax.set_ylabel("lateral error at first close (mm)")
        ax.set_title("aiming error by capacity and recipe (training scenes)", fontsize=9)
        ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(ASSETS / "v2_optimization_lateral.png", dpi=125); plt.close(fig)


def fig_stage_f(runs, out):
    """Held-out success by recipe and capacity, per condition, and one-sided versus surrounded."""
    f = {k: r for k, r in runs.items() if r["stage"] == "stageF" and r.get("heldout")}
    cells = json.loads(RECOVERY.read_text())["cells"] if RECOVERY.exists() else {}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    if not f:
        for ax, t in zip(axes, ["held-out success by recipe\n(Stage F not yet run)",
                                "per-condition results\n(Stage F not yet run)",
                                "one-sided vs surrounded\n(Stage F not yet run)"]):
            placeholder(ax, t)
    else:
        caps = [c for c in CAP_ORDER if any(r["capacity"] == c for r in f.values())]
        for i, cap in enumerate(caps):                       # panel 1: optimized vs frozen baseline
            for j, (src, color) in enumerate((("baseline", "0.6"), ("optimized", COLORS[cap]))):
                if src == "baseline":
                    tot = sum(sum(cells.get(f"{cap}|{c}", {}).get("per_seed", [])) for c in CONDITIONS)
                    n = sum(3 * N_EVAL[c] for c in CONDITIONS if f"{cap}|{c}" in cells)
                else:
                    tot = sum(r["heldout"][0] for r in f.values() if r["capacity"] == cap)
                    n = sum(r["heldout"][1] for r in f.values() if r["capacity"] == cap)
                p, lo, hi = wilson(tot, n)
                axes[0].bar(i + j * .38 - .19, 100 * p, .36, color=color, edgecolor="k", lw=.4,
                            yerr=[[100 * max(0, p - lo)], [100 * max(0, hi - p)]], capsize=3,
                            label=src if i == 0 else None)
        axes[0].set_xticks(range(len(caps)), caps); axes[0].set_ylabel("held-out success (%)")
        axes[0].set_title("pooled held-out: frozen baseline vs optimized recipe", fontsize=9)
        axes[0].legend(fontsize=7); axes[0].grid(alpha=.3, axis="y")

        for cap in caps:                                      # panel 2: per condition
            xs, ys, los, his = [], [], [], []
            for i, c in enumerate(CONDITIONS):
                m = [r for r in f.values() if r["capacity"] == cap and r["condition"] == c]
                if m:
                    p, lo, hi = wilson(sum(r["heldout"][0] for r in m), sum(r["heldout"][1] for r in m))
                    xs.append(i); ys.append(100 * p); los.append(100 * max(0, p - lo)); his.append(100 * max(0, hi - p))
            if xs:
                axes[1].errorbar(xs, ys, yerr=[los, his], fmt="o-", color=COLORS[cap], capsize=3, lw=1.3, label=f"{cap} optimized")
            bs = [100 * sum(cells[f"{cap}|{c}"]["per_seed"]) / (3 * N_EVAL[c]) for c in CONDITIONS if f"{cap}|{c}" in cells]
            if bs:
                axes[1].plot(range(len(bs)), bs, "s--", color=COLORS[cap], alpha=.45, lw=1.1, label=f"{cap} baseline")
        axes[1].set_xticks(range(len(CONDITIONS)), CONDITIONS, fontsize=7, rotation=20)
        axes[1].set_ylabel("held-out success (%)"); axes[1].set_title("per-condition results", fontsize=9)
        axes[1].legend(fontsize=6.5); axes[1].grid(alpha=.3)

        for i, cap in enumerate(caps):                        # panel 3: one-sided vs surrounded at r7.5
            for j, (c, hatch) in enumerate((("r7.5", ""), ("r7.5_onesided", "//"))):
                m = [r for r in f.values() if r["capacity"] == cap and r["condition"] == c]
                if m:
                    p, lo, hi = wilson(sum(r["heldout"][0] for r in m), sum(r["heldout"][1] for r in m))
                    axes[2].bar(i + j * .38 - .19, 100 * p, .36, color=COLORS[cap], hatch=hatch, edgecolor="k", lw=.4,
                                yerr=[[100 * max(0, p - lo)], [100 * max(0, hi - p)]], capsize=3,
                                label=("surrounded r7.5" if j == 0 else "one-sided r7.5") if i == 0 else None)
        axes[2].set_xticks(range(len(caps)), caps); axes[2].set_ylabel("held-out success (%)")
        axes[2].set_title("support geometry at matched radius, optimized recipe", fontsize=9)
        axes[2].legend(fontsize=7); axes[2].grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(ASSETS / "v2_optimization_heldout.png", dpi=125); plt.close(fig)


def fig_offline_vs_closed_loop(runs, out):
    """Offline MAE against closed-loop success: the secondary-diagnostic check."""
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    pts = [(r["train_mae"], 100 * r["train_scene"][0] / r["train_scene"][1], r)
           for r in runs.values() if r.get("train_scene") and r["train_scene"][1] >= 10]
    if len(pts) < 3:
        placeholder(ax, "not enough evaluated runs for a fit yet")
    else:
        for x, y, r in pts:
            ax.scatter(x, y, color=COLORS.get(r["capacity"], "0.5"),
                       marker="o" if r["recipe"] == "baseline" else "^", s=58, edgecolor="k", lw=.5)
        xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        rho = float(np.corrcoef(xs, ys)[0, 1])
        b, a = np.polyfit(xs, ys, 1)
        g = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(g, a + b * g, "k--", lw=1, label=f"least squares, r = {rho:+.2f}")
        ax.axvline(MAE_THRESHOLD, color="crimson", ls=":", lw=1.2, label="gate threshold")
        ax.set_xlabel("final sampled training MAE"); ax.set_ylabel("training-scene success (%)")
        ax.set_title("offline MAE vs closed-loop success (circles baseline, triangles candidate)", fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(ASSETS / "v2_optimization_offline_vs_loop.png", dpi=125); plt.close(fig)


def main():
    out = json.loads((O / "summary.json").read_text())
    runs = out["runs"]
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig_mae_by_capacity_recipe(runs, out)
    fig_capacity_response(runs, out)
    fig_lateral(runs, out)
    fig_stage_f(runs, out)
    fig_offline_vs_closed_loop(runs, out)
    print("wrote 5 figures to docs/assets/")


if __name__ == "__main__":
    main()
