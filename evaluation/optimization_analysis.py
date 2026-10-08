"""Analysis for the optimization scaling study (docs/research/optimization_scaling_spec.md).

Reads every run under `artifacts/optimization_scaling_v1/`, reconstructs its training diagnostics (loss, gradient norms, learning-rate
trajectory, EMA/raw divergence), applies the pre-registered training-health gate criterion by criterion, and — for runs that reached Stage F
— summarises held-out closed-loop performance.

Recipes are judged by the gate only. Held-out results are reported, never used for selection.
"""
from __future__ import annotations
import argparse
import glob
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.generalization_analysis import classify, wilson

O = Path("artifacts/optimization_scaling_v1")
# Gate constants, fixed by the pre-registration and Amendment 2 (derived from 2.0M/4.3M baselines only).
MAE_THRESHOLD = 0.01012          # 1.25 x the frozen 2.0M r10 three-seed mean
TAIL_MEDIAN_MAX = 0.159          # 3 x the 2.0M Stage-A tail median (0.053)
SPIKE_RATE_MAX = 0.034           # 3 x the 2.0M spike rate (0.81%) + 1 pp
TRAIN_SCENE_MIN = 0.90
CAP_ORDER = ["m2.0", "m4.3", "m9.1", "m19.5"]
COLORS = {"m2.0": "C3", "m4.3": "C1", "m9.1": "C2", "m19.5": "C0"}


def parse_name(name):
    """stageA_m2.0_r10.0_seed0 / stageB_m9.1_lr0.5x_r10.0_seed0 -> stage, capacity, recipe, condition, seed."""
    parts = name.split("_")
    stage, capacity = parts[0], parts[1]
    seed = int(parts[-1].replace("seed", "")); condition = parts[-2]
    recipe = "_".join(parts[2:-2]) or "baseline"
    return stage, capacity, recipe, condition, seed


def series(run):
    loss, ema, grad = [], [], []
    path = O / "models" / run / "metrics.jsonl"
    if not path.exists(): return loss, ema, grad
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if "grad_norm" in row: grad.append(row)
        elif "ema_sampled" in row: ema.append(row)
        elif "sampled" in row: loss.append(row)
    return loss, ema, grad


def rollouts(kind, run):
    out = []
    for f in sorted(glob.glob(str(O / kind / run / "ep*_k8.json"))):
        r = json.loads(Path(f).read_text()); rec = np.load(f.replace(".json", ".npz"))
        cat, flags, close, geo = classify(r, rec)
        out.append({"episode": r["episode"], "success": bool(r["success"]), "outcome": cat, "close": close, **geo,
                    "invalid_reason": r.get("invalid_reason")})
    return out


def evaluate_gate(run):
    loss, ema, grad = series(run)
    done = (O / "models" / run / "result.json").exists()
    if not (loss and done): return None
    result = json.loads((O / "models" / run / "result.json").read_text())
    train_mae = result["sampled"]["action_mae"]
    ema_mae = ema[-1]["ema_sampled"]["action_mae"] if ema else None
    g = [r["grad_norm"] for r in grad]
    med = st.median(g) if g else None
    tail = g[int(0.9 * len(g)):] if g else []
    tail_med = st.median(tail) if tail else None
    spike = (sum(1 for x in g if x > 10 * med) / len(g)) if g else None
    finite = all(np.isfinite(x) for x in g) and all(np.isfinite(r["loss"]) for r in loss)
    scenes = rollouts("train_scenes", run)
    scene_rate = (sum(r["success"] for r in scenes) / len(scenes)) if scenes else None
    ck = O / "models" / run / "last.pt"
    criteria = {
        "1_training_scene_ge_90pct": None if scene_rate is None else scene_rate >= TRAIN_SCENE_MIN,
        "2_no_nan_or_inf": bool(finite),
        "3_train_mae_le_threshold": train_mae <= MAE_THRESHOLD,
        "4a_grad_tail_median_le_bound": None if tail_med is None else tail_med <= TAIL_MEDIAN_MAX,
        "4b_spike_rate_le_bound": None if spike is None else spike <= SPIKE_RATE_MAX,
        "5_ema_not_worse_than_raw": None if ema_mae is None else ema_mae <= train_mae,
        "6_checkpoint_integrity": ck.exists(),
    }
    known = [v for v in criteria.values() if v is not None]
    return {"run": run, "passes_gate": bool(known) and all(known), "criteria": criteria,
            "train_mae": train_mae, "ema_mae": ema_mae, "train_scene": [sum(r["success"] for r in scenes), len(scenes)],
            "grad": {"median": med, "tail_median": tail_med, "max": max(g) if g else None,
                     "spike_rate": spike, "max_over_median": (max(g) / med) if g and med else None},
            "final_loss": loss[-1]["loss"], "steps": result["step"] + 1,
            "wall_min": result.get("elapsed_s", float("nan")) / 60, "vram_mb": result["peak_vram_bytes"] / 1e6}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--figures", default="docs/assets"); a = p.parse_args()
    runs = sorted(q.name for q in (O / "models").iterdir() if q.is_dir())
    out = {"gate_constants": {"train_mae_threshold": MAE_THRESHOLD, "grad_tail_median_max": TAIL_MEDIAN_MAX,
                              "spike_rate_max": SPIKE_RATE_MAX, "train_scene_min": TRAIN_SCENE_MIN},
           "runs": {}, "by_stage": {}}
    for run in runs:
        rec = evaluate_gate(run)
        if rec is None: out.setdefault("incomplete", []).append(run); continue
        stage, capacity, recipe, condition, seed = parse_name(run)
        held = rollouts("closed_loop", run)
        rec.update({"stage": stage, "capacity": capacity, "recipe": recipe, "condition": condition, "seed": seed,
                    "heldout": [sum(r["success"] for r in held), len(held)] if held else None,
                    "lateral_at_close_mm": (float(np.median([r["close"]["lateral_mm"] for r in held + rollouts("train_scenes", run) if r["close"]]))
                                            if (held or rollouts("train_scenes", run)) else None)})
        out["runs"][run] = rec
        out["by_stage"].setdefault(stage, []).append(run)
    # stage summaries ordered by capacity then recipe
    out["summary"] = {}
    for run, rec in out["runs"].items():
        key = f"{rec['capacity']}|{rec['recipe']}"
        s = out["summary"].setdefault(key, {"capacity": rec["capacity"], "recipe": rec["recipe"], "runs": [], "seeds": [],
                                            "train_scene": [0, 0], "heldout": [0, 0], "train_mae": [], "passes": []})
        s["runs"].append(run); s["seeds"].append(rec["seed"]); s["train_mae"].append(rec["train_mae"]); s["passes"].append(rec["passes_gate"])
        s["train_scene"][0] += rec["train_scene"][0]; s["train_scene"][1] += rec["train_scene"][1]
        if rec["heldout"]: s["heldout"][0] += rec["heldout"][0]; s["heldout"][1] += rec["heldout"][1]
    for s in out["summary"].values():
        s["train_scene_rate"] = s["train_scene"][0] / s["train_scene"][1] if s["train_scene"][1] else None
        s["heldout_rate"] = s["heldout"][0] / s["heldout"][1] if s["heldout"][1] else None
        s["heldout_wilson95"] = wilson(*s["heldout"]) if s["heldout"][1] else None
        s["gate_passed_all_seeds"] = all(s["passes"])
    O.mkdir(parents=True, exist_ok=True); (O / "summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    figures(out, Path(a.figures))
    brief = {k: {"train_scene": f"{v['train_scene'][0]}/{v['train_scene'][1]}", "train_mae": [round(x, 5) for x in v["train_mae"]],
                 "gate": v["gate_passed_all_seeds"], "heldout": f"{v['heldout'][0]}/{v['heldout'][1]}" if v["heldout"][1] else None}
             for k, v in sorted(out["summary"].items())}
    print(json.dumps({"gate_constants": out["gate_constants"], "summary": brief, "incomplete": out.get("incomplete", [])}, indent=1))


def figures(out, adir):
    runs = out["runs"]
    if not runs: return
    stages = sorted({r["stage"] for r in runs.values()})
    # training diagnostics: loss, gradient norm, learning rate
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for run, rec in sorted(runs.items()):
        loss, ema, grad = series(run)
        style = "-" if rec["recipe"] == "baseline" else "--"
        label = f"{rec['capacity']} {rec['recipe']}"
        axes[0].plot([r["step"] for r in loss], [r["sampled"]["action_mae"] for r in loss], style, color=COLORS[rec["capacity"]], lw=1.3, label=label)
        if grad:
            step = [r["step"] for r in grad]; gn = [r["grad_norm"] for r in grad]
            k = max(1, len(gn) // 200)
            axes[1].plot(step[::k], [float(np.median(gn[i:i + k])) for i in range(0, len(gn), k)][:len(step[::k])], style, color=COLORS[rec["capacity"]], lw=1.1, label=label)
            axes[2].plot(step[::k], [r["lr"] for r in grad][::k], style, color=COLORS[rec["capacity"]], lw=1.1, label=label)
    axes[0].set_xlabel("optimizer update"); axes[0].set_ylabel("sampled training MAE"); axes[0].set_yscale("log")
    axes[0].axhline(out["gate_constants"]["train_mae_threshold"], color="k", ls=":", lw=1, label="gate threshold")
    axes[1].set_xlabel("optimizer update"); axes[1].set_ylabel("gradient norm (binned median)"); axes[1].set_yscale("log")
    axes[1].axhline(out["gate_constants"]["grad_tail_median_max"], color="k", ls=":", lw=1, label="tail-median bound")
    axes[2].set_xlabel("optimizer update"); axes[2].set_ylabel("learning rate")
    for ax in axes: ax.grid(alpha=.3); ax.legend(fontsize=6)
    fig.suptitle("optimization study: training diagnostics", fontsize=10)
    fig.tight_layout(); fig.savefig(adir / "v2_optimization_diagnostics.png", dpi=125); plt.close(fig)

    # training-scene success and held-out by recipe
    summary = out["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    keys = sorted(summary, key=lambda k: (CAP_ORDER.index(summary[k]["capacity"]) if summary[k]["capacity"] in CAP_ORDER else 9, summary[k]["recipe"]))
    xs = np.arange(len(keys))
    axes[0].bar(xs, [100 * summary[k]["train_scene_rate"] for k in keys],
                color=[COLORS.get(summary[k]["capacity"], "0.5") for k in keys],
                edgecolor=["k" if summary[k]["gate_passed_all_seeds"] else "none" for k in keys], lw=1.5)
    axes[0].axhline(100 * out["gate_constants"]["train_scene_min"], color="k", ls=":", lw=1.2, label="gate: 90%")
    axes[0].set_xticks(xs, [k.replace("|", "\n") for k in keys], fontsize=6.5); axes[0].set_ylabel("training-scene success (%)")
    axes[0].set_title("training health (black edge = gate passed)", fontsize=9); axes[0].legend(fontsize=7); axes[0].grid(alpha=.3, axis="y")
    have = [k for k in keys if summary[k]["heldout_rate"] is not None]
    if have:
        axes[1].bar(np.arange(len(have)), [100 * summary[k]["heldout_rate"] for k in have],
                    color=[COLORS.get(summary[k]["capacity"], "0.5") for k in have])
        axes[1].set_xticks(np.arange(len(have)), [k.replace("|", "\n") for k in have], fontsize=6.5)
        axes[1].set_ylabel("held-out success (%)"); axes[1].set_title("held-out outcome (never used for selection)", fontsize=9)
        axes[1].grid(alpha=.3, axis="y")
    else:
        axes[1].text(.5, .5, "no held-out evaluation yet\n(Stage F only)", ha="center", va="center"); axes[1].set_axis_off()
    fig.tight_layout(); fig.savefig(adir / "v2_optimization_health.png", dpi=125); plt.close(fig)


if __name__ == "__main__":
    main()
