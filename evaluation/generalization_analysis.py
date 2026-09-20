"""Analysis for the physics-v2 spatial generalization study (docs/research/physics_v2_generalization_spec.md).

It reads the frozen split, the expert control, the offline evaluations and the closed-loop rollouts, and writes:
- summary.json;
- the figures;
- representative media.

Every statistic follows the pre-registered protocol: Wilson CIs, exact McNemar, a position-cluster bootstrap, a logistic slope with a
bootstrap CI, and Spearman ρ with a permutation p-value.
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

G = Path("artifacts/physics_v2_generalization")
MODELS = ("CLEAN10", "TRAIN20", "TRAIN40", "TRAIN80", "TRAIN80_80k")
NN_KEY = {"CLEAN10": "nn_clean10_mm", "TRAIN20": "nn_train20_mm", "TRAIN40": "nn_train40_mm", "TRAIN80": "nn_train80_mm", "TRAIN80_80k": "nn_train80_mm"}
BINS = [0, 5, 10, 15, np.inf]
CATS = ("A_interpolation", "B_sparse_interpolation", "C_extrapolation")
LAT_MAX, H_MIN, H_MAX, EARLY = 8.0, 2.0, 14.0, 20.0  # pre-registered taxonomy thresholds (mm)
rng = np.random.default_rng(0)


# ---------------------------------------------------------------- statistics
def wilson(k, n, z=1.959964):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0, c - h)), float(min(1, c + h))]


def rate(xs):
    xs = [bool(x) for x in xs]; k, n = sum(xs), len(xs)
    return {"k": k, "n": n, "rate": k / n if n else None, "wilson95": wilson(k, n)}


def mcnemar_exact(a, b):
    """Two-sided exact McNemar on paired booleans (a = model 1, b = model 2)."""
    n10 = sum(x and not y for x, y in zip(a, b)); n01 = sum(y and not x for x, y in zip(a, b)); n = n10 + n01
    if n == 0: return {"only_first": 0, "only_second": 0, "p_two_sided": 1.0}
    tail = sum(comb(n, i) for i in range(0, min(n10, n01) + 1)) / 2 ** n
    return {"only_first": n10, "only_second": n01, "p_two_sided": float(min(1.0, 2 * tail))}


def logistic_fit(x, y, ridge=1e-3):
    X = np.c_[np.ones(len(x)), x]; w = np.zeros(2)
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ w)); W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + ridge * np.eye(2); g = X.T @ (y - p) - ridge * w
        step = np.linalg.solve(H, g); w += step
        if np.abs(step).max() < 1e-8: break
    return w


def logistic_slope(dist_mm, success, B=2000):
    """Slope per 10 mm, with a position-bootstrap 95% CI. A small ridge keeps it finite under separation; this is reported."""
    x = np.asarray(dist_mm) / 10.0; y = np.asarray(success, float)
    if len(set(y)) < 2: return {"slope_per_10mm": None, "note": "all outcomes identical", "n": len(y)}
    w = logistic_fit(x, y); bs = []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x))
        if len(set(y[i])) == 2: bs.append(logistic_fit(x[i], y[i])[1])
    return {"slope_per_10mm": float(w[1]), "intercept": float(w[0]), "bootstrap95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": len(y), "ridge": 1e-3}


def spearman(x, y, P=10000):
    rk = lambda v: np.argsort(np.argsort(v)).astype(float)
    x, y = np.asarray(x, float), np.asarray(y, float); rx, ry = rk(x), rk(y)
    rho = float(np.corrcoef(rx, ry)[0, 1]); perm = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(P)])
    return {"rho": rho, "p_permutation_two_sided": float((np.abs(perm) >= abs(rho) - 1e-12).mean()), "n": len(x)}


def cluster_bootstrap(per_position, B=10000):
    """per_position: list of success lists (one per position). CI for pooled success, resampling positions."""
    arr = [np.asarray(v, float) for v in per_position]; k = sum(v.sum() for v in arr); n = sum(len(v) for v in arr); bs = []
    for _ in range(B):
        i = rng.integers(0, len(arr), len(arr)); bs.append(sum(arr[j].sum() for j in i) / sum(len(arr[j]) for j in i))
    return {"k": int(k), "n": int(n), "rate": k / n, "bootstrap95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "positions": len(arr)}


# ---------------------------------------------------------------- taxonomy
def tilt_deg(q):
    w, x, y, z = q[..., 3], q[..., 4], q[..., 5], q[..., 6]
    return np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))


def classify(r, rec):
    """Pre-registered failure taxonomy (first matching rule) + non-exclusive flags."""
    ex, cube, gc = rec["executed"], rec["cube"], rec["grasp_center"]; closed = np.where(ex[:, 5] < 0.1)[0]
    t_end = int(closed[0]) + 1 if len(closed) else len(cube) - 1  # state index after the first close command (or final)
    lat = 1000 * np.linalg.norm(gc[:, :2] - cube[:, :2], axis=1); h = 1000 * (gc[:, 2] - cube[:, 2])
    disp = 1000 * np.linalg.norm(cube[:t_end + 1, :2] - cube[0, :2], axis=1).max(); tilt = tilt_deg(cube[:t_end + 1]).max()
    sp = rec["side_pinch"].astype(bool); z = cube[:, 2]
    flags = {"vertical_pad_contact": bool(rec["vertical_pad_contact"].any()), "robot_table_contact_gt_0p1mm": (r.get("max_robot_table_penetration_mm") or 0) > 0.1,
             "cube_displaced_gt5mm_before_close": bool(disp > 5), "cube_tilt_gt20deg_before_close": bool(tilt > 20),
             "closed": bool(len(closed)), "ever_side_pinch": bool(sp.any()), "cube_lifted_30mm": bool(z.max() >= 0.03 + 0.01)}
    close = None if not len(closed) else {"t": int(closed[0]), "lateral_mm": float(lat[t_end]), "height_mm": float(h[t_end])}
    if r["success"]: cat = "success"
    elif r.get("invalid_reason"): cat = "invalid_physics"
    elif tilt > 20: cat = "cube_tilt"
    elif disp > 5: cat = "cube_nudge_displacement"
    elif not len(closed):
        cat = "never_closed_at_closable_pose" if ((lat <= LAT_MAX) & (h >= H_MIN) & (h <= H_MAX)).any() else "approach_positioning"
    elif close["height_mm"] > EARLY: cat = "early_close"
    elif close["lateral_mm"] > LAT_MAX: cat = "wrong_lateral_alignment"
    elif not H_MIN <= close["height_mm"] <= H_MAX: cat = "wrong_height"
    elif not sp.any(): cat = "failed_side_pinch"
    elif z.max() >= 0.04 and z[-1] < 0.03: cat = "drop_after_grasp"  # cube centre rest height 0.01 m: lifted >= 30 mm, fell below 20 mm
    elif sp.any(): cat = "lift_or_hold_failure"
    else: cat = "other"
    return cat, flags, close, {"min_lateral_mm": float(lat.min()), "max_disp_before_close_mm": float(disp), "max_tilt_before_close_deg": float(tilt)}


# ---------------------------------------------------------------- loading
def load_rollouts(model, positions):
    rows = []
    for f in sorted(glob.glob(str(G / "closed_loop" / model / "ep*_k*.json"))):
        r = json.loads(Path(f).read_text()); rec = np.load(f.replace(".json", ".npz"))
        pos = positions[r["episode"]]; cat, flags, close, geo = classify(r, rec)
        dev = r.get("executed_vs_expert_mae_first_n")
        rows.append({"model": model, "episode": r["episode"], "k": r["k"], "seed": r.get("policy_seed", 0), "success": bool(r["success"]), "steps": r["steps"],
                     "category": pos["category"], "nn_mm": pos[NN_KEY[model]], "nn_train80_mm": pos["nn_train80_mm"], "outcome": cat, "flags": flags, "close": close, **geo,
                     "trajectory_deviation_rad": float(np.mean(dev[:5])) if dev else None, "invalid_reason": r.get("invalid_reason"),
                     "any_vertical_pad_contact": r.get("any_vertical_pad_contact"), "max_cube_z": r.get("max_cube_z"),
                     **{k: r.get(k) for k in ("max_robot_table_penetration_mm", "max_cube_table_penetration_mm", "max_pad_cube_penetration_mm")}})
    return rows


def replay_rows():
    return [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(G / "closed_loop" / "TRAIN80" / "ep*_expert_replay.json")))]


# ---------------------------------------------------------------- figures
INK, MUTED = "#0b0b0b", "#52514e"


def fig_maps(positions, rows, split, out):
    from data.generalization_split import dataset_positions
    xy = {int(k): v for k, v in dataset_positions(split["dataset"]).items()}
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 8.2), sharey=True)
    W = 1000 * np.array([split["workspace_x_m"], split["workspace_y_m"]]); band = 1000 * np.array(split["rule"]["hole_band_y_m"])
    panels = (("nearest-TRAIN80 distance", None), ("TRAIN80, 80k steps, K=8", "TRAIN80_80k"), ("TRAIN80, 20k steps, K=8", "TRAIN80"), ("CLEAN10, K=8", "CLEAN10"))
    for ax, (title, model) in zip(axes, panels):
        ax.add_patch(plt.Rectangle((W[0, 0], W[1, 0]), np.ptp(W[0]), np.ptp(W[1]), fill=False, ls="--", color="0.6")); ax.axhspan(*band, color="C1", alpha=.08)
        train = split["train_subsets"]["CLEAN10" if model == "CLEAN10" else "TRAIN80"]  # 80k and 20k share TRAIN80
        t = 1000 * np.array([xy[i] for i in train]); ax.scatter(*t.T, s=10 if model != "CLEAN10" else 40, color="0.55", marker="o", label=f"{'CLEAN10' if model == 'CLEAN10' else 'TRAIN80'} training positions")
        if model is None:
            p = [positions[i] for i in sorted(positions)]; sc = ax.scatter([1000 * q["cube_xy"][0] for q in p], [1000 * q["cube_xy"][1] for q in p], c=[q["nn_train80_mm"] for q in p], cmap="viridis", s=55, edgecolor="k", lw=.4)
            fig.colorbar(sc, ax=ax, fraction=.05, label="nearest TRAIN80 (mm)")
        else:
            sel = {r["episode"]: r for r in rows if r["model"] == model and r["k"] == 8 and r["seed"] == 0}
            for ep, r in sel.items():
                q = positions[ep]; ax.scatter(1000 * q["cube_xy"][0], 1000 * q["cube_xy"][1], s=60, marker="o" if r["success"] else "X", color="#2a9d4b" if r["success"] else "#d1352b", edgecolor="k", lw=.4)
            ax.scatter([], [], marker="o", color="#2a9d4b", label="success"); ax.scatter([], [], marker="X", color="#d1352b", label="failure")
            for ep in sorted(set(positions) - set(sel)): q = positions[ep]; ax.scatter(1000 * q["cube_xy"][0], 1000 * q["cube_xy"][1], s=40, marker="s", facecolor="none", edgecolor="0.4")
        ax.set_title(title, fontsize=10); ax.set_aspect("equal"); ax.grid(alpha=.25); ax.set_xlabel("cube x (mm)"); ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("cube y (mm)"); fig.tight_layout(); fig.savefig(out / "v2_gen_success_map.png", dpi=120); plt.close(fig)


def fig_distance(rows, offline, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]; centers = [2.5, 7.5, 12.5, 17.5]
    for model, col in (("TRAIN80", "C0"), ("CLEAN10", "C3")):
        sel = [r for r in rows if r["model"] == model and r["k"] == 8 and r["seed"] == 0]
        if not sel: continue
        rs = [rate([r["success"] for r in sel if lo <= r["nn_mm"] < hi]) for lo, hi in zip(BINS[:-1], BINS[1:])]
        c = [x for x, r in zip(centers, rs) if r["n"]]; v = [r for r in rs if r["n"]]
        ax.errorbar(c, [100 * r["rate"] for r in v], yerr=np.array([[100 * (r["rate"] - r["wilson95"][0]), 100 * (r["wilson95"][1] - r["rate"])] for r in v]).T, fmt="o-", color=col, capsize=3, label=f"{model} (own nearest-training distance)")
        for x, r in zip(c, v): ax.annotate(f"{r['k']}/{r['n']}", (x, 100 * r["rate"]), textcoords="offset points", xytext=(6, 4), fontsize=7, color=col)
    ax.set_xticks(centers, ["<5", "5–10", "10–15", "≥15"]); ax.set_xlabel("nearest training position (mm)"); ax.set_ylabel("success at K=8 (%)"); ax.set_ylim(-5, 108); ax.grid(alpha=.3); ax.legend(fontsize=7)
    ax = axes[1]
    for model, col in (("TRAIN80", "C0"), ("CLEAN10", "C3")):
        o = offline.get(model, {}).get("test_per_episode")
        if o: ax.scatter([p[0] for p in o], [p[1] for p in o], s=16, color=col, label=model, alpha=.8)
    ax.set_xlabel("nearest training position (mm)"); ax.set_ylabel("offline action MAE per test episode"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "v2_gen_distance.png", dpi=120); plt.close(fig)


def fig_scaling(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]; ks = [1, 2, 4, 8, 16]
    for model, col in (("TRAIN80_80k", "C2"), ("TRAIN80", "C0"), ("CLEAN10", "C3")):
        v = [summary["closed_loop"].get(model, {}).get("by_k_seed0", {}).get(str(k)) for k in ks]
        pts = [(k, r) for k, r in zip(ks, v) if r and r["n"]]
        if pts: ax.errorbar([str(k) for k, _ in pts], [100 * r["rate"] for _, r in pts], yerr=np.array([[100 * (r["rate"] - r["wilson95"][0]), 100 * (r["wilson95"][1] - r["rate"])] for _, r in pts]).T, fmt="o-", capsize=3, color=col, label={"TRAIN80_80k": "TRAIN80, 80k steps", "TRAIN80": "TRAIN80, 20k steps", "CLEAN10": "CLEAN10, 20k steps"}[model])
    ax.set_xlabel("K (actions executed per replan)"); ax.set_ylabel("held-out success (%)"); ax.set_ylim(-5, 105); ax.grid(alpha=.3); ax.legend(fontsize=8); ax.set_title("held-out K sweep (seed 0)", fontsize=10)
    ax = axes[1]; sizes = [("CLEAN10", 10), ("TRAIN20", 20), ("TRAIN40", 40), ("TRAIN80", 80)]
    pts = [(n, summary["closed_loop"].get(m, {}).get("by_k_seed0", {}).get("8")) for m, n in sizes]; pts = [(n, r) for n, r in pts if r and r["n"]]
    if pts:
        ax.errorbar([n for n, _ in pts], [100 * r["rate"] for _, r in pts], yerr=np.array([[100 * (r["rate"] - r["wilson95"][0]), 100 * (r["wilson95"][1] - r["rate"])] for _, r in pts]).T, fmt="o-", capsize=3, color="C0", label="closed loop K=8 (all test)")
        for n, r in pts: ax.annotate(f"{r['k']}/{r['n']}", (n, 100 * r["rate"]), textcoords="offset points", xytext=(5, -12), fontsize=7)
    r80 = summary["closed_loop"].get("TRAIN80_80k", {}).get("by_k_seed0", {}).get("8")
    if r80 and r80["n"]: ax.scatter([80], [100 * r80["rate"]], marker="*", s=120, color="C2", zorder=5, label=f"TRAIN80, 80k steps ({r80['k']}/{r80['n']})")
    ax.set_xscale("log", base=2); ax.set_xticks([10, 20, 40, 80], ["10", "20", "40", "80"]); ax.set_xlabel("training demonstrations (nested)"); ax.set_ylabel("held-out success (%)")
    ax.set_ylim(-5, 105); ax.grid(alpha=.3); ax.legend(fontsize=7); ax.set_title("data scaling, same 2M model, 20k steps", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "v2_gen_scaling.png", dpi=120); plt.close(fig)


def render_rollout(model, ep, test_root, out_path, label):
    """Re-simulate the executed actions (deterministic) with a task camera and a side camera, so the grasp type is visible."""
    import imageio.v2 as imageio, mujoco
    from PIL import Image, ImageDraw
    from simulation.env import PickCubeConfig, SO101PickCubeEnv
    rec = np.load(G / "closed_loop" / model / f"ep{ep:03d}_k8.npz"); meta = json.loads((Path(test_root) / "episodes" / f"episode_{ep:06d}" / "episode.json").read_text())
    env = SO101PickCubeEnv(PickCubeConfig(render_observations=False, workspace_x=(0.22, 0.295), workspace_y=(-0.085, 0.085))); env.reset(seed=meta["seed"], options={"cube_xy": meta["cube_xy"]})
    renderer = mujoco.Renderer(env.model, height=300, width=400); side = mujoco.MjvCamera(); side.type = mujoco.mjtCamera.mjCAMERA_FREE
    def shot(txt):
        renderer.update_scene(env.data, camera="external_rgb"); task = renderer.render().copy()
        c = env.cube_pose[:3]; side.lookat[:] = [c[0], c[1], max(0.03, c[2])]; side.distance = 0.20; side.azimuth = 90.0; side.elevation = -6.0
        renderer.update_scene(env.data, camera=side); img = Image.fromarray(np.concatenate((task, renderer.render()), axis=1)); ImageDraw.Draw(img).text((8, 8), txt, fill=(255, 255, 255)); return np.asarray(img)
    frames = [shot(f"{label} | t=0")]; info = {}
    for t, a in enumerate(rec["executed"]):
        _, _, term, _, info = env.step(a); c = info["contacts"]
        frames.append(shot(f"{label} | t={(t + 1) / 20:.2f}s | {'SIDE PINCH' if c['side_pinch'] else ('closing' if a[5] < 0.1 else 'approach')} | cube z {1000 * env.cube_pose[2]:.0f} mm"))
        if term: break
    frames += [frames[-1]] * 12
    imageio.mimsave(out_path, [np.asarray(Image.fromarray(f).resize((640, 240), Image.LANCZOS)) for f in frames], duration=70, loop=0)
    idx = np.linspace(0, len(frames) - 13, 5).astype(int); fig, axes = plt.subplots(1, 5, figsize=(15, 2.6))
    for ax, i in zip(axes, idx): ax.imshow(frames[i][:, 400:]); ax.set_axis_off(); ax.set_title(f"t = {i / 20:.2f} s", fontsize=9)
    fig.tight_layout(); fig.savefig(str(out_path).replace(".gif", "_strip.png"), dpi=100); plt.close(fig)
    return {"replayed_success": bool(info.get("success")), "invalid_reason": info.get("invalid_reason"), "side_pinch_at_end": bool(info.get("contacts", {}).get("side_pinch"))}


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(); p.add_argument("--split", default="docs/research/physics_v2_generalization_split.json")
    p.add_argument("--test", default="artifacts/pickcube_physics_v2_gen_test_rgb160"); p.add_argument("--figures", default="docs/assets"); p.add_argument("--media", action="store_true")
    a = p.parse_args(); split = json.loads(Path(a.split).read_text()); out = Path(a.figures)
    control = json.loads((G / "expert_control.json").read_text()); in_bench = {r["index"] for r in control["test_rows"] if r["in_benchmark"]}
    positions = {r["index"]: r for r in split["test"] if r["index"] in in_bench}
    rows = [r for m in MODELS for r in load_rollouts(m, positions) if (G / "closed_loop" / m).exists()]
    summary = {"test_positions_in_benchmark": len(positions), "outside_benchmark": sorted({r["index"] for r in split["test"]} - in_bench),
               "expert_replay": rate([r["success"] for r in replay_rows()]), "closed_loop": {}, "offline": {}}
    for m in MODELS:
        mr = [r for r in rows if r["model"] == m]
        if not mr: continue
        s0 = [r for r in mr if r["seed"] == 0]; k8 = [r for r in s0 if r["k"] == 8]
        d = {"by_k_seed0": {str(k): rate([r["success"] for r in s0 if r["k"] == k]) for k in sorted({r["k"] for r in s0})},
             "k8_seed0_by_category": {c: rate([r["success"] for r in k8 if r["category"] == c]) for c in CATS},
             "by_k_seed0_by_category": {str(k): {c: rate([r["success"] for r in s0 if r["k"] == k and r["category"] == c]) for c in CATS} for k in sorted({r["k"] for r in s0})},
             "k8_binned_by_own_nn": {f"{lo}-{hi}": rate([r["success"] for r in k8 if lo <= r["nn_mm"] < hi]) for lo, hi in zip(BINS[:-1], BINS[1:])},
             "k8_binned_by_nn_train80": {f"{lo}-{hi}": rate([r["success"] for r in k8 if lo <= r["nn_train80_mm"] < hi]) for lo, hi in zip(BINS[:-1], BINS[1:])},
             "k8_logistic_vs_own_nn": logistic_slope([r["nn_mm"] for r in k8], [r["success"] for r in k8]) if k8 else None,
             "outcomes_seed0_by_k": {str(k): {o: sum(r["outcome"] == o for r in s0 if r["k"] == k) for o in sorted({r["outcome"] for r in s0 if r["k"] == k})} for k in sorted({r["k"] for r in s0})},
             "outcomes_k8_seed0_by_category": {c: {o: sum(r["outcome"] == o for r in k8 if r["category"] == c) for o in sorted({r["outcome"] for r in k8 if r["category"] == c})} for c in CATS},
             "flags_failures_seed0": {f: sum(r["flags"][f] for r in s0 if not r["success"]) for f in (s0[0]["flags"] if s0 else {})},
             "success_steps_median": float(np.median([r["steps"] for r in mr if r["success"]])) if any(r["success"] for r in mr) else None,
             "trajectory_deviation_rad_k8_seed0": {"success": float(np.median([r["trajectory_deviation_rad"] for r in k8 if r["success"]])) if any(r["success"] for r in k8) else None,
                                                   "failure": float(np.median([r["trajectory_deviation_rad"] for r in k8 if not r["success"]])) if any(not r["success"] for r in k8) else None},
             "validity": {"successes": sum(r["success"] for r in mr), "successes_with_invalid_reason": sum(r["success"] and r["invalid_reason"] is not None for r in mr),
                          "successes_with_any_vertical_pad_contact": sum(r["success"] and bool(r["any_vertical_pad_contact"]) for r in mr),
                          "invalid_physics_rollouts": sum(r["invalid_reason"] is not None for r in mr),
                          "failures_that_lifted_cube_30mm": sum((not r["success"]) and r["flags"]["cube_lifted_30mm"] for r in mr),
                          "max_robot_table_penetration_mm": max((r["max_robot_table_penetration_mm"] or 0) for r in mr),
                          "max_cube_table_penetration_mm": max((r["max_cube_table_penetration_mm"] or 0) for r in mr),
                          "max_pad_cube_penetration_mm": max((r["max_pad_cube_penetration_mm"] or 0) for r in mr)}}
        seeds = sorted({r["seed"] for r in mr if r["k"] == 8})
        if len(seeds) > 1:
            per_pos = [[r["success"] for r in mr if r["k"] == 8 and r["episode"] == ep] for ep in sorted(positions)]
            d["k8_pooled_seeds"] = {"seeds": seeds, **cluster_bootstrap([v for v in per_pos if len(v) == len(seeds)])}
            d["k8_by_seed"] = {str(s): rate([r["success"] for r in mr if r["k"] == 8 and r["seed"] == s]) for s in seeds}
            d["k8_positions_by_successes"] = {str(j): sum(sum(v) == j for v in per_pos if len(v) == len(seeds)) for j in range(len(seeds) + 1)}
        summary["closed_loop"][m] = d
    t80 = {r["episode"]: r["success"] for r in rows if r["model"] == "TRAIN80" and r["k"] == 8 and r["seed"] == 0}
    c10 = {r["episode"]: r["success"] for r in rows if r["model"] == "CLEAN10" and r["k"] == 8 and r["seed"] == 0}
    common = sorted(set(t80) & set(c10))
    if common:
        summary["paired_TRAIN80_vs_CLEAN10_k8_seed0"] = {"n": len(common), **mcnemar_exact([t80[e] for e in common], [c10[e] for e in common]),
            "by_category": {c: {"n": len(ce := [e for e in common if positions[e]["category"] == c]), **mcnemar_exact([t80[e] for e in ce], [c10[e] for e in ce])} for c in CATS}}
    for m in MODELS:
        o = {}
        for part in ("train", "val", "test"):
            f = G / "offline" / f"{m}_{part}.json"
            if f.exists():
                r = json.loads(f.read_text())
                o[part] = {"windows": r["windows"], "episodes": len(r["episodes"]), "action_mae": r["all"]["action_mae"], "arm_mae_rad": r["all"]["arm_mae_rad"], "gripper_mae": r["all"]["gripper_mae"],
                           "per_joint_mae": r["all"]["per_joint_mae"], "episode_start_mae": r["episode_start"]["action_mae"],
                           "quarters_mae": {q: v["action_mae"] for q, v in r["quarters"].items()}, "expert_phase_mae": {q: v["action_mae"] for q, v in r["expert_phase"].items()}}
                if part == "test":
                    pe = [(positions[int(e)][NN_KEY[m]], v["action_mae"], positions[int(e)]["category"]) for e, v in r["per_episode"].items() if int(e) in positions]
                    o["test_per_episode"] = pe; o["test_spearman_mae_vs_own_nn"] = spearman([x[0] for x in pe], [x[1] for x in pe])
                    o["test_by_category_mae"] = {c: float(np.mean([x[1] for x in pe if x[2] == c])) if any(x[2] == c for x in pe) else None for c in CATS}
                    o["test_start_mae_by_category"] = {c: float(np.mean([v["start_mae"] for e, v in r["per_episode"].items() if int(e) in positions and positions[int(e)]["category"] == c])) for c in CATS if any(positions[int(e)]["category"] == c for e in r["per_episode"] if int(e) in positions)}
        curve = sorted(glob.glob(str(G / "offline" / f"{m}_val_ema_step*.json")))
        if curve: o["val_curve"] = [(json.loads(Path(f).read_text())["step"], json.loads(Path(f).read_text())["all"]["action_mae"]) for f in curve]
        if o: summary["offline"][m] = o
    summary["rows"] = rows
    (G / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    fig_maps(positions, rows, split, out); fig_distance(rows, summary["offline"], out); fig_scaling(summary, out)
    if a.media:
        k8 = [r for r in rows if r["model"] == "TRAIN80" and r["k"] == 8 and r["seed"] == 0]
        media = {}
        ok = [r for r in k8 if r["success"]]; bad = [r for r in k8 if not r["success"]]
        if ok:
            r = max(ok, key=lambda r: (r["nn_train80_mm"], -r["episode"])); media["success"] = {"episode": r["episode"], "category": r["category"], "nn_train80_mm": r["nn_train80_mm"],
                **render_rollout("TRAIN80", r["episode"], a.test, out / "v2_gen_heldout_success.gif", f"TRAIN80 held-out {r['category'][0]} ep{r['episode']} nn {r['nn_train80_mm']:.1f}mm")}
        if bad:
            r = min(bad, key=lambda r: (r["nn_train80_mm"], r["episode"])); media["failure"] = {"episode": r["episode"], "category": r["category"], "nn_train80_mm": r["nn_train80_mm"], "outcome": r["outcome"],
                **render_rollout("TRAIN80", r["episode"], a.test, out / "v2_gen_heldout_failure.gif", f"TRAIN80 held-out {r['category'][0]} ep{r['episode']} nn {r['nn_train80_mm']:.1f}mm FAIL")}
        summary["media"] = media; (G / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rows",)}, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
