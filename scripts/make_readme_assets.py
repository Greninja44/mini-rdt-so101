"""Regenerate the README figures in docs/assets/ from experiment artifacts.

Usage: python scripts/make_readme_assets.py --artifacts artifacts
Hi-res rollout media are re-rendered by replaying a recorded closed-loop rollout's executed
actions (the simulator is deterministic, so the trajectory is identical).
"""
from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import mujoco
import numpy as np
from PIL import Image, ImageDraw

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def style(ax):
    ax.grid(color=GRID, lw=.6, axis="y"); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)


def rollout_media(art: Path, out: Path):
    """Physics-v2 TinyRDT success (CLEAN10 scene seed 3006, K=8): task camera + side view at table height."""
    from simulation.env import PickCubeConfig, SO101PickCubeEnv
    rec = np.load(art / "physics_v2/closed_loop/ep006_k8.npz")
    env = SO101PickCubeEnv(PickCubeConfig(render_observations=False, physics="v2")); env.reset(seed=3006)
    renderer = mujoco.Renderer(env.model, height=300, width=400); side = mujoco.MjvCamera(); side.type = mujoco.mjtCamera.mjCAMERA_FREE
    def shot():
        renderer.update_scene(env.data, camera="external_rgb"); task = renderer.render().copy()
        c = env.cube_pose[:3]; side.lookat[:] = [c[0], c[1], max(0.03, c[2])]; side.distance = 0.20; side.azimuth = 90.0; side.elevation = -6.0
        renderer.update_scene(env.data, camera=side); return np.concatenate((task, renderer.render()), axis=1)
    frames = [shot()]; success = False; labels = ["start"]
    for a in rec["executed"]:
        _, _, terminated, _, info = env.step(a); frames.append(shot())
        labels.append("side pinch" if info["contacts"]["side_pinch"] else ("closing" if a[5] < 0.1 else "approach"))
        if terminated: success = bool(info["success"]); break
    assert success and info["invalid_reason"] is None, "replayed rollout must reproduce the recorded physics-v2 success"
    frames += [frames[-1]] * 12
    imageio.mimsave(out / "rollout_success.gif", [np.asarray(Image.fromarray(f).resize((640, 240), Image.LANCZOS)) for f in frames], duration=70, loop=0)
    first_pinch = labels.index("side pinch"); n = len(labels) - 1
    keys = [(0, "start"), (18, "approach"), (first_pinch - 6, "aligned above cube"), (first_pinch + 1, "side pinch"), (n, "lifted (success)")]
    fig, axes = plt.subplots(1, 5, figsize=(15, 2.6))
    for ax, (i, label) in zip(axes, keys):
        ax.imshow(frames[i][:, 400:]); ax.set_axis_off(); ax.set_title(f"t = {i / 20:.2f} s · {label}", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(out / "rollout_strip.png", dpi=110); plt.close(fig)


def k_sweep_v2(art: Path, out: Path):
    rows = [json.load(open(f)) for f in glob.glob(str(art / "physics_v2/closed_loop/ep*_k*.json"))]
    ks = [1, 2, 4, 8, 16]; rate = [100 * np.mean([r["success"] for r in rows if r["k"] == k]) for k in ks]
    replay = [json.load(open(f)) for f in glob.glob(str(art / "physics_v2/closed_loop/ep*_expert_replay.json"))]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(ks, rate, marker="o", lw=2, ms=7, color=BLUE, label="TinyRDT 2.0M, 10 demos (physics-v2)")
    ax.axhline(100 * np.mean([r["success"] for r in replay]), color=MUTED, lw=1.2, ls="--", label="expert replay")
    for k, v in zip(ks, rate): ax.text(k, v - 9, f"{round(v / 10)}/10", ha="center", fontsize=9, color=INK)
    ax.set_xscale("log", base=2); ax.set_xticks(ks, [str(k) for k in ks]); ax.set_ylim(0, 108)
    ax.set_xlabel("K = actions executed per replan (chunk H = 16)"); ax.set_ylabel("valid side-grasp success (%)")
    ax.set_title("Physics-v2 closed loop on the 10 memorised cubes", fontsize=10, color=INK); style(ax); ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "k_sweep_v2.png", dpi=120); plt.close(fig)


def physics_fix(art: Path, out: Path):
    audit = json.load(open(art.parent / "docs/audit/table_collision/audit.json")) if (art.parent / "docs/audit/table_collision/audit.json").exists() else None
    v2 = json.load(open(art / "physics_v2/closed_loop_summary.json")); val = json.load(open(art / "physics_v2/expert_validation/validation.json"))
    v1_pen = max(x["max_penetration_mm"] for x in audit["tinyrdt_rollouts"]) if audit else 100.3
    v1_pad = max(x["pinch"]["pad_penetration_at_first_pinch_mm"] for x in audit["expert_demos"]) if audit else 9.9
    v1_cube = max(x["pinch"]["cube_pushed_into_table_max_mm"] for x in audit["expert_demos"]) if audit else 5.6
    rows = [("robot through table\n(max, mm)", v1_pen, max(v2["max_robot_table_penetration_mm"], val["random100"]["max_robot_table_penetration_mm"])),
            ("pad inside table\nat pinch (mm)", v1_pad, 0.0),
            ("cube pushed into\ntable (mm)", v1_cube, max(v2["max_cube_table_penetration_mm"], val["random100"]["max_cube_table_penetration_mm"]))]
    fig, ax = plt.subplots(figsize=(7.5, 3.3)); x = np.arange(len(rows)); w = .36
    b1 = ax.bar(x - w / 2, [r[1] for r in rows], w, color=ORANGE, label="physics-v1 (INVALID)"); b2 = ax.bar(x + w / 2, [r[2] for r in rows], w, color=AQUA, label="physics-v2")
    for b in list(b1) + list(b2): ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.15 + 0.02, f"{b.get_height():.2f}" if b.get_height() < 1 else f"{b.get_height():.1f}", ha="center", fontsize=9, color=INK)
    ax.set_yscale("symlog", linthresh=1); ax.set_xticks(x, [r[0] for r in rows], fontsize=9); ax.set_ylabel("mm (symlog)")
    ax.set_title("Collision audit fix: v1 grasps went through the table; v2 grasps do not", fontsize=10, color=INK); style(ax); ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(out / "physics_fix.png", dpi=120); plt.close(fig)


def pipeline(out: Path):
    fig, ax = plt.subplots(figsize=(12, 2.6)); ax.set_xlim(0, 11.3); ax.set_ylim(-0.45, 2.15); ax.set_axis_off()
    boxes = [(0.1, "RGB 160×120\n+ joint state (6)", MUTED), (2.35, "MobileNetV3-S\n(frozen / fine-tuned)", BLUE),
             (4.6, "TinyRDT\n4× Transformer, d=192\n~2.0M params", VIOLET), (6.85, "Diffusion head\ncosine, x0-pred\nDDIM-10", AQUA),
             (9.1, "H=16 action chunk\nexecute K, replan\n→ SO-101 (MuJoCo)", ORANGE)]
    for x, text, color in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.55), 2.0, 1.5, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white", ec=color, lw=2))
        ax.text(x + 1.0, 1.3, text, ha="center", va="center", fontsize=9.5, color=INK)
    for x in (2.1, 4.35, 6.6, 8.85):
        ax.annotate("", xy=(x + 0.25, 1.3), xytext=(x, 1.3), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.5))
    ax.annotate("", xy=(1.1, 0.5), xytext=(10.1, 0.5), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2, connectionstyle="arc3,rad=-0.12"))
    ax.text(5.6, 0.12, "receding horizon: observe → predict 16 → execute K → observe", ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(); fig.savefig(out / "pipeline.png", dpi=120); plt.close(fig)


def diffusion_fix(art: Path, out: Path):
    ra = art / "research_audit"
    rows = [("linear β, ε-pred\n(original)", json.load(open(ra / "linear_epsilon/result.json"))["sampled"]["action_mae"], MUTED),
            ("cosine, ε-pred", json.load(open(ra / "cosine_epsilon/result.json"))["sampled"]["action_mae"], ORANGE),
            ("cosine, x0-pred", json.load(open(ra / "cosine_x0/result.json"))["sampled"]["action_mae"], YELLOW),
            ("+ hold padding", json.load(open(ra / "cosine_x0_hold/result.json"))["sampled"]["action_mae"], AQUA),
            ("+ 20k steps, EMA", json.load(open(ra / "cosine_x0_hold_ema/diagnostics_ema_last/diagnostics.json"))["full_training_windows"][0]["all"]["action_mae"], BLUE)]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    bars = ax.bar(range(len(rows)), [r[1] for r in rows], color=[r[2] for r in rows], width=.6)
    for b, r in zip(bars, rows): ax.text(b.get_x() + b.get_width() / 2, r[1] * 1.15, f"{r[1]:.3g}", ha="center", fontsize=9, color=INK)
    ax.set_yscale("log"); ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=9); ax.set_ylabel("sampled action MAE (rad, log)")
    ax.set_title("10-demo overfit: fixing the diffusion sampler (DDIM-10, same 2M model)", fontsize=10, color=INK); style(ax)
    fig.tight_layout(); fig.savefig(out / "diffusion_fix.png", dpi=120); plt.close(fig)


def k_sweep(art: Path, out: Path):
    def rate(d, k):
        rows = [json.load(open(f)) for f in glob.glob(str(art / d / f"ep*_k{k}.json"))]
        return sum(r["success"] for r in rows) / len(rows) if rows else None
    series = [("TinyRDT (10 demos)", ["closed_loop_v2/tinyrdt_ema"] * 4 + ["closed_loop_v3/K16_A"], BLUE, "o", 0.97),
              ("+ spatial tokens", ["closed_loop_v3/S"] * 4 + ["closed_loop_v3/K16_S"], AQUA, "s", 0.99),
              ("+ DAgger data", ["closed_loop_v3/C40k"] * 4 + ["closed_loop_v3/K16_C40k"], YELLOW, "D", 1.01),
              ("80 demos, held-out cubes", ["closed_loop_v3/D80_validation"] * 4 + ["closed_loop_v3/K16_D80_validation"], ORANGE, "^", 1.03)]
    ks = [1, 2, 4, 8, 16]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for label, dirs, color, marker, dx in series:  # small x offsets keep coincident points visible
        ax.plot([k * dx for k in ks], [100 * rate(d, k) for d, k in zip(dirs, ks)], marker=marker, lw=2, ms=6, color=color, label=label,
                markeredgecolor="white", markeredgewidth=1)
    ax.set_xscale("log", base=2); ax.set_xticks(ks, [str(k) for k in ks]); ax.set_ylim(0, 105)
    ax.set_xlabel("K = actions executed per replan (chunk H = 16)"); ax.set_ylabel("PickCube success (%)")
    ax.set_title("Closed-loop success vs replanning interval (n = 10 cubes per point)", fontsize=10, color=INK); style(ax); ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(out / "k_sweep.png", dpi=120); plt.close(fig)


def grasp_tolerance(art: Path, out: Path):
    rows = json.load(open(art / "closed_loop_v2/grasp_sensitivity.json"))["rows"]
    joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]; offs = [-.08, -.05, -.03, -.02, -.01, .01, .02, .03, .05, .08]
    grid = np.array([[sum(r["success"] for r in rows if r["joint"] == j and abs(r["delta"] - o) < 1e-9) for o in offs] for j in joints])
    fig, ax = plt.subplots(figsize=(8, 2.8))
    im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=10, aspect="auto")
    for i in range(len(joints)):
        for j in range(len(offs)): ax.text(j, i, grid[i, j], ha="center", va="center", fontsize=9, color="white" if grid[i, j] > 6 else INK)
    ax.set_xticks(range(len(offs)), [f"{o:+.2f}" for o in offs]); ax.set_yticks(range(len(joints)), joints)
    ax.set_xlabel("constant joint offset from descent onward (rad)"); ax.set_title("Grasp tolerance: expert successes out of 10 seeds", fontsize=10, color=INK)
    fig.colorbar(im, ax=ax, fraction=.025); fig.tight_layout(); fig.savefig(out / "grasp_tolerance.png", dpi=120); plt.close(fig)


def isolation(art: Path, out: Path):
    def total(d):
        rows = [json.load(open(f)) for f in glob.glob(str(art / d / "ep*_k*.json"))]; return sum(r["success"] for r in rows), len(rows)
    def prefix(d):
        rows = [json.load(open(f)) for f in glob.glob(str(art / d / "ep*.json"))]; return sum(r["success"] for r in rows), len(rows)
    items = [("TinyRDT alone", total("closed_loop_v2/tinyrdt_ema"), BLUE), ("policy arm +\nexpert gripper rule", total("phase6/oracle/gripper_oracle"), MUTED),
             ("expert arm +\npolicy gripper", total("phase6/oracle/arm_oracle"), AQUA), ("expert drives first\n9 steps, then policy", prefix("recovery_prefix/A"), ORANGE)]
    fig, ax = plt.subplots(figsize=(7.5, 3.3))
    vals = [100 * s / n for _, (s, n), _ in items]
    bars = ax.bar(range(len(items)), vals, color=[c for *_, c in items], width=.55)
    for b, (_, (s, n), _) in zip(bars, items): ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{s}/{n}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(items)), [i[0] for i in items], fontsize=9); ax.set_ylim(0, 112); ax.set_ylabel("success (%)")
    ax.set_title("Failure isolation: the gripper is fine; early arm placement is not", fontsize=10, color=INK); style(ax)
    fig.tight_layout(); fig.savefig(out / "failure_isolation.png", dpi=120); plt.close(fig)


def failure_gif(art: Path, out: Path):
    frames = imageio.mimread(art / "phase5_analysis/gifs/A_ep007_k1.gif")[:60:2]
    imageio.mimsave(out / "failure_annotated.gif", [np.asarray(Image.fromarray(f).convert("RGB").resize((630, 330), Image.LANCZOS)) for f in frames], duration=160, loop=0)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--artifacts", default="artifacts"); p.add_argument("--output", default="docs/assets")
    a = p.parse_args(); art, out = Path(a.artifacts), Path(a.output); out.mkdir(parents=True, exist_ok=True)
    # v1 manipulation figures (k_sweep, isolation, grasp_tolerance, failure_gif) are kept only as invalid history in
    # docs/assets/physics_v1_invalid/ and are no longer regenerated.
    for fn in (pipeline, lambda o: diffusion_fix(art, o), lambda o: k_sweep_v2(art, o), lambda o: physics_fix(art, o), lambda o: rollout_media(art, o)):
        fn(out)
    for f in sorted(out.iterdir()): print(f"{f.stat().st_size / 1e3:8.0f} kB  {f}")


if __name__ == "__main__":
    main()
