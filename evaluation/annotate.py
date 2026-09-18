"""Annotated GIF of a closed-loop rollout or a DAgger collection episode.

Left: camera observation (3x). Right: top-down grasp-center path vs the
nominal expert path and the cube. Text: step, phase, deviation, gripper, and
event markers (divergence onset 0.03 rad time-free, takeover, close, pinch,
lift, outcome).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

INK, MUTED, POLICY, EXPERT, CUBE, NOMINAL = (11, 11, 11), (82, 81, 78), (42, 120, 214), (27, 175, 122), (227, 73, 72), (160, 160, 155)


def load(path: Path):
    if path.is_dir():  # DAgger / perturbation episode
        g = lambda n: np.load(path / f"{n}.npy")
        return {"rgb": g("rgb"), "q": g("joint_pos"), "cube": g("cube_pose"), "gc": g("grasp_center"), "cmd": g("executed_action")[:, 5],
                "controller": g("controller"), "grasped": None, "t": g("t")}
    r = np.load(path)
    T = len(r["executed"])
    return {"rgb": r["rgb"][:T], "q": r["state"][:T, :5], "cube": r["cube"][:T], "gc": r["grasp_center"][:T], "cmd": r["executed"][:, 5],
            "controller": np.array(["policy"] * T), "grasped": r["grasped"][:T], "t": np.arange(T)}


def phase(cmd, gc, cube):
    xy = np.linalg.norm(gc[:2] - cube[:2])
    return ("approach" if xy > .02 else "descent") if cmd >= .5 else ("grasp" if cube[2] < .03 else "lift")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path"); p.add_argument("--episode", type=int, required=True); p.add_argument("--output", required=True); p.add_argument("--title", default="")
    a = p.parse_args()
    d = load(Path(a.path)); ref = np.load(f"artifacts/closed_loop_v2/tinyrdt_ema/ep{a.episode:03d}_expert_replay.npz")
    nq, ngc = ref["state"][:, :5], ref["grasp_center"]
    dist = np.array([np.linalg.norm(nq - q, axis=1).min() for q in d["q"]])
    path_mm = 1000 * np.array([np.linalg.norm(ngc - g, axis=1).min() for g in d["gc"]])
    events = {}
    above = dist > .03
    for i in range(len(above) - 2):
        if above[i:i + 3].all(): events.setdefault(i, []).append("DIVERGES >0.03 rad"); break
    tk = np.where(d["controller"] == "expert")[0]
    if len(tk) and (d["controller"] == "policy").any(): events.setdefault(int(tk[0]), []).append("EXPERT TAKEOVER")
    cl = np.where(d["cmd"] < .5)[0]
    if len(cl): events.setdefault(int(cl[0]), []).append("GRIPPER CLOSE")
    if d["grasped"] is not None and d["grasped"].any(): events.setdefault(int(np.argmax(d["grasped"])), []).append("PINCH CONTACT")
    lift = np.where(d["cube"][:, 2] > .03)[0]
    if len(lift): events.setdefault(int(lift[0]), []).append("CUBE LIFTED")
    xy = np.concatenate((ngc[:, :2], d["gc"][:, :2], d["cube"][:1, :2])); lo, hi = xy.min(0) - .02, xy.max(0) + .02; span = (hi - lo).max()
    to_px = lambda v: (int(500 + 320 * (v[1] - lo[1]) / span), int(20 + 320 * (1 - (v[0] - lo[0]) / span)))
    frames, seen = [], []
    for t in range(len(d["rgb"])):
        img = Image.new("RGB", (840, 440), (252, 252, 251)); img.paste(Image.fromarray(d["rgb"][t]).resize((480, 360), Image.NEAREST), (0, 0))
        dr = ImageDraw.Draw(img)
        dr.line([to_px(g) for g in ngc[:, :2]], fill=NOMINAL, width=2)
        c = to_px(d["cube"][t, :2]); dr.rectangle([c[0] - 6, c[1] - 6, c[0] + 6, c[1] + 6], outline=CUBE, width=2)
        for i in range(1, t + 1):
            dr.line([to_px(d["gc"][i - 1, :2]), to_px(d["gc"][i, :2])], fill=POLICY if d["controller"][i] == "policy" else EXPERT, width=2)
        g = to_px(d["gc"][t, :2]); dr.ellipse([g[0] - 4, g[1] - 4, g[0] + 4, g[1] + 4], fill=INK)
        dr.text((490, 345), "top-down: gray=expert path, blue=policy, green=expert-driven", fill=MUTED)
        if t in events: seen.extend(f"t={t}: {e}" for e in events[t])
        lines = [f"{a.title}   step {t}   phase {phase(d['cmd'][min(t, len(d['cmd']) - 1)], d['gc'][t], d['cube'][t])}   controller {d['controller'][t]}",
                 f"joint dist to expert trajectory {dist[t]:.3f} rad   grasp-center path deviation {path_mm[t]:5.1f} mm   gripper cmd {d['cmd'][min(t, len(d['cmd']) - 1)]:.2f}   cube z {100 * d['cube'][t, 2]:.1f} cm"]
        for i, s in enumerate(lines): dr.text((8, 366 + 16 * i), s, fill=INK)
        dr.text((8, 366 + 36), "events: " + (" | ".join(seen[-4:]) if seen else "-"), fill=(235, 104, 52) if seen else MUTED)
        frames.append(np.asarray(img))
    imageio.mimsave(a.output, frames, duration=100, loop=0)
    print(a.output, len(frames), "frames; events", events)


if __name__ == "__main__":
    main()
