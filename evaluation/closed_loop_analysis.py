"""Why does accurate offline prediction fail closed-loop? Post-hoc analysis.

Reads the balanced sweep produced by scripts/run_closed_loop_diagnostics.sh.
The expert reference is the physically replayed expert rollout (identical to
the dataset, plus grasp-center / contact signals). Nothing is re-simulated
except (a) frozen-encoder features of stored frames and (b) the expert's IK
label at policy states, computed kinematically from stored joint positions.

Definitions (fixed before looking at results; documented in CLAUDE_PROGRESS.md):
  policy phase per step, from the policy's own command and geometry:
    approach: gripper cmd >= 0.5 and grasp-center xy > 20 mm from cube
    descent : gripper cmd >= 0.5 and xy <= 20 mm
    grasp   : gripper cmd <  0.5 and cube z < 0.03 m
    lift    : gripper cmd <  0.5 and cube z >= 0.03 m
  rollout failure phase (first applicable):
    approach: grasp center never within 20 mm xy of the cube with gripper open
    descent : reached the cube but never commanded close
    grasp   : commanded close but never achieved a two-pad pinch
    drop    : cube lifted >= 0.03 m, then fell below 0.02 m
    lift    : pinched but never lifted to success height (and no drop)
  divergence onset: first step where max |q_policy - q_expert| over the five
    arm joints exceeds the threshold for 3 consecutive steps. Thresholds:
    0.015 rad (~ offline worst-joint MAE: beyond reconstruction noise) and
    0.03 rad (measured grasp tolerance, evaluation/grasp_sensitivity.py).
    Time-free: grasp-center distance to the expert's grasp-center path > 5 mm.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

from evaluation.research_audit import save_json
from simulation.controllers import position_ik_step
from simulation.env import PickCubeConfig, SO101PickCubeEnv
from simulation.expert import ExpertConfig

SUCCESS_C, FAIL_C, REF_C, GRID_C = "#2a78d6", "#eb6834", "#52514e", "#d9d8d4"
PHASES = ["approach", "descent", "grasp", "lift"]
EXPERT_PHASE = {"HOME": "approach", "MOVE_ABOVE_OBJECT": "approach", "DESCEND": "descent", "CLOSE_GRIPPER": "grasp", "LIFT": "lift", "HOLD": "lift"}


def load_rollouts(folder: Path):
    rows = []
    for j in sorted(folder.glob("ep*_k*.json")):
        r = json.loads(j.read_text()); r["npz"] = str(j.with_suffix(".npz")); rows.append(r)
    return rows


def rot_angle_deg(a, b):
    R = a.reshape(3, 3).T @ b.reshape(3, 3)
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def policy_phase(cmd, gc, cube):
    xy = np.linalg.norm(gc[:2] - cube[:2])
    if cmd >= .5: return "approach" if xy > .020 else "descent"
    return "grasp" if cube[2] < .03 else "lift"


def onset(dev, threshold, run=3):
    above = dev > threshold
    for t in range(len(above) - run + 1):
        if above[t:t + run].all(): return t
    return None


def classify(rec):
    cmd = rec["executed"][:, 5]; gc = rec["grasp_center"]; cube = rec["cube"]
    xy = np.linalg.norm(gc[:-1, :2] - cube[:-1, :2], axis=1)
    closed = cmd < .5; z = cube[:, 2]
    lifted = bool(z.max() >= .03); peak = int(z.argmax())
    dropped = lifted and bool((z[peak:] < .02).any())
    pinched = bool(rec["grasped"].any())
    if not (xy[~closed] <= .020).any() and not closed.any(): return "approach", lifted, dropped
    if not closed.any(): return "descent", lifted, dropped
    if not pinched: return "grasp", lifted, dropped
    if dropped: return "drop", lifted, dropped
    return "lift", lifted, dropped


class ExpertLabeler:
    """Expert DLS-IK arm command at an arbitrary arm configuration (DAgger label)."""
    def __init__(self):
        self.env = SO101PickCubeEnv(PickCubeConfig(render_observations=False)); self.cfg = ExpertConfig()

    def __call__(self, q_arm, cube_ref, lifting):
        e = self.env; e.data.qpos[e._joint_qposadr[:5]] = q_arm; mujoco.mj_forward(e.model, e.data)
        target = np.array([cube_ref[0], cube_ref[1], self.cfg.lift_height]) if lifting else cube_ref + np.array([0, 0, self.cfg.grasp_offset])
        return position_ik_step(e.model, e.data, e._grasp_site_id, target)


def analyze_rollout(r, ref, demo_states, labeler):
    rec = dict(np.load(r["npz"])); T = len(rec["executed"])
    N = len(ref["executed"])
    idx = np.minimum(np.arange(T + 1), N)
    q, qe = rec["state"][:, :5], ref["state"][idx, :5]
    a, ae = rec["executed"][:, :5], ref["executed"][np.minimum(np.arange(T), N - 1), :5]
    gc, gce = rec["grasp_center"], ref["grasp_center"][idx]
    dev = {"joint_max_abs_rad": np.abs(q - qe).max(1), "joint_norm_rad": np.linalg.norm(q - qe, axis=1),
           "action_norm_rad": np.r_[np.linalg.norm(a - ae, axis=1), np.nan],
           "cube_mm": 1000 * np.linalg.norm(rec["cube"][:, :3] - ref["cube"][idx, :3], axis=1),
           "grasp_center_mm": 1000 * np.linalg.norm(gc - gce, axis=1),
           "ee_ori_deg": np.array([rot_angle_deg(rec["ee"][t, 3:], ref["ee"][idx[t], 3:]) for t in range(T + 1)]),
           "grasp_center_path_mm": 1000 * np.array([np.linalg.norm(ref["grasp_center"] - g, axis=1).min() for g in gc]),
           "state_nn_own_rad": np.array([np.linalg.norm(ref["state"][:, :5] - s, axis=1).min() for s in q]),
           "state_nn_all_rad": np.array([np.linalg.norm(demo_states - s, axis=1).min() for s in q])}
    phases = [policy_phase(rec["executed"][min(t, T - 1), 5], gc[t], rec["cube"][t]) for t in range(T + 1)]
    expert_phase = [EXPERT_PHASE[s] for s in ref["expert_state"][np.minimum(np.arange(T + 1), len(ref["expert_state"]) - 1)]]
    failure, lifted, dropped = ("success", bool(rec["cube"][:, 2].max() >= .03), False) if r["success"] else classify(rec)
    closes = np.where(rec["executed"][:, 5] < .5)[0]; close_t = int(closes[0]) if len(closes) else None
    exp_close = int(np.argmax(ref["executed"][:, 5] < .5))
    grasp = None
    if close_t is not None:
        after = slice(close_t, min(close_t + 13, T + 1))
        grasp = {"close_t": close_t, "expert_close_t": exp_close, "close_timing_vs_expert_steps": close_t - exp_close,
                 "grasp_center_minus_cube_mm": (1000 * (gc[close_t] - rec["cube"][close_t, :3])).tolist(),
                 "xy_offset_mm": float(1000 * np.linalg.norm(gc[close_t, :2] - rec["cube"][close_t, :2])),
                 "z_offset_mm": float(1000 * (gc[close_t, 2] - rec["cube"][close_t, 2])),
                 "expert_z_offset_mm": float(1000 * (ref["grasp_center"][exp_close, 2] - ref["cube"][exp_close, 2])),
                 "ee_ori_vs_expert_close_deg": rot_angle_deg(rec["ee"][close_t, 3:], ref["ee"][exp_close, 3:]),
                 "joint_dev_vs_expert_close_rad": (q[close_t] - ref["state"][exp_close, :5]).tolist(),
                 "gripper_cmd": float(rec["executed"][close_t, 5]),
                 "max_pad_contacts_next_12": int(rec["pad_contacts"][after].max()), "pinched_next_12": bool(rec["grasped"][after].any()),
                 "misaligned_vs_expert_4mm_rule": bool(np.linalg.norm(gc[close_t] - (rec["cube"][close_t, :3] + [0, 0, .001])) > .004)}
    # DAgger-style label error at descent/grasp/lift policy states vs off-demo distance.
    cube_ref = rec["cube"][0, :3]; label = []
    for t in range(T):
        if phases[t] == "approach": continue
        target = labeler(q[t], cube_ref, phases[t] == "lift")
        label.append((t, phases[t], float(np.linalg.norm(a[t] - target)), float(dev["state_nn_all_rad"][t])))
    chunks, steps = rec["chunks"], rec["chunk_steps"]; consistency = []
    for i in range(len(steps) - 1):
        shift = int(steps[i + 1] - steps[i]); H = chunks.shape[1]
        if shift >= H: continue
        d = np.abs(chunks[i, shift:, :5] - chunks[i + 1, :H - shift, :5]).max(1)
        consistency.append({"t": int(steps[i + 1]), "phase": phases[int(steps[i + 1])], "next_action_jump_rad": float(d[0]), "overlap_mean_rad": float(d.mean())})
    return {"episode": r["episode"], "seed": r["seed"], "k": r["k"], "success": r["success"], "steps": r["steps"], "time_s": r["steps"] / 20,
            "replans": r["replans"], "grasped": bool(rec["grasped"].any()), "lifted": lifted, "dropped": dropped, "failure_phase": failure,
            "final_cube_pose": rec["cube"][-1].tolist(), "final_robot_state": rec["state"][-1].tolist(),
            "onset_0.015rad": onset(dev["joint_max_abs_rad"], .015), "onset_0.03rad": onset(dev["joint_max_abs_rad"], .03),
            "onset_path_5mm": onset(dev["grasp_center_path_mm"], 5.), "expert_close_t": exp_close,
            "phase_at_onset_0.03": expert_phase[o] if (o := onset(dev["joint_max_abs_rad"], .03)) is not None else None,
            "grasp": grasp, "label_error": label, "consistency": consistency,
            "_dev": dev, "_expert_phase": expert_phase}


def on_distribution_consistency(policy, dataset, ids, needs_cube):
    """Replan every step on the RECORDED expert observations (K=1 analogue)."""
    rows = []
    for ep in ids:
        root = Path(dataset) / "episodes" / f"episode_{ep:06d}"
        rgb, jp, gr, cube = (np.load(root / f) for f in ("rgb.npy", "joint_pos.npy", "gripper.npy", "cube_pose.npy"))
        es = np.load(root / "expert_state.npy"); prev = None
        for t in range(len(rgb)):
            st = np.r_[jp[t], gr[t]].astype(np.float32)
            c = (policy.predict(rgb[t], st, cube[t]) if needs_cube else policy.predict(rgb[t], st)).numpy()
            if prev is not None:
                d = np.abs(prev[1:, :5] - c[:-1, :5]).max(1)
                rows.append({"phase": EXPERT_PHASE[str(es[t])], "next_action_jump_rad": float(d[0]), "overlap_mean_rad": float(d.mean())})
            prev = c
    return rows


def summarize_phase(rows, key):
    out = {}
    for ph in PHASES:
        v = [r[key] for r in rows if r["phase"] == ph]
        if v: out[ph] = {"n": len(v), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "max": float(np.max(v))}
    return out


def vision_drift(pairs, dataset, ids, checkpoint, device):
    from training.policy import TinyRDTPolicy
    vis = TinyRDTPolicy(checkpoint, device=device).model.vision
    def feats(frames):
        out = []
        with torch.no_grad():
            for i in range(0, len(frames), 64):
                x = torch.as_tensor(frames[i:i + 64], device=device).permute(0, 3, 1, 2)
                out.append(vis(x).cpu())
        return torch.cat(out)
    demo, demo_ep = [], []
    for ep in ids:
        f = feats(np.load(Path(dataset) / "episodes" / f"episode_{ep:06d}" / "rgb.npy")); demo.append(f); demo_ep += [ep] * len(f)
    demo = torch.cat(demo); demo_ep = torch.tensor(demo_ep)
    adjacent = torch.cat([(d[1:] - d[:-1]).norm(dim=1) for d in torch.split(demo, [int((demo_ep == e).sum()) for e in ids])])
    cross = torch.stack([torch.cdist(demo[i:i + 1], demo[demo_ep != demo_ep[i]]).min() for i in range(len(demo))])
    scale = {"demo_adjacent_frame_median": float(adjacent.median()), "demo_nearest_other_scene_median": float(cross.median())}
    for r, an in pairs:
        f = feats(np.load(r["npz"])["rgb"])
        an["_dev"]["vision_nn_own"] = torch.cdist(f, demo[demo_ep == r["episode"]]).min(1).values.numpy()
    return scale


def plot_rollout(path, a, title):
    dev, t = a["_dev"], np.arange(len(a["_dev"]["joint_max_abs_rad"]))
    fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    panels = [("joint_max_abs_rad", "max |q - q_expert| (rad)", [.015, .03]), ("action_norm_rad", "||a - a_expert|| arm (rad)", []),
              ("grasp_center_mm", "grasp-center deviation (mm)", [5.]), ("ee_ori_deg", "EE orientation deviation (deg)", [])]
    for ax, (key, label, lines) in zip(axes, panels):
        ax.plot(t, dev[key], color=SUCCESS_C if a["success"] else FAIL_C, lw=2, label="time-aligned")
        if key == "grasp_center_mm": ax.plot(t, dev["grasp_center_path_mm"], color=REF_C, lw=1.2, ls="--", label="distance to expert path (time-free)"); ax.legend(fontsize=7, frameon=False)
        for y in lines: ax.axhline(y, color=REF_C, lw=.8, ls=":")
        ax.set_ylabel(label, fontsize=8); ax.grid(color=GRID_C, lw=.5); ax.spines[["top", "right"]].set_visible(False)
        for ph_t in (a["expert_close_t"],): ax.axvline(ph_t, color=REF_C, lw=.8, ls="-.")
        if a["onset_0.03rad"] is not None: ax.axvline(a["onset_0.03rad"], color=FAIL_C, lw=.8)
    axes[-1].set_xlabel("control step (20 Hz); dash-dot = expert close step; solid = onset (0.03 rad)")
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_overlay(path, analyses, key, ylabel, threshold):
    fig, ax = plt.subplots(figsize=(8, 4))
    for a in analyses:
        ax.plot(a["_dev"][key][:60], color=SUCCESS_C if a["success"] else FAIL_C, lw=1, alpha=.7)
    ax.plot([], [], color=SUCCESS_C, label="success"); ax.plot([], [], color=FAIL_C, label="failure"); ax.legend(frameon=False, fontsize=8)
    if threshold: ax.axhline(threshold, color=REF_C, lw=.8, ls=":")
    ax.set_xlabel("control step (first 60)"); ax.set_ylabel(ylabel); ax.grid(color=GRID_C, lw=.5); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", default="artifacts/closed_loop_v2")
    p.add_argument("--dataset", default="artifacts/pickcube_smoke100_rgb160")
    p.add_argument("--tinyrdt", default="artifacts/research_audit/cosine_x0_hold_ema/ema_last.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    sweep = Path(a.sweep); out = sweep / "analysis"; out.mkdir(exist_ok=True)
    ids = [0, 2, 3, 4, 5, 6, 7, 8, 9, 11]
    refs = {}
    for ep in ids:
        rec = dict(np.load(sweep / "tinyrdt_ema" / f"ep{ep:03d}_expert_replay.npz"))
        rec["expert_state"] = np.load(Path(a.dataset) / "episodes" / f"episode_{ep:06d}" / "expert_state.npy"); refs[ep] = rec
    replay = [json.loads((sweep / "tinyrdt_ema" / f"ep{ep:03d}_expert_replay.json").read_text()) for ep in ids]
    demo_states = np.concatenate([refs[ep]["state"][:, :5] for ep in ids])
    labeler = ExpertLabeler(); report = {"definitions": __doc__, "expert_replay": replay, "policies": {}}
    calib = []
    for ep in ids:
        fake = {"episode": ep, "seed": 0, "k": 0, "success": True, "steps": 0, "replans": 0}
        rec = refs[ep]; T = len(rec["executed"])
        for t in range(T):
            ph = policy_phase(rec["executed"][t, 5], rec["grasp_center"][t], rec["cube"][t])
            if ph == "approach": continue
            calib.append({"phase": ph, "err": float(np.linalg.norm(rec["executed"][t, :5] - labeler(rec["state"][t, :5], rec["cube"][0, :3], ph == "lift"))), "nn": 0.0, "success": True})
    report["label_calibration_on_expert_states"] = {ph: float(np.median([c["err"] for c in calib if c["phase"] == ph])) for ph in ("descent", "grasp", "lift") if any(c["phase"] == ph for c in calib)}
    all_analyses = {}
    for name in ("tinyrdt_ema", "bc_rgb", "bc_privileged"):
        rows = load_rollouts(sweep / name)
        analyses = [analyze_rollout(r, refs[r["episode"]], demo_states, labeler) for r in rows]
        all_analyses[name] = (rows, analyses)
    scale = vision_drift(list(zip(*all_analyses["tinyrdt_ema"])), a.dataset, ids, a.tinyrdt, a.device)
    for name, (rows, analyses) in all_analyses.items():
        table = {}
        for k in (1, 2, 4, 8):
            sel = [x for x in analyses if x["k"] == k]
            table[f"K={k}"] = {"episodes": sorted(x["episode"] for x in sel), "n": len(sel), "successes": sum(x["success"] for x in sel),
                               "failures": sum(not x["success"] for x in sel), "success_rate": (sum(x["success"] for x in sel) / len(sel)) if sel else None,
                               "failure_phases": {ph: sum(x["failure_phase"] == ph for x in sel) for ph in ("approach", "descent", "grasp", "lift", "drop", "other")}}
        consistency = [c for x in analyses for c in x["consistency"]]
        labels = [{"phase": l[1], "err": l[2], "nn": l[3], "success": x["success"]} for x in analyses for l in x["label_error"]]
        report["policies"][name] = {"table": table,
            "rollouts": [{k2: v for k2, v in x.items() if not k2.startswith("_") and k2 not in ("label_error", "consistency")} for x in analyses],
            "closed_loop_consistency_by_phase": {"next_action_jump": summarize_phase(consistency, "next_action_jump_rad"), "overlap_mean": summarize_phase(consistency, "overlap_mean_rad")},
            "label_error_vs_offdemo": label_bins(labels)}
    from training.policy import BCPolicy, TinyRDTPolicy
    on = {"tinyrdt_ema": on_distribution_consistency(TinyRDTPolicy(a.tinyrdt, device=a.device), a.dataset, ids, False),
          "bc_rgb": on_distribution_consistency(BCPolicy("artifacts/research_audit/bc_rgb/best.pt", a.device), a.dataset, ids, False),
          "bc_privileged": on_distribution_consistency(BCPolicy("artifacts/research_audit/bc_privileged/best.pt", a.device), a.dataset, ids, True)}
    for name, rows in on.items():
        report["policies"][name]["on_distribution_consistency_by_phase"] = {"next_action_jump": summarize_phase(rows, "next_action_jump_rad"), "overlap_mean": summarize_phase(rows, "overlap_mean_rad")}
    # Observation drift, TinyRDT: state / vision off-demo distance around divergence onset.
    rows, analyses = all_analyses["tinyrdt_ema"]
    drift = []
    for r, x in zip(rows, analyses):
        v = x["_dev"]["vision_nn_own"]; d = x["_dev"]
        o = x["onset_0.03rad"]
        drift.append({"episode": x["episode"], "k": x["k"], "success": x["success"], "onset_0.03rad": o,
                      "state_nn_own_at_steps": {str(t): float(d["state_nn_own_rad"][t]) for t in (5, 10, 15, 20, 25, 30) if t < len(v)},
                      "vision_nn_own_at_steps": {str(t): float(v[t]) for t in (5, 10, 15, 20, 25, 30) if t < len(v)},
                      "max_state_nn_all_before_close": float(d["state_nn_all_rad"][:x["expert_close_t"] + 1].max()),
                      "max_vision_nn_own_before_close": float(v[:x["expert_close_t"] + 1].max())})
    report["tinyrdt_observation_drift"] = {"vision_scale": scale, "rollouts": drift}
    save_json(out / "analysis.json", report)
    for name, (rows, analyses) in all_analyses.items():
        plot_overlay(out / f"{name}_joint_dev_overlay.png", analyses, "joint_max_abs_rad", "max |q - q_expert| (rad)", .03)
        plot_overlay(out / f"{name}_grasp_center_path_overlay.png", analyses, "grasp_center_path_mm", "distance to expert path (mm)", 5.)
    for x in all_analyses["tinyrdt_ema"][1]:
        plot_rollout(out / f"tinyrdt_ep{x['episode']:03d}_k{x['k']}_divergence.png", x,
                     f"TinyRDT ep {x['episode']} K={x['k']}: {'SUCCESS' if x['success'] else 'FAIL (' + x['failure_phase'] + ')'}")
    print(json.dumps({n: report["policies"][n]["table"] for n in report["policies"]}, indent=1))


def label_bins(labels):
    """Expert-label error of the executed action, binned by off-demo state distance."""
    edges = [0, .01, .02, .05, .1, .2, 10]; out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        v = [l["err"] for l in labels if lo <= l["nn"] < hi]
        if v: out.append({"offdemo_rad": f"[{lo},{hi})", "n": len(v), "median_label_error_rad": float(np.median(v)), "p90": float(np.percentile(v, 90))})
    return out


if __name__ == "__main__":
    main()
