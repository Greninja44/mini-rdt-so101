"""Table-collision audit (read-only): does the robot penetrate the table, and does any success depend on it?

Measures geometry numerically from the simulator (never from pixels). Rollouts are re-simulated by
replaying their recorded executed actions; the simulator is deterministic and replay fidelity is
checked against the recorded joint states. No physics, geometry, expert or success code is changed.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import glob
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from simulation.env import PickCubeConfig, SO101PickCubeEnv

CUBE_HALF = 0.01
DISTAL = ("gripper", "moving_jaw_so101_v1")
TRACKED = ("upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1")


def gname(m, g): return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])}/geom{g}"


class Probe:
    def __init__(self, env):
        self.env = env; m = env.model; self.m = m
        tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table"); self.table = tid
        self.table_top = float(m.geom_pos[tid][2] + m.geom_size[tid][2])
        self.cube = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.pads = env._grasp_pad_ids
        self.body = {g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) for g in range(m.ngeom)}
        self.meshes = [g for g in range(m.ngeom) if m.geom_dataid[g] >= 0 and self.body[g] in TRACKED]
        self.verts = {g: m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:m.mesh_vertadr[m.geom_dataid[g]] + m.mesh_vertnum[m.geom_dataid[g]]] for g in self.meshes}
        self.robot_bodies = set(range(1, m.nbody)) - {m.geom_bodyid[self.cube]}

    def box_min_z(self, g):
        d = self.env.data; corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * self.m.geom_size[g]
        return float((corners @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min())

    def mesh_min_z(self, g):
        d = self.env.data; return float((self.verts[g] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min())

    def measure(self):
        d, m, e = self.env.data, self.m, self.env
        mesh = {}
        for g in self.meshes:
            key = f"{self.body[g]}[{'collision' if m.geom_group[g] == 3 else 'visual'}]"
            mesh[key] = min(mesh.get(key, np.inf), self.mesh_min_z(g))
        pads = {gname(m, g): self.box_min_z(g) for g in self.pads}
        contacts = []
        for i in range(d.ncon):
            c = d.contact[i]; b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            contacts.append({"pair": f"{gname(m, c.geom1)}<->{gname(m, c.geom2)}", "dist_mm": 1000 * float(c.dist), "z": float(c.pos[2]),
                             "normal_z": float(c.frame[2]), "robot_table": (c.geom1 == self.table and b2 in self.robot_bodies) or (c.geom2 == self.table and b1 in self.robot_bodies),
                             "pad_cube": {c.geom1, c.geom2} <= ({self.cube} | set(self.pads)) and self.cube in (c.geom1, c.geom2)})
        distal_min = min([v for k, v in mesh.items() if k.split("[")[0] in DISTAL] + list(pads.values()))
        return {"eef_z": float(e.end_effector_pose[2]), "grasp_center_z": float(e.grasp_center_position[2]), "cube_z": float(e.cube_pose[2]),
                "table_z": self.table_top, "pad_min_z": pads, "mesh_min_z": mesh, "distal_min_z": distal_min,
                "penetration_mm": 1000 * max(0.0, self.table_top - distal_min),
                "pad_penetration_mm": 1000 * max(0.0, self.table_top - min(pads.values())),
                "pads_below_cube_bottom_mm": 1000 * max(0.0, (e.cube_pose[2] - CUBE_HALF) - min(pads.values())),
                "contacts": contacts, "robot_table_contacts": sum(c["robot_table"] for c in contacts),
                "pad_cube_contacts": [c for c in contacts if c["pad_cube"]],
                "cube_table_contacts": sum({"cube_geom", "table"} == set(c["pair"].split("<->")) for c in contacts)}


def replay(env, probe, seed, actions, recorded_q=None, render=None):
    env.config = replace(env.config, render_observations=False)
    obs, _ = env.reset(seed=seed); rows = [probe.measure()]; frames = [render()] if render else None; qerr = 0.0; succ_steps = []
    for t, a in enumerate(actions):
        obs, _, term, _, info = env.step(a); r = probe.measure(); r.update(t=t + 1, cmd=float(a[5]), grasped=bool(info["grasped"]), success=bool(info["success"]), elevated=bool(info["elevated"]))
        rows.append(r)
        if render: frames.append(render())
        if recorded_q is not None and t + 1 < len(recorded_q): qerr = max(qerr, float(np.abs(obs["joint_pos"] - recorded_q[t + 1][:5]).max()))
        if term: break
    return rows, frames, qerr


def summary(rows):
    pen = [r["penetration_mm"] for r in rows]; t_max = int(np.argmax(pen))
    first = next((i for i, p in enumerate(pen) if p > 0), None)
    return {"max_penetration_mm": max(pen), "t_max": t_max, "first_penetration_t": first, "steps_penetrating": sum(p > 0 for p in pen),
            "max_pad_penetration_mm": max(r["pad_penetration_mm"] for r in rows),
            "robot_table_contacts_total": sum(r["robot_table_contacts"] for r in rows),
            "deepest_part_at_max": min(rows[t_max]["mesh_min_z"].items(), key=lambda kv: kv[1])[0] if rows[t_max]["mesh_min_z"] else None,
            "success": any(r.get("success") for r in rows), **grasp_metrics(rows)}


def grasp_metrics(rows):
    """Grasp validity: is the pinch a side pinch on a cube resting on a solid table?

    At the first pinched step and over pinched steps while the cube is still low (< 3 cm):
    pad penetration into the table, cube pushed into the table (center below its 10 mm rest height),
    and the direction of pad-cube contact normals (|n_z| ~ 0 = side faces, ~ 1 = top/bottom faces).
    """
    pinched = [r for r in rows[1:] if r.get("grasped")]
    if not pinched: return {"pinch": None}
    first = pinched[0]; low = [r for r in pinched if r["cube_z"] < .03] or [first]
    nz = [abs(c["normal_z"]) for r in low for c in r["pad_cube_contacts"]]
    return {"pinch": {"t_first": first["t"], "pad_penetration_at_first_pinch_mm": first["pad_penetration_mm"],
                      "max_pad_penetration_while_pinched_low_mm": max(r["pad_penetration_mm"] for r in low),
                      "cube_pushed_into_table_max_mm": 1000 * max(0.0, CUBE_HALF - min(r["cube_z"] for r in low)),
                      "pads_below_cube_bottom_max_mm": max(r["pads_below_cube_bottom_mm"] for r in low),
                      "pad_cube_normal_abs_z_median": float(np.median(nz)) if nz else None,
                      "frac_vertical_normals": float(np.mean(np.array(nz) > .7)) if nz else None}}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--artifacts", default="artifacts"); p.add_argument("--output", default="docs/audit/table_collision")
    a = p.parse_args(); art, out = Path(a.artifacts), Path(a.output); out.mkdir(parents=True, exist_ok=True)
    env = SO101PickCubeEnv(PickCubeConfig(render_observations=False)); probe = Probe(env); report = {}

    # A. collision configuration (effective model after env init; pads toggled only while closing)
    m = env.model
    report["collision_config"] = [{"geom": gname(m, g), "body": probe.body[g], "type": mujoco.mjtGeom(m.geom_type[g]).name, "group": int(m.geom_group[g]),
                                   "contype": int(m.geom_contype[g]), "conaffinity": int(m.geom_conaffinity[g])} for g in range(m.ngeom)]
    report["table_top_z"] = probe.table_top

    # B/C. the README rollout (TinyRDT EMA, seed 3006, K=8) with frames
    src = art / "closed_loop_v2/tinyrdt_ema/ep006_k8.npz"; meta = json.loads(src.with_suffix(".json").read_text()); rec = np.load(src)
    renderer = mujoco.Renderer(m, height=360, width=480); side = mujoco.MjvCamera(); side.type = mujoco.mjtCamera.mjCAMERA_FREE
    def render():
        renderer.update_scene(env.data, camera="external_rgb"); main_view = renderer.render().copy()
        gc = env.grasp_center_position; side.lookat[:] = [gc[0], gc[1], 0.0]; side.distance = 0.16; side.azimuth = 90.0; side.elevation = -2.0
        renderer.update_scene(env.data, camera=side); return main_view, renderer.render().copy()
    rows, frames, qerr = replay(env, probe, meta["seed"], rec["executed"], rec["state"], render)
    s = summary(rows); s.update(source=str(src), checkpoint=str(art / "research_audit/cosine_x0_hold_ema/ema_last.pt"), seed=meta["seed"], k=meta["k"], replay_joint_error_rad=qerr)
    close_t = next((r["t"] for r in rows[1:] if r["cmd"] < .5), None); pinch_t = next((r["t"] for r in rows[1:] if r["grasped"]), None)
    contact_t = next((r["t"] for r in rows[1:] if r["pad_cube_contacts"]), None); lift_t = next((r["t"] for r in rows[1:] if r["cube_z"] > CUBE_HALF + .003), None)
    succ_t = next((r["t"] for r in rows[1:] if r["success"]), None)
    events = {"first_penetration": s["first_penetration_t"], "max_penetration": s["t_max"], "gripper_close_command": close_t,
              "first_pad_cube_contact": contact_t, "first_pinch": pinch_t, "first_cube_lift_3mm": lift_t, "success": succ_t}
    s["events"] = events
    hold = [r for r in rows if succ_t is not None and succ_t - 4 <= r.get("t", -1) <= succ_t]
    s["success_window"] = [{"t": r["t"], "cube_z_mm": 1000 * r["cube_z"], "pinched": r["grasped"], "pad_cube_contacts": len(r["pad_cube_contacts"]),
                            "max_abs_normal_z": max((abs(c["normal_z"]) for c in r["pad_cube_contacts"]), default=None),
                            "pads_below_cube_bottom_mm": r["pads_below_cube_bottom_mm"], "penetration_mm": r["penetration_mm"]} for r in hold]
    s["per_step"] = [{k: (v if not isinstance(v, float) else round(v, 5)) for k, v in r.items() if k not in ("contacts",)} for r in rows]
    report["readme_rollout"] = s
    for name, t in events.items():
        if t is None: continue
        r = rows[t]; main_view, side_view = frames[t]
        img = Image.new("RGB", (960, 440), (252, 252, 251)); img.paste(Image.fromarray(main_view), (0, 0)); img.paste(Image.fromarray(side_view), (480, 0))
        dr = ImageDraw.Draw(img)
        pairs = sorted({c["pair"] for c in r["contacts"]}) or ["none"]
        lines = [f"{name}  t={t} ({t / 20:.2f}s)   left: task camera   right: side view at table height",
                 f"EEF z {1000 * r['eef_z']:.1f} mm   grasp-center z {1000 * r['grasp_center_z']:.1f} mm   table z {1000 * r['table_z']:.1f} mm   cube z {1000 * r['cube_z']:.1f} mm",
                 f"lowest distal robot geometry {1000 * r['distal_min_z']:.1f} mm  -> penetration {r['penetration_mm']:.1f} mm   pad min z " + ", ".join(f"{k.split('_')[0]} {1000 * v:.1f}" for k, v in r["pad_min_z"].items()),
                 f"contacts: {'; '.join(pairs)[:120]}   robot-table contacts: {r['robot_table_contacts']}"]
        for i, l in enumerate(lines): dr.text((8, 364 + 18 * i), l, fill=(11, 11, 11))
        img.save(out / f"readme_rollout_{name}_t{t:03d}.png")

    # D. expert demonstrations (recorded actions replayed; the legacy expert reproduces them bit-exactly)
    experts = []
    for ep in (0, 2, 3, 4, 5, 6, 7, 8, 9, 11):
        root = art / "pickcube_smoke100_rgb160/episodes" / f"episode_{ep:06d}"; seed = json.loads((root / "episode.json").read_text())["seed"]
        q = np.concatenate((np.load(root / "joint_pos.npy"), np.load(root / "gripper.npy")), 1)
        rows_e, _, qe = replay(env, probe, seed, np.load(root / "action.npy"), q)
        experts.append({"episode": ep, "replay_joint_error_rad": qe, **{k: v for k, v in summary(rows_e).items()}})
    report["expert_demos"] = experts
    # TinyRDT closed-loop rollouts (all 40 of the clean model)
    policies = []
    for f in sorted(glob.glob(str(art / "closed_loop_v2/tinyrdt_ema/ep*_k*.npz"))):
        meta_p = json.loads(Path(f).with_suffix(".json").read_text()); r_p = np.load(f)
        rows_p, _, qp = replay(env, probe, meta_p["seed"], r_p["executed"], r_p["state"])
        sp = summary(rows_p)
        pinch_rows = [r for r in rows_p[1:] if r["grasped"]]
        sp.update(episode=meta_p["episode"], k=meta_p["k"], recorded_success=meta_p["success"], replay_joint_error_rad=qp,
                  pads_below_cube_bottom_max_mm_while_pinched=max((r["pads_below_cube_bottom_mm"] for r in pinch_rows), default=0.0),
                  max_abs_pad_cube_normal_z_while_lifted=max((abs(c["normal_z"]) for r in rows_p[1:] if r["cube_z"] > .02 for c in r["pad_cube_contacts"]), default=None))
        policies.append(sp)
    report["tinyrdt_rollouts"] = policies
    (out / "audit.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(json.dumps({"readme_rollout": {k: v for k, v in s.items() if k not in ("per_step",)}}, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
