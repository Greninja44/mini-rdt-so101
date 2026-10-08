"""Derive the next stage's manifest from measured training health alone.

The staged logic of docs/research/optimization_scaling_spec.md is mechanical once Stage A/B results
exist, so it is encoded here rather than applied by hand: a capacity whose screen produces a recipe
that passes the training-health gate stops screening and advances to Stage F; a capacity with no
passing recipe moves to the next screening variable (C warmup, D weight decay, E clipping).

Selection never consults held-out success. Ranking is Amendment 3(b): highest training-scene success
on the screening cell, then lowest final training MAE, then the lower learning rate.
"""
import argparse, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "artifacts/optimization_scaling_v1/summary.json"
ARCH = {"m2.0": (192, 4, 6), "m4.3": (256, 5, 8), "m9.1": (320, 7, 10), "m19.5": (416, 9, 13)}
SCREENED = ["m9.1", "m19.5"]          # capacities under study; 2.0M/4.3M are controls
CONDITIONS = ["r7.5", "r7.5_onesided", "r10.0", "r15.0", "r20.0"]
SEEDS = [0, 1, 2]
STEPS = 80222
BASE_LR = 1e-3
LR = {"baseline": 1e-3, "lr0.25x": 2.5e-4, "lr0.5x": 5e-4, "lr2x": 2e-3}
WARMUP = {"warmup2pct": 1604, "warmup5pct": 4011}
WD = {"wd0": 0.0, "wd1e-2": 1e-2}
CLIP = {"clip0.5": 0.5, "clip0.25": 0.25}
NEXT_STAGE = {"stageA": ("stageB", ["lr0.25x", "lr0.5x", "lr2x"]), "stageB": ("stageC", list(WARMUP)), "stageC": ("stageD", list(WD)),
              "stageD": ("stageE", list(CLIP)), "stageE": (None, [])}


def flags(recipe_chain):
    """Compose trainer flags for a chain of recipe components, e.g. ['lr0.5x','warmup2pct']."""
    lr, out = None, []
    for r in recipe_chain:
        if r in LR and r != "baseline":
            lr = LR[r]
        elif r in WARMUP:
            out += ["--warmup-steps", str(WARMUP[r])]
        elif r in WD:
            out += ["--weight-decay", repr(WD[r])]
        elif r in CLIP:
            out += ["--grad-clip", repr(CLIP[r])]
    if lr is not None:
        out = ["--learning-rate", repr(lr)] + out
    return " ".join(out + ["--grad-log-interval", "50"])


ART = ROOT / "artifacts/optimization_scaling_v1"


def is_complete(run):
    """True when this run is trained and its required evaluations are already on disk.

    Completed runs are dropped from the plan so a finished stage stops being re-proposed: otherwise
    the next-stage filename would keep repeating and the service loop's anti-loop guard would halt
    the study while another capacity still had screening left.
    """
    if not (ART / "models" / run["name"] / "result.json").exists():
        return False
    if run["eval_train_scenes"]:
        if len(list((ART / "train_scenes" / run["name"]).glob("ep*_k8.json"))) < 10:
            return False
    if run["eval_heldout"]:
        log = ART / "pipeline.log"
        done = log.exists() and f"held-out evaluation {run['name']} done" in log.read_text()
        if not done:
            return False
    return True


def rank_key(run):
    scene = run["train_scene"]
    rate = scene[0] / scene[1] if scene and scene[1] else 0.0
    chain = run["recipe"].split("+")
    lr = next((LR[c] for c in chain if c in LR), BASE_LR)
    return (-rate, run["train_mae"], lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write the manifest; otherwise print the plan")
    ap.add_argument("--out-dir", default="docs/research")
    a = ap.parse_args()
    s = json.loads(SUMMARY.read_text())
    runs = [dict(v, recipe=v["recipe"]) for v in s["runs"].values()]

    pending = s.get("incomplete", [])
    plan, decisions = [], {}
    for cap in SCREENED:
        waiting = [n for n in pending if f"_{cap}_" in n]
        if waiting:
            decisions[cap] = {"state": "waiting on screening runs", "incomplete": waiting}
            continue
        mine = [r for r in runs if r["capacity"] == cap and r["train_scene"] and r["train_scene"][1] >= 10]
        if not mine:
            decisions[cap] = {"state": "no complete screening run yet"}
            continue
        latest = max((r["stage"] for r in mine), key=lambda x: x)
        passing = sorted([r for r in mine if r["passes_gate"]], key=rank_key)
        if passing:
            sel = passing[0]
            decisions[cap] = {"state": "advance to Stage F", "recipe": sel["recipe"],
                              "why": f"passes gate; {sel['train_scene'][0]}/{sel['train_scene'][1]} scenes, "
                                     f"train MAE {sel['train_mae']:.5f}",
                              "ranked": [r["recipe"] for r in passing]}
            for cond in CONDITIONS:
                for seed in SEEDS:
                    h, l, hd = ARCH[cap]
                    plan.append({"name": f"stageF_{cap}_{sel['recipe']}_{cond}_seed{seed}",
                                 "hidden": h, "layers": l, "heads": hd, "condition": cond, "seed": seed,
                                 "steps": STEPS, "extra": flags(sel["recipe"].split("+")),
                                 "eval_train_scenes": True, "eval_heldout": True})
        else:
            nxt, comps = NEXT_STAGE.get(latest, (None, []))
            best = sorted(mine, key=rank_key)[0]
            if nxt is None:
                decisions[cap] = {"state": "screening exhausted; no recipe passes", "best": best["recipe"]}
                continue
            decisions[cap] = {"state": f"continue screening at {nxt}", "base": best["recipe"],
                              "why": f"no recipe passes the gate; best so far {best['recipe']} "
                                     f"({best['train_scene'][0]}/{best['train_scene'][1]} scenes, "
                                     f"train MAE {best['train_mae']:.5f})",
                              "candidates": comps}
            for comp in comps:
                chain = [c for c in best["recipe"].split("+") if c != "baseline"] + [comp]
                h, l, hd = ARCH[cap]
                plan.append({"name": f"{nxt}_{cap}_{'+'.join(chain)}_r10.0_seed0",
                             "hidden": h, "layers": l, "heads": hd, "condition": "r10.0", "seed": 0,
                             "steps": STEPS, "extra": flags(chain),
                             "eval_train_scenes": True, "eval_heldout": False})

    done = [r["name"] for r in plan if is_complete(r)]
    plan = [r for r in plan if not is_complete(r)]
    print(json.dumps({"decisions": decisions, "runs": len(plan), "already_complete": len(done)}, indent=1))
    if a.write and plan:
        stages = sorted({r["name"].split("_")[0].lower() for r in plan})
        out = ROOT / a.out_dir / f"optimization_{'_'.join(stages)}_manifest.json"
        out.write_text(json.dumps(plan, indent=1) + "\n")
        print(f"wrote {out.relative_to(ROOT)} ({len(plan)} runs)")
    for r in plan:
        print(" ", r["name"], "|", r["extra"])


if __name__ == "__main__":
    main()
