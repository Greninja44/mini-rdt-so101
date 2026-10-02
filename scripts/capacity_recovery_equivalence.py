"""Gate for the capacity recovery protocol (docs/research/capacity_density_recovery_spec.md).

Proves on the REAL TinyRDT trainer, for every recovery architecture, that an interrupted-and-resumed run reaches exactly the same state as
an uninterrupted run at the same final update: raw weights, EMA weights, optimizer state and every random stream.

The existing unit tests cover the checkpoint mechanism on a toy module; this exercises the actual training entry point, which is what the
recovery runs will use. A failure here stops the recovery launch.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ARCHS = {"m4.3": (256, 5, 8), "m9.1": (320, 7, 10), "m19.5": (416, 9, 13)}
BASE = ["--train-all", "--schedule", "cosine", "--prediction", "x0", "--padding", "hold", "--batch-size", "8",
        "--learning-rate", "0.001", "--device", "cuda", "--eval-interval", "200", "--ema-decay", "0.999"]


def train(dataset, out, steps, arch, seed, resume=None):
    d, layers, heads = arch
    cmd = [sys.executable, "-m", "training.audit_overfit", "--dataset", dataset, "--output", str(out), "--steps", str(steps),
           "--seed", str(seed), "--hidden-dim", str(d), "--layers", str(layers), "--heads", str(heads), *BASE]
    if resume: cmd += ["--resume", str(resume)]
    r = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONWARNINGS": "ignore", "PATH": "/usr/bin:/bin", "HOME": str(Path.home())})
    if r.returncode: raise RuntimeError(f"training failed ({out}): {r.stderr[-2000:]}")


def compare(a, b):
    """Exact comparison of every stateful tensor saved in the two checkpoints."""
    ca = torch.load(a, map_location="cpu", weights_only=False); cb = torch.load(b, map_location="cpu", weights_only=False)
    out = {"step_a": ca["step"], "step_b": cb["step"]}
    for key in ("model", "ema_model"):
        da, db = ca[key], cb[key]
        out[key] = {"identical": all(torch.equal(da[k], db[k]) for k in da),
                    "max_abs_diff": max(float((da[k].float() - db[k].float()).abs().max()) for k in da)}
    sa, sb = ca["optimizer"]["state"], cb["optimizer"]["state"]
    diffs = [float((sa[i][t].float() - sb[i][t].float()).abs().max()) for i in sa for t in ("exp_avg", "exp_avg_sq") if t in sa[i]]
    out["optimizer"] = {"identical": all(d == 0 for d in diffs), "max_abs_diff": max(diffs) if diffs else 0.0,
                        "steps_match": all(sa[i]["step"] == sb[i]["step"] for i in sa if "step" in sa[i])}
    for key in ("train_rng", "noise_rng", "torch_rng"):
        out[key] = {"identical": bool(torch.equal(ca[key].cpu(), cb[key].cpu()))}
    out["cuda_rng"] = {"identical": all(torch.equal(x.cpu(), y.cpu()) for x, y in zip(ca["cuda_rng"], cb["cuda_rng"]))}
    out["all_identical"] = all(v["identical"] for k, v in out.items() if isinstance(v, dict) and "identical" in v)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="artifacts/density_sweep/data/r20.0_train")
    p.add_argument("--work", default="artifacts/capacity_recovery_v1/smoke")
    p.add_argument("--output", default="docs/research/capacity_recovery_equivalence.json")
    p.add_argument("--steps", type=int, default=400); p.add_argument("--interrupt-at", type=int, default=200)
    a = p.parse_args(); work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    result = {"dataset": a.dataset, "steps": a.steps, "interrupt_at": a.interrupt_at, "architectures": {}}
    for name, arch in ARCHS.items():
        full, part = work / f"{name}_full", work / f"{name}_resumed"
        for q in (full, part):
            if q.exists(): __import__("shutil").rmtree(q)
        train(a.dataset, full, a.steps, arch, seed=0)
        train(a.dataset, part, a.interrupt_at, arch, seed=0)                      # "interrupted" run
        train(a.dataset, part, a.steps, arch, seed=0, resume=part / "last.pt")    # resumed to the same final update
        result["architectures"][name] = {"hidden_layers_heads": arch, "last_pt": compare(full / "last.pt", part / "last.pt")}
        print(name, json.dumps(result["architectures"][name]["last_pt"]), flush=True)
    result["gate_passed"] = all(v["last_pt"]["all_identical"] for v in result["architectures"].values())
    Path(a.output).write_text(json.dumps(result, indent=1) + "\n")
    print("GATE PASSED" if result["gate_passed"] else "GATE FAILED")
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
