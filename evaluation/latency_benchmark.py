"""Inference latency of each capacity (docs/research/capacity_density_scaling_spec.md).

`TinyRDTPolicy.predict` (10 DDIM steps, batch 1, identical sampler for every size) on a fixed recorded observation, on the GPU and on the
CPU, after warm-up. Simulation latency only: nothing here is a real-robot claim.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from training.policy import TinyRDTPolicy

MODELS = {"m2.0": "artifacts/density_sweep/models/r10.0_seed0/ema_last.pt",
          "m4.3": "artifacts/capacity_scaling/models/m4.3_r10.0_seed0/ema_last.pt",
          "m9.1": "artifacts/capacity_scaling/models/m9.1_r10.0_seed0/ema_last.pt",
          "m19.5": "artifacts/capacity_scaling/models/m19.5_r10.0_seed0/ema_last.pt"}


def bench(ck, device, n):
    torch.set_num_threads(2)
    p = TinyRDTPolicy(ck, device=device, sampling_steps=10, seed=0)
    ep = Path("artifacts/density_sweep/data/r10.0_eval/episodes/episode_000000")
    rgb = np.load(ep / "rgb.npy")[0]; state = np.r_[np.load(ep / "joint_pos.npy")[0], np.load(ep / "gripper.npy")[0]].astype(np.float32)
    for _ in range(5): p.predict(rgb, state)
    if device == "cuda": torch.cuda.synchronize()
    t = []
    for _ in range(n):
        s = time.perf_counter(); p.predict(rgb, state)
        if device == "cuda": torch.cuda.synchronize()
        t.append(time.perf_counter() - s)
    t = 1000 * np.array(t)
    return {"median_ms": float(np.median(t)), "p90_ms": float(np.percentile(t, 90)), "n": n,
            "params": sum(v.numel() for k, v in p.model.state_dict().items() if not k.startswith("vision."))}


def main():
    a = argparse.ArgumentParser(); a.add_argument("--output", required=True); a = a.parse_args()
    out = {"sampler": "DDIM-10, batch 1, identical for every size", "control_period_ms_at_20hz": 50.0, "models": {}}
    for name, ck in MODELS.items():
        if not Path(ck).exists(): continue
        out["models"][name] = {"cuda": bench(ck, "cuda", 50) if torch.cuda.is_available() else None, "cpu": bench(ck, "cpu", 20)}
        print(name, json.dumps(out["models"][name]), flush=True)
    Path(a.output).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
