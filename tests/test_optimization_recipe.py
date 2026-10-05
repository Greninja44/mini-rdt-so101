"""Optimization-recipe knobs (docs/research/optimization_scaling_spec.md).

The study varies only optimizer settings, so two properties must hold:
1. the defaults reproduce the historical recipe exactly, keeping every earlier run reproducible;
2. an interrupted run resumed under a NON-default recipe still matches an uninterrupted one bit-for-bit, since the schedule is a pure
   function of the step and therefore needs no checkpointed scheduler state. An interruption must keep --steps fixed, because the schedule
   horizon is part of the recipe; --stop-after emulates that faithfully.
"""
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from training.audit_overfit import lr_multiplier

DATASET = Path("artifacts/density_sweep/data/r20.0_train")
BASE = ["--train-all", "--schedule", "cosine", "--prediction", "x0", "--padding", "hold", "--batch-size", "8",
        "--learning-rate", "0.001", "--device", "cuda" if torch.cuda.is_available() else "cpu",
        "--eval-interval", "60", "--ema-decay", "0.999", "--hidden-dim", "192", "--layers", "4", "--heads", "6"]


def train(out, steps, extra=(), resume=None, stop_after=None):
    cmd = [sys.executable, "-m", "training.audit_overfit", "--dataset", str(DATASET), "--output", str(out),
           "--steps", str(steps), "--seed", "0", *BASE, *extra]
    if stop_after: cmd += ["--stop-after", str(stop_after)]
    if resume: cmd += ["--resume", str(resume)]
    r = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONWARNINGS": "ignore", "PATH": "/usr/bin:/bin", "HOME": str(Path.home())})
    assert r.returncode == 0, r.stderr[-2000:]


def identical(a, b, keys=("model", "ema_model")):
    ca, cb = (torch.load(p, map_location="cpu", weights_only=False) for p in (a, b))
    assert ca["step"] == cb["step"]
    for key in keys:
        assert all(torch.equal(ca[key][k], cb[key][k]) for k in ca[key]), f"{key} differs"
    sa, sb = ca["optimizer"]["state"], cb["optimizer"]["state"]
    assert all(torch.equal(sa[i][t], sb[i][t]) for i in sa for t in ("exp_avg", "exp_avg_sq") if t in sa[i])


def test_constant_schedule_is_exactly_one_everywhere():
    assert all(lr_multiplier(s, 1000, 0, "constant", 0.1) == 1.0 for s in (0, 1, 500, 999))


def test_warmup_then_cosine_follows_the_declared_shape():
    assert lr_multiplier(0, 100, 10, "cosine", 0.1) == pytest.approx(0.1)      # first update, ramping up
    assert lr_multiplier(9, 100, 10, "cosine", 0.1) == pytest.approx(1.0)      # end of warmup
    assert lr_multiplier(10, 100, 10, "cosine", 0.1) == pytest.approx(1.0)     # cosine starts at the peak
    assert lr_multiplier(99, 100, 10, "cosine", 0.1) == pytest.approx(0.1, abs=1e-3)   # decays to the floor
    mults = [lr_multiplier(s, 100, 10, "cosine", 0.1) for s in range(10, 100)]
    assert all(a >= b - 1e-12 for a, b in zip(mults, mults[1:]))               # monotone after warmup


@pytest.mark.skipif(not DATASET.exists(), reason="needs the density datasets")
def test_defaults_reproduce_the_historical_recipe(tmp_path):
    train(tmp_path / "implicit", 120)
    train(tmp_path / "explicit", 120, ["--weight-decay", "1e-4", "--grad-clip", "1.0", "--warmup-steps", "0", "--lr-schedule", "constant"])
    identical(tmp_path / "implicit/last.pt", tmp_path / "explicit/last.pt")


@pytest.mark.skipif(not DATASET.exists(), reason="needs the density datasets")
def test_resume_is_exact_under_a_non_default_recipe(tmp_path):
    recipe = ["--warmup-steps", "30", "--lr-schedule", "cosine", "--weight-decay", "1e-2", "--grad-clip", "0.5", "--grad-log-interval", "20"]
    train(tmp_path / "full", 120, recipe)
    train(tmp_path / "part", 120, recipe, stop_after=60)        # same schedule horizon, interrupted at 60
    train(tmp_path / "part", 120, recipe, resume=tmp_path / "part/last.pt")
    identical(tmp_path / "full/last.pt", tmp_path / "part/last.pt")
