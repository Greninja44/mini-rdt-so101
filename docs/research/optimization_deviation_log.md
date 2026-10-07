# Optimization scaling study — deviation and interruption log

Every departure from `optimization_scaling_spec.md`, and every interruption of a run, recorded as it happened.
None of the entries below changes a gate threshold, a candidate recipe, or a selection rule.

## I1 — 2026-10-06, Stage A 19.5M: WSL shutdown mid-training
The host VM shut down during `stageA_m19.5_r10.0_seed0`. The run resumed from its atomic checkpoint with
`--steps` unchanged and completed 80,222 updates (wall 89 min). The recipe is the constant-LR baseline, so the
schedule horizon is not involved; resume exactness for scheduled recipes is covered by
`tests/test_optimization_recipe.py`.

## I2 — 2026-10-06, Stage B `m9.1_lr0.5x`: trainer outlived a service stop
The study service was stopped on request at 07:47Z. The stop removed the runner shell but the training
subprocess continued as an orphan and finished normally at 08:35Z, writing its checkpoint and `result.json`.
Two consequences, both verified rather than assumed:

- **The run is complete and valid.** `last.pt` records step 80221 (80,222 updates), carries its embedded
  `ema_model`, and `run_config.learning_rate` is 5e-4 as the manifest specifies. `result.json` is only written
  after the training loop returns, so the run was not truncated.
- **Its evaluation was truncated, not its training.** The evaluation worker *was* killed, leaving 2 of 10
  training-scene rollouts. `evaluation/closed_loop.py` skips rollouts that already carry a completed record, so
  the restarted queue completes the remaining 8 without recomputing the first 2.

The pipeline log has no `train ... lr0.5x` completion line for this run because the logging shell was the process
that died. The run's own artifacts, not the log, are the record.

## D1 — 2026-10-07, Stage B retained at 9.1M although its screening cell passes on baseline
Stage A shows the baseline recipe already satisfies the training-health gate at 9.1M (train MAE 0.00745, 10/10
training scenes, spike rate 2.74% within the 3.4% bound), so the stopping rule would send 9.1M to Stage F
without screening. Stage B is still run at 9.1M exactly as registered, because Amendment 3(a) recorded this
situation in advance: the 9.1M screen's purpose is to show whether a candidate *breaks* a healthy cell, and
dropping it after seeing Stage A would be an outcome-dependent change to the design. No run is added or removed.
