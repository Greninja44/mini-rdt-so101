# Capacity × density scaling: infrastructure deviation log

## 2026-09-23 — duplicate detached launcher

The original capacity pipeline (PID/PGID 21681) was already running. A later monitoring action misidentified its detached process as terminated and launched a second pipeline (PID/PGID 255224). Between 2026-09-22 19:25 UTC and 2026-09-23 02:22 UTC, both launchers entered the same resumable `m4.3_r7.5_seed2` and `m9.1_r10.0_seed0` directories. This temporarily violated the pre-registered maximum of two training workers.

The duplicate process group was terminated at 2026-09-23 02:23 UTC. A subsequent process audit found that the original launcher's eager evaluator had started two simulation workers while its two trainers remained active. At 2026-09-23 02:59 UTC that process group was stopped and the committed corrected launcher (`29e0276`) was started. It resumes both active training checkpoints exactly and defers evaluation until all training workers finish, keeping the total at two. No historical density-sweep artifact was touched and no capacity artifact was deleted manually.

Scientific handling:

- This is an infrastructure deviation, not a hyperparameter, data, architecture, seed, or evaluation-protocol change.
- The involved runs are retained; they are not selectively restarted.
- Before analysis, every raw/EMA checkpoint from these two runs will be loaded, its recorded configuration, seed, data manifest and final step checked, and its SHA-256 included in the capacity artifact manifest.
- If either checkpoint fails integrity or configuration validation, the affected run is stopped and reported as an infrastructure failure rather than silently retrained. The pre-registration and hypotheses remain unchanged.

## 2026-09-23 — WSL process interruption

At 2026-09-23 16:33 UTC, following a WSL disconnect, the corrected detached launcher was no longer present. All 25 completed runs and their artifacts remained present. `m9.1_r7.5_onesided_seed1` and seed 2 had preserved `last.pt` checkpoints at steps 60,000 and 2,000, respectively; both were resumed in place at 16:34 UTC by the same corrected two-worker launcher. This is an infrastructure interruption only: no completed run, seed, dataset, model configuration, or evaluation setting was changed.

At 2026-09-24 04:05 UTC, a second WSL crash again stopped the launcher. Thirty completed runs remained intact. `m19.5_r10.0_seed0` and seed 1 were resumed in place from their saved step-0 and step-28,000 checkpoints at 04:06 UTC, again with exactly two workers and unchanged configuration.

At 2026-09-24 14:58 UTC, a third WSL crash stopped the launcher after 32 completed runs. `m19.5_r15.0_seed0` and `m19.5_r10.0_seed2` were resumed in place from their saved step-36,000 and step-0 checkpoints. No completed result or fixed experimental setting changed.
