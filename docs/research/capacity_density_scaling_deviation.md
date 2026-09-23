# Capacity × density scaling: infrastructure deviation log

## 2026-09-23 — duplicate detached launcher

The original capacity pipeline (PID/PGID 21681) was already running. A later monitoring action misidentified its detached process as terminated and launched a second pipeline (PID/PGID 255224). Between 2026-09-22 19:25 UTC and 2026-09-23 02:22 UTC, both launchers entered the same resumable `m4.3_r7.5_seed2` and `m9.1_r10.0_seed0` directories. This temporarily violated the pre-registered maximum of two training workers.

The duplicate process group was terminated at 2026-09-23 02:23 UTC. The original group continues with exactly two trainers. No historical density-sweep artifact was touched and no capacity artifact was deleted manually. The working launcher was previously amended to resume an interrupted run from `last.pt` rather than remove its directory (`29e0276`); the original already-running process retains its loaded script, while any future restart uses that safe version.

Scientific handling:

- This is an infrastructure deviation, not a hyperparameter, data, architecture, seed, or evaluation-protocol change.
- The involved runs are retained; they are not selectively restarted.
- Before analysis, every raw/EMA checkpoint from these two runs will be loaded, its recorded configuration, seed, data manifest and final step checked, and its SHA-256 included in the capacity artifact manifest.
- If either checkpoint fails integrity or configuration validation, the affected run is stopped and reported as an infrastructure failure rather than silently retrained. The pre-registration and hypotheses remain unchanged.
