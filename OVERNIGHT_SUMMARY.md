# Overnight summary (2026-09-18 → 2026-09-19): why the 2M TinyRDT fails closed-loop, and what helps

Model size never changed (≈2.01M trainable). Every experiment was pre-registered in `CLAUDE_PROGRESS.md` before its result. Detailed
reports: `CLOSED_LOOP_DIAGNOSTIC_REPORT.md`, `CORRECTIVE_DATA_REPORT.md`, `PHASE6_REPORT.md`. Numbers are successes on identical seeds
(10 memorised CLEAN10 cubes, or 10 held-out validation cubes); max 150 steps; software rendering.

## Bottom line
1. **Corrective data does not make the 2M TinyRDT robust.**
   - Normal success: DAgger 26/40, DAgger at equal clean exposure 28/40, perturbation 18/40, against the clean baseline's 28/40.
   - Recovery: no improvement.
2. **The failure is early arm placement caused by scene ambiguity, not the gripper and not the diffusion sampler.**
   - With the expert's arm, the policy's own gripper timing succeeds 40/40.
   - If the expert drives only the first 5–9 steps, the unchanged policy succeeds on all 10 seeds.
   - All demos start from the identical home pose, and their trajectories stay within 0.008 rad of each other for ~6 steps. The policy's
     tracking error there is 0.01–0.05 rad.
   - So the state-dominant policy lands closer to a *neighbouring* scene's trajectory and follows it. Failing grasps are offset toward the
     nearest other cube (median cosine 0.92).
3. **Model-side fixes tried, none solved it at 2M:**
   - spatial visual tokens: 29/40;
   - state dropout: 27/40;
   - image-only: 18/40. It fixes the lock-on seeds but loses arm precision;
   - 80-demo density: 22/40 on memorised seeds, 20/40 on held-out.
4. **The practical lever: execute long chunks.**
   - K ≥ 8 (8–16 actions open-loop per replan) is best in every model.
   - Memorised seeds: 73/80 pooled over A, S, C40k and D80 at K=8/16, against 50% at K=1.
   - Held-out cubes (D80, first generalisation look): 15/20 at K ≥ 8, with expert replay 10/10.
   - The one persistent memorised failure is ep7, the most crowded cube.

## Key tables

| model (all ≈2.01M) | memorised K=1 / 2 / 4 / 8 / 16 | normal total (K 1–8) | recovery /80 |
|---|---|---|---|
| A clean CLEAN10 | 5 / 6 / 8 / 9 / 9 | 28/40 | 63 |
| C40k +DAgger (equal clean exposure) | 7 / 5 / 7 / 9 / 9 | 28/40 | 51 |
| S spatial tokens | 7 / 5 / 7 / 10 / 10 | 29/40 | 61 |
| SD state dropout 0.3 | 6 / 3 / 10 / 8 / – | 27/40 | 60 |
| IO image-only | 4 / 4 / 5 / 5 / – | 18/40 | 39 |
| D80 (80 training demos) | 4 / 5 / 4 / 9 / 8 | 22/40 | – |
| D80 on held-out validation cubes | 3 / 5 / 5 / 7 / 8 | 20/40 | – |

Oracle split (A): expert arm + policy gripper **40/40**; policy arm + expert gripper rule 19/40. Expert prefix N = 0 / 1 / 2 / 3 / 5 / 9
steps, then A at K=4: 8 / 9 / 7 / 9 / 10 / 10 out of 10.

## Bugs found and fixed tonight
- `expert.py` is not the expert that generated the dataset. It was reconstructed bit-exactly (`simulation/legacy_expert.py`).
- Exact counterfactual labels need a full `mjData` copy: the IK reads kinematics that mj_step leaves one substep stale.
- Recovery filenames collapsed all magnitudes (`with_suffix` ate ".01").
- The per-seed perturbation sign depended on the `--episodes` subset.
- Orchestrators waited on process names that matched launcher shells. They now wait on files and log markers.
- NaN medians in the analysis for unfinished conditions.

## Recommendation (needs your decision)
- The strict "memorised scenes reliable" gate is ~91% at K≥8 (ep7 remains) and 50–70% at small K. It is **not** 100%.
- Evidence suggests the remaining problem is a closed-loop state-ambiguity issue that persists at 80 demos, not a memorisation problem.
- Two options:
  - **(A)** Accept K=8 (fixed a priori) as the execution setting and start the 80-demo held-out benchmark on validation seeds (not test),
    with the current 2M model. First look: 15/20 at K≥8.
  - **(B)** Keep working at 10 demos on the start ambiguity. Candidates:
    - temporal ensembling of overlapping chunks;
    - a longer horizon (H=32);
    - varied start poses in the demos, so that the start state is not identical across scenes. This is the root cause, and likely the
      most effective fix, but it needs new data.

My recommendation is B's last item (varied start poses) as the next single experiment, then A. No scaling until one of them is done.

## Reproduction
`scripts/run_closed_loop_diagnostics.sh`, `scripts/run_corrective_phase.sh`, `scripts/run_phase5_remaining.sh`, `scripts/run_phase6.sh`,
`scripts/run_phase6c.sh`, `scripts/run_phase7a.sh`, `scripts/run_phase7b.sh`, `scripts/run_phase7c.sh`, `scripts/run_phase8.sh`,
`scripts/run_phase8b.sh`. Then `.venv/bin/python -m evaluation.phase5_analysis`. Back up `artifacts/` (hashes in `ARTIFACT_MANIFEST.json`).
