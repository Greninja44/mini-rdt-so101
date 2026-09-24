# Capacity × density scaling: interim progress record

**Status at 2026-09-23 16:35 UTC:** all 15 4.3M-policy runs have trained and completed their pre-registered K=8 closed-loop evaluation. The 9.1M and 19.5M matrix is incomplete. This document is a progress record, not a changed hypothesis, a model-selection decision, or a final analysis.

## Completed 4.3M member

The exact configuration is `d=256`, 5 Transformer blocks, 8 heads, 4d MLP, with 4,304,902 trainable policy parameters. It uses the frozen MobileNet, the same datasets, 80,222-step matched-exposure recipe, seeds 0/1/2, DDIM-10, K=8 and physics-v2 criteria specified in `capacity_density_scaling_spec.md`.

| condition | 2M frozen baseline | 4.3M seeds | 4.3M aggregate | invalid physics rollouts |
|---|---:|---:|---:|---:|
| r10 surrounded | 18/24 (75.0%) | 7, 7, 7 | 21/24 (87.5%) | 0 |
| r15 surrounded | 7/12 (58.3%) | 3, 3, 2 | 8/12 (66.7%) | 0 |
| r20 surrounded | 3/9 (33.3%) | 1, 1, 2 | 4/9 (44.4%) | 0 |
| r7.5 one-sided | 17/30 (56.7%) | 6, 8, 6 | 20/30 (66.7%) | 0 |
| r7.5 surrounded control | 28/30 (93.3%) | 10, 9, 4 | 23/24 (95.8%) | 0 |

The apparent improvement is directionally consistent on every currently completed primary condition, but it is **not a capacity-scaling conclusion**: the 9.1M and 19.5M members, their distance curves, interaction model, mechanistic diagnostics, cost measures and final clustered analysis are required before deciding whether capacity expands range or merely provides a small hard-condition gain.

## Pause record — 2026-09-24

At the user's instruction, no incomplete training run is resumed. At pause, 32 of the 45 pre-registered training runs had completed; only
the 15 completed 4.3M runs had their full closed-loop/offline evaluation. The remaining finished checkpoints and partial checkpoints are
preserved under `artifacts/capacity_scaling/`, but are intentionally not used to make an unregistered partial conclusion. The WIP PR and
README describe the evaluated 4.3M slice only.
