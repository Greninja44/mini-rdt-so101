"""Cluster bootstrap for the corrected capacity matrix (docs/research/capacity_density_recovery_spec.md).

The protocol fixes the resampling structure, because the earlier implementation resampled shared numeric seed IDs across capacities and so
treated different models' seeds as the same cluster:

- **positions** are resampled with replacement *within each density condition* (condition-stratified) and the same draw is shared by every
  capacity, so capacities are always compared on the same scenes;
- **training seeds** are resampled with replacement *independently within each capacity*, and a capacity's draw is reused across its
  conditions, because one trained model spans all of its conditions' rollouts;
- 4,000 replicates with a fixed RNG seed, so the intervals are reproducible.

`draw` is exposed separately so the resampling structure itself can be regression-tested.
"""
from __future__ import annotations

import numpy as np

REPLICATES = 4000
RNG_SEED = 20260930


def logistic_fit(X, y, ridge=1e-3, iters=200):
    """Newton IRLS with a small ridge, which keeps the fit finite under separation. The ridge is reported wherever this is used."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w)); W = np.clip(p * (1 - p), 1e-9, None)
        H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1]); g = X.T @ (y - p) - ridge * w
        step = np.linalg.solve(H, g); w += step
        if np.abs(step).max() < 1e-9: break
    return w


def structure(rows):
    """Positions per condition (shared across capacities) and seeds per capacity."""
    conditions = sorted({r["condition"] for r in rows})
    capacities = sorted({r["capacity"] for r in rows})
    positions = {c: sorted({r["position"] for r in rows if r["condition"] == c}) for c in conditions}
    seeds = {k: sorted({r["seed"] for r in rows if r["capacity"] == k}) for k in capacities}
    return conditions, capacities, positions, seeds


def draw(rng, positions, seeds):
    """One replicate's clusters: positions resampled per condition (shared), seeds resampled per capacity (independent)."""
    position_draw = {c: list(rng.choice(p, len(p), replace=True)) for c, p in positions.items()}
    seed_draw = {k: list(rng.choice(s, len(s), replace=True)) for k, s in seeds.items()}
    return position_draw, seed_draw


def bootstrap_logistic(rows, predictors, replicates=REPLICATES, rng_seed=RNG_SEED, ridge=1e-3):
    """Fit `success ~ predictors` and return each coefficient with a cluster-bootstrap 95% interval.

    `rows` carry capacity, condition, position, seed, success and every predictor value.
    """
    conditions, capacities, positions, seeds = structure(rows)
    index = {(r["capacity"], r["condition"], r["position"], r["seed"]): r for r in rows}
    design = lambda rs: np.c_[np.ones(len(rs)), np.column_stack([[r[p] for r in rs] for p in predictors])] if predictors else np.ones((len(rs), 1))
    y = np.array([r["success"] for r in rows], float)
    point = logistic_fit(design(rows), y, ridge)
    rng = np.random.default_rng(rng_seed); boots = []
    for _ in range(replicates):
        position_draw, seed_draw = draw(rng, positions, seeds)
        picked = [index[(k, c, p, s)]
                  for k in capacities for c in conditions
                  for p in position_draw[c] for s in seed_draw[k]
                  if (k, c, p, s) in index]
        yy = np.array([r["success"] for r in picked], float)
        if len(set(yy)) < 2: continue
        boots.append(logistic_fit(design(picked), yy, ridge))
    boots = np.array(boots)
    return {name: {"coef": float(point[i]),
                   "bootstrap95": [float(np.percentile(boots[:, i], 2.5)), float(np.percentile(boots[:, i], 97.5))]}
            for i, name in enumerate(["intercept"] + list(predictors))} | {
        "n_rollouts": int(len(rows)), "n_positions": int(sum(len(p) for p in positions.values())),
        "capacities": capacities, "replicates": int(len(boots)), "rng_seed": rng_seed, "ridge": ridge}


def bootstrap_rate(rows, replicates=REPLICATES, rng_seed=RNG_SEED):
    """Cluster-bootstrap interval for a plain success rate over the same structure."""
    conditions, capacities, positions, seeds = structure(rows)
    index = {}
    for r in rows: index.setdefault((r["capacity"], r["condition"], r["position"], r["seed"]), r)
    rng = np.random.default_rng(rng_seed); boots = []
    for _ in range(replicates):
        position_draw, seed_draw = draw(rng, positions, seeds)
        vals = [index[(k, c, p, s)]["success"] for k in capacities for c in conditions
                for p in position_draw[c] for s in seed_draw[k] if (k, c, p, s) in index]
        if vals: boots.append(float(np.mean(vals)))
    k = sum(r["success"] for r in rows)
    return {"successes": int(k), "trials": len(rows), "rate": k / len(rows) if rows else None,
            "bootstrap95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else [None, None],
            "replicates": len(boots), "rng_seed": rng_seed}
