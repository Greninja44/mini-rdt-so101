"""Regression tests for the corrected cluster-bootstrap structure (docs/research/capacity_density_recovery_spec.md).

The earlier implementation resampled shared numeric seed IDs across capacities. These tests pin the required structure: positions are
condition-stratified and shared across capacities, seeds are independent per capacity but reused across that capacity's conditions.
"""
import numpy as np
import pytest

from evaluation.recovery_bootstrap import bootstrap_logistic, draw, structure

CAPS = ["m2.0", "m19.5"]
CONDS = ["r10.0", "r15.0"]


def rows(success=lambda k, c, p, s: (p % 2 == 0)):
    out = []
    for k in CAPS:
        for c in CONDS:
            for p in range(6):
                for s in (0, 1, 2):
                    out.append({"capacity": k, "condition": c, "position": f"{c}#{p}", "seed": s,
                                "success": float(success(k, c, p, s)), "distance": 1.0 if c == "r10.0" else 1.5,
                                "log2_params": 0.0 if k == "m2.0" else 3.28})
    return out


def test_positions_are_condition_stratified_and_shared_across_capacities():
    _, _, positions, seeds = structure(rows())
    rng = np.random.default_rng(7)
    for _ in range(50):
        position_draw, _ = draw(rng, positions, seeds)
        for c in CONDS:
            assert len(position_draw[c]) == len(positions[c])          # same size as the condition's own position set
            assert set(position_draw[c]) <= set(positions[c])          # never borrows another condition's positions
    # the draw is a single mapping, so every capacity necessarily sees the identical position multiset
    assert set(position_draw) == set(CONDS)


def test_seeds_are_independent_across_capacities_but_shared_within_one():
    _, _, positions, seeds = structure(rows())
    rng = np.random.default_rng(11)
    differed = 0
    for _ in range(200):
        _, seed_draw = draw(rng, positions, seeds)
        assert set(seed_draw) == set(CAPS)
        assert all(len(v) == 3 for v in seed_draw.values())
        differed += seed_draw[CAPS[0]] != seed_draw[CAPS[1]]
    assert differed > 100, "capacity seed draws must be independent, not a single shared draw"


def test_fixed_rng_seed_makes_intervals_reproducible():
    a = bootstrap_logistic(rows(), ["distance"], replicates=100)
    b = bootstrap_logistic(rows(), ["distance"], replicates=100)
    assert a["distance"]["bootstrap95"] == b["distance"]["bootstrap95"]


def test_recovers_a_planted_effect_and_reports_structure():
    planted = rows(success=lambda k, c, p, s: c == "r10.0")      # success depends only on distance
    fit = bootstrap_logistic(planted, ["distance"], replicates=300)
    assert fit["distance"]["coef"] < 0                            # larger distance -> lower success
    assert fit["distance"]["bootstrap95"][1] < 0                  # interval excludes zero
    assert fit["n_rollouts"] == len(planted) and fit["rng_seed"] == 20260930


def test_absent_cells_are_skipped_rather_than_fabricated():
    partial = [r for r in rows() if not (r["capacity"] == "m19.5" and r["seed"] == 2)]
    fit = bootstrap_logistic(partial, ["distance"], replicates=50)
    assert fit["n_rollouts"] == len(partial)
