"""The pre-registered spatial split (docs/research/physics_v2_generalization_spec.md) is frozen, nested and leak-free."""
import json
from pathlib import Path

import numpy as np
import pytest

SPLIT = Path("docs/research/physics_v2_generalization_split.json")
DATASET = Path("artifacts/pickcube_physics_v2_rgb160")


@pytest.fixture(scope="module")
def split():
    return json.loads(SPLIT.read_text())


def test_nested_subsets_and_no_leak(split):
    s = {k: set(v) for k, v in split["train_subsets"].items()}
    assert [len(s[k]) for k in ("CLEAN10", "TRAIN20", "TRAIN40", "TRAIN80")] == [10, 20, 40, 80]
    assert s["CLEAN10"] == {0, 2, 3, 4, 5, 6, 7, 8, 9, 11} and s["CLEAN10"] < s["TRAIN20"] < s["TRAIN40"] < s["TRAIN80"]
    hole = {int(r["source"].split("_")[-1]) for r in split["test"] if r["category"].startswith("B")}
    assert len(hole) == 20 and not hole & s["TRAIN80"] and hole | s["TRAIN80"] == set(range(100))


def test_positions_unique_and_categorised(split):
    rows = split["validation"] + split["test"]; xy = np.array([r["cube_xy"] for r in rows])
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2) + np.eye(len(xy))
    assert d.min() > 0  # no duplicate coordinates
    fresh = np.array([r["source"] == "fresh" for r in rows])
    assert d[fresh].min() >= 0.002 - 1e-12  # fresh positions keep 2 mm from every other val/test position
    assert min(r["nn_train80_mm"] for r in rows) >= 2.0 - 1e-9
    cats = [r["category"][0] for r in split["test"]]
    assert cats.count("A") == cats.count("B") == cats.count("C") == 20
    assert all(r["outside_workspace_mm"] >= 3 for r in split["test"] if r["category"][0] == "C")
    assert all(r["outside_workspace_mm"] == 0 for r in split["test"] + split["validation"] if r["category"][0] != "C")
    lo, hi = split["rule"]["hole_band_y_m"]
    assert all(not lo <= r["cube_xy"][1] <= hi for r in split["validation"] + split["test"] if r["category"][0] in "Av")


@pytest.mark.skipif(not DATASET.exists(), reason="needs the physics-v2 dataset")
def test_split_regenerates_identically(split):
    from data.generalization_split import build
    assert json.loads(json.dumps(build(str(DATASET)))) == split
