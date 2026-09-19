"""POST-HOC diagnostic (added after tier 1; not pre-registered). How precisely do the policy's frozen visual features encode cube XY?

The input is the first frame, where the robot is always at home, so only the cube differs. Features are the pooled frozen MobileNet
token the policy receives.

Readouts, fitted on the TRAIN80 first frames only (ridge λ by 5-fold CV inside TRAIN80):
- ridge regression to cube XY;
- 1-nearest-neighbour in feature space.

Each readout is evaluated on VAL10 and the 56 benchmark TEST positions. Neither touches policy training.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluation.research_audit import load


@torch.no_grad()
def first_frame_features(model, root, ids, device):
    rgb = np.stack([np.load(Path(root) / "episodes" / f"episode_{i:06d}" / "rgb.npy", mmap_mode="r")[0] for i in ids])
    xy = np.array([np.load(Path(root) / "episodes" / f"episode_{i:06d}" / "cube_pose.npy")[0, :2] for i in ids])
    f = model.vision(torch.from_numpy(rgb).permute(0, 3, 1, 2).to(device)).reshape(len(ids), -1).float().cpu().numpy()
    return f, xy


def ridge(x, y, lam):
    m, s = x.mean(0), x.std(0).clip(1e-3); z = (x - m) / s; ym = y.mean(0)
    w = np.linalg.solve(z.T @ z + lam * np.eye(z.shape[1]), z.T @ (y - ym))
    return lambda q: ym + ((q - m) / s) @ w


def main():
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", default="artifacts/physics_v2_generalization/models/TRAIN80/ema_last.pt")
    p.add_argument("--split", default="docs/research/physics_v2_generalization_split.json"); p.add_argument("--dataset", default="artifacts/pickcube_physics_v2_rgb160")
    p.add_argument("--val", default="artifacts/pickcube_physics_v2_gen_val_rgb160"); p.add_argument("--test", default="artifacts/pickcube_physics_v2_gen_test_rgb160")
    p.add_argument("--output", default="artifacts/physics_v2_generalization/diagnostic_vision_probe.json"); a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"; ck, model, *_ = load(a.checkpoint, device)
    split = json.loads(Path(a.split).read_text()); train_ids = split["train_subsets"]["TRAIN80"]
    ep_ids = lambda root: sorted(int(q.name.split("_")[1]) for q in (Path(root) / "episodes").glob("episode_*"))
    ftr, ytr = first_frame_features(model, a.dataset, train_ids, device)
    fva, yva = first_frame_features(model, a.val, ep_ids(a.val), device)
    tids = ep_ids(a.test); fte, yte = first_frame_features(model, a.test, tids, device); cat = {r["index"]: r["category"] for r in split["test"]}
    folds = np.arange(len(ftr)) % 5; cv = {}
    for lam in (1e-2, 1e-1, 1, 10, 100, 1000):
        err = np.concatenate([np.linalg.norm(ridge(ftr[folds != k], ytr[folds != k], lam)(ftr[folds == k]) - ytr[folds == k], axis=1) for k in range(5)]); cv[lam] = float(1000 * np.median(err))
    lam = min(cv, key=cv.get); f = ridge(ftr, ytr, lam)
    def nn(q):
        d = np.linalg.norm(q[:, None] - ftr[None], axis=2); return ytr[d.argmin(1)]
    def report(pred, y, cats=None):
        e = 1000 * np.linalg.norm(pred - y, axis=1); out = {"median_mm": float(np.median(e)), "p90_mm": float(np.percentile(e, 90)), "max_mm": float(e.max()), "n": len(e)}
        if cats is not None: out["by_category_median_mm"] = {c: float(np.median(e[np.array(cats) == c])) for c in sorted(set(cats))}
        return out, e.tolist()
    cats = [cat[i] for i in tids]
    rtest, rerr = report(f(fte), yte, cats); ntest, nerr = report(nn(fte), yte, cats)
    nearest_true = np.array([ytr[np.linalg.norm(ytr - y, axis=1).argmin()] for y in yte])
    result = {"post_hoc": True, "checkpoint": a.checkpoint, "feature_dim": int(ftr.shape[1]), "ridge_lambda_cv": cv, "ridge_lambda": lam,
              "ridge": {"train_cv_median_mm": cv[lam], "val": report(f(fva), yva)[0], "test": rtest},
              "nearest_neighbour_in_feature_space": {"val": report(nn(fva), yva)[0], "test": ntest,
                  "test_retrieves_geometrically_nearest_training_cube": float(np.mean(np.all(np.isclose(nn(fte), nearest_true), axis=1)))},
              "test_rows": [{"index": i, "category": c, "ridge_error_mm": re, "nn_error_mm": ne} for i, c, re, ne in zip(tids, cats, rerr, nerr)]}
    Path(a.output).write_text(json.dumps(result, indent=1) + "\n"); print(json.dumps({k: v for k, v in result.items() if k != "test_rows"}, indent=1))


if __name__ == "__main__":
    main()
