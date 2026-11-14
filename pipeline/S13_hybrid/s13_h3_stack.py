"""
S13 H3: residual stacking. A LightGBM model corrects the network's prediction cell by cell.

For every open-ground cell the stacker sees the network's out-of-fold prediction and the S07 pixel
features (local density and heights, wall distance, fetches, sheltering, wind direction, case
descriptors), and learns the network's error there: residual = truth - network prediction. The
hybrid prediction is the network prediction plus the predicted residual, floored at zero.

Protocol, as S07:
- The network is the final S13 model, the three-seed mean of the 200-epoch U-Net
  (results/preds/s13_unet_tuned_e200_mean3/, written by s13_final.py). Every one of its predictions
  is out of fold.
- For outer fold k, the stacker trains on cases of the other four folds only, with N_SAMPLE random
  open-ground cells per case, and predicts every open cell of the fold-k cases.
- Same LightGBM settings as S07. LightGBM and PyTorch cannot share a process on macOS, so this
  script never imports PyTorch; s13_vs_s11.py scores the result in a separate process.

With --local the stacker leaves out the six case-level descriptors and sees only per-cell features
and the network prediction, so it cannot learn corrections that depend on the case as a whole.

With --small the stacker is a much smaller, more regularised LightGBM (SMALL_PARAMS), which can only
learn broad biases of the network, such as under-predicted peaks, rather than fit per-cell noise.

Writes results/preds/s13_h3_stack[_local][_small]/<case>.npz (not tracked) and
outputs/h3[_local][_small]_feature_importance.csv.

Usage: uv run python -m pipeline.S13_hybrid.s13_h3_stack [--local] [--small]
"""

from __future__ import annotations

import pathlib
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from pipeline.S07_pixel_model import s07_pixel_model as s07

STAGE = pathlib.Path(__file__).resolve().parent
OUT = STAGE / "outputs"
PREDS = s07.PREDS
BASE = "s13_unet_tuned_e200_mean3"
NETWORK_FEATURE = "network_prediction"
N_CASE = len(s07.CASE_FEATURES)  # the case descriptors are the last columns of the S07 feature matrix
SMALL_PARAMS = s07.PARAMS | {"n_estimators": 200, "num_leaves": 15, "min_child_samples": 1000}


def training_rows(case_fold: np.ndarray, k: int) -> np.ndarray:
    """Rows a fold-k stacker may train on: cells of cases outside fold k."""
    return case_fold != k


def network_prediction(case: str, valid: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """The network's out-of-fold prediction on the valid cells, after checking it is on the same grid."""
    with np.load(PREDS / BASE / f"{case}.npz") as f:
        if not np.array_equal(~f["mask"], valid) or not np.allclose(f["y_true"][valid], truth):
            raise ValueError(f"{case}: the network prediction is not on the S07 grid")
        return f["y_pred"][valid].astype(np.float32)


def case_table(case: str, row: pd.Series, local_only: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Features (S07 pixel features plus the network prediction), truth, network prediction and validity."""
    X, y, valid, _, _ = s07.case_pixels(case, row)
    if local_only:
        X = X[:, :-N_CASE]
    net = network_prediction(case, valid, y)
    return np.hstack([X, net[:, None]]), y, net, valid


def build_training_table(df: pd.DataFrame, local_only: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(s07.SEED)
    Xs, residuals, case_idx = [], [], []
    for i, row in df.iterrows():
        X, y, net, _ = case_table(row["case"], row, local_only)
        pick = rng.choice(len(y), size=min(s07.N_SAMPLE, len(y)), replace=False)
        Xs.append(X[pick])
        residuals.append((y - net)[pick])
        case_idx.append(np.full(len(pick), i))
    return np.vstack(Xs), np.concatenate(residuals), np.concatenate(case_idx)


def save_field(run: str, case: str, valid: np.ndarray, truth: np.ndarray, pred: np.ndarray) -> None:
    true_map = np.full(valid.shape, np.nan, np.float32)
    pred_map = np.full(valid.shape, np.nan, np.float32)
    true_map[valid], pred_map[valid] = truth, pred
    (PREDS / run).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PREDS / run / f"{case}.npz", y_true=true_map, y_pred=pred_map, mask=~valid)


def main(local_only: bool = False, small: bool = False) -> None:
    t0 = time.time()
    tag = ("_local" if local_only else "") + ("_small" if small else "")
    run = f"s13_h3_stack{tag}"
    params = SMALL_PARAMS if small else s07.PARAMS
    df, _ = s07.dev_cases()
    missing = [c for c in df["case"] if not (PREDS / BASE / f"{c}.npz").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} cases have no {BASE} prediction; run s13_final.py first")
    X, residual, case_idx = build_training_table(df, local_only)
    print(f"training table: {X.shape[0]:,} cells x {X.shape[1]} features [{time.time() - t0:.0f}s]", flush=True)
    case_fold = df["outer_fold"].to_numpy()[case_idx]
    importances = []
    for k in sorted(df["outer_fold"].unique()):
        rows = training_rows(case_fold, k)
        held_out = set(df.loc[df["outer_fold"] == k, "case"])
        assert not held_out & set(df["case"].to_numpy()[case_idx[rows]]), "a fold-k case leaked into training"
        model = lgb.LGBMRegressor(**params).fit(X[rows], residual[rows])
        importances.append(model.booster_.feature_importance("gain"))
        for _, row in df[df["outer_fold"] == k].iterrows():
            Xc, yc, net, valid = case_table(row["case"], row, local_only)
            save_field(run, row["case"], valid, yc, np.maximum(net + model.predict(Xc), 0.0))
        print(f"fold {k} done [{time.time() - t0:.0f}s]", flush=True)
    _, _, _, names, _ = s07.case_pixels(df["case"].iloc[0], df.iloc[0])
    feature_names = [*names, *([] if local_only else s07.CASE_FEATURES), NETWORK_FEATURE]
    imp = np.mean(importances, axis=0)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": feature_names, "gain_share": imp / imp.sum()}).sort_values(
        "gain_share", ascending=False).to_csv(OUT / f"h3{tag}_feature_importance.csv", index=False, float_format="%.4f")
    print(f"done [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main(local_only="--local" in sys.argv[1:], small="--small" in sys.argv[1:])
