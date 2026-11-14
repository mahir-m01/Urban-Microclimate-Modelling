"""
S13, step 1: is the U-Net's remaining error in the level of each case or in the pattern inside it?

Two measurements on the saved S11 tuned predictions (no training):

1. Level test. Each case's prediction is rescaled so its open-ground mean equals another estimate
   of the case's mean speed: the S06c LightGBM out-of-fold prediction (the H1 idea, done after the
   fact), the average of that and the U-Net's own mean, or the true mean (an upper bound no model
   can use). The LightGBM prediction for a case comes from a model that never saw its group.
2. Error anatomy. How the squared error splits between idealized and realistic cases, between level
   and pattern, and how concentrated it is in the worst cases.

Reads results/preds/s11_unet_tuned_seed{0,1,2}/ (rebuilt from the saved weights with
`python -m urbanwind.dl.train --config pipeline/S11_unet_baseline/configs/unet_tuned.yaml --seed <s> --repredict`),
pipeline/S11_unet_baseline/outputs/per_case_metrics.csv and the S06c out-of-fold predictions.
Writes outputs/level_test.csv and outputs/error_anatomy.csv.

Usage: uv run python pipeline/S13_hybrid/s13_level_pattern.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from urbanwind.dl.train import fold_table
from urbanwind.metrics import case_metrics

STAGE = pathlib.Path(__file__).resolve().parent
REPO = STAGE.parents[1]
OUT = STAGE / "outputs"
PREDS = REPO / "results" / "preds"
SEEDS = [f"s11_unet_tuned_seed{s}" for s in range(3)]
ENSEMBLE = "mean of the 3 seeds"
S06C_OOF = REPO / "pipeline" / "S06_case_models" / "outputs" / "s06c" / "oof_predictions.csv"
S11_PER_CASE = REPO / "pipeline" / "S11_unet_baseline" / "outputs" / "per_case_metrics.csv"
LEVELS = ("U-Net", "LightGBM", "average of U-Net and LightGBM", "true mean (upper bound)")
TOP_K = (1, 5, 10, 20)


def rescale_to_level(pred: np.ndarray, valid: np.ndarray, level: float) -> np.ndarray:
    """The prediction multiplied by one factor, so its mean over the valid cells equals ``level``."""
    return pred * (level / float(pred[valid].mean()))


def pooled_r2(rows: pd.DataFrame) -> float:
    """R² over all cells of all cases, from per-case sums."""
    n, s, ss, sse = rows["n"].sum(), rows["sum_t"].sum(), rows["sum_t2"].sum(), rows["sse"].sum()
    return float(1 - sse / (ss - s * s / n))


def r2(true: pd.Series, pred: pd.Series) -> float:
    return float(1 - ((pred - true) ** 2).sum() / ((true - true.mean()) ** 2).sum())


def load_case(run: str, case: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Truth, prediction and building mask of one case; the ensemble averages the seeds' predictions."""
    runs = SEEDS if run == ENSEMBLE else [run]
    preds = []
    for r in runs:
        with np.load(PREDS / r / f"{case}.npz") as f:
            truth, mask = f["y_true"], f["mask"]
            preds.append(np.nan_to_num(f["y_pred"]).astype(np.float64))
    return truth, np.mean(preds, axis=0), mask


def level_test(run: str, cases: list[str], lgbm: pd.Series, dev: pd.DataFrame) -> pd.DataFrame:
    rows, means = {level: [] for level in LEVELS}, []
    for case in cases:
        truth, pred, mask = load_case(run, case)
        valid = ~mask
        t = truth[valid].astype(np.float64)
        own, true_mean = float(pred[valid].mean()), float(t.mean())
        means.append({"true": true_mean, "U-Net": own, "LightGBM": float(lgbm[case])})
        targets = dict(zip(LEVELS, (own, float(lgbm[case]), 0.5 * (own + float(lgbm[case])), true_mean)))
        for level, value in targets.items():
            q = rescale_to_level(pred, valid, value)
            m = case_metrics(y_true=truth, y_pred=q, mask=mask, case=case)
            rows[level].append(m | {"subset": dev.loc[case, "subset"], "sum_t": t.sum(), "sum_t2": (t * t).sum(),
                                    "sse": ((q[valid] - t) ** 2).sum()})
    means = pd.DataFrame(means)
    out = []
    for level, r in rows.items():
        r = pd.DataFrame(r)
        real = r[r["subset"] == "realistic"]
        out.append({"run": run, "level_from": level, "n_cases": len(r), "pooled_r2": pooled_r2(r),
                    "median_case_r2": r["r2"].median(), "realistic_pooled_r2": pooled_r2(real),
                    "realistic_median_case_r2": real["r2"].median(), "mean_mae": r["mae"].mean(),
                    "level_r2": r2(means["true"], means[level]) if level in means else np.nan})
    return pd.DataFrame(out)


def error_anatomy(per_case: pd.DataFrame) -> pd.DataFrame:
    """One row per measure, one column per seed."""
    cols = {}
    for run in SEEDS:
        pc = per_case[per_case["run"] == run].assign(sse=lambda d: d["n"] * d["rmse"] ** 2,
                                                       sst=lambda d: d["n"] * d["var_true"])
        real = pc[pc["subset"] == "realistic"].sort_values("sse", ascending=False)
        grand = (real["mean_true"] * real["n"]).sum() / real["n"].sum()
        between = (real["n"] * (real["mean_true"] - grand) ** 2).sum()
        col = {"realistic share of cells": real["n"].sum() / pc["n"].sum(),
               "realistic share of squared error": real["sse"].sum() / pc["sse"].sum(),
               "realistic error share from level": (real["n"] * real["level_err"]).sum() / real["sse"].sum(),
               "realistic error share from pattern": (real["n"] * real["pattern_err"]).sum() / real["sse"].sum(),
               "realistic truth variance share within cases": real["sst"].sum() / (real["sst"].sum() + between)}
        for k in TOP_K:
            col[f"realistic error share of the worst {k} cases"] = real["sse"].head(k).sum() / real["sse"].sum()
            col[f"realistic cell share of the worst {k} cases"] = real["n"].head(k).sum() / real["n"].sum()
        cols[run] = col
    return pd.DataFrame(cols).rename_axis("measure").reset_index()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dev = fold_table("grouped")
    oof = pd.read_csv(S06C_OOF).query("model == 'lightgbm' and features == 'full' and target == 'U_mean'")
    lgbm = oof.set_index("case")["y_pred"]
    cases = sorted(p.stem for p in (PREDS / SEEDS[0]).glob("*.npz"))
    if len(cases) != len(dev) or set(cases) != set(lgbm.index):
        raise SystemExit(f"expected the {len(dev)} development cases in both inputs, found {len(cases)} predictions")
    level = pd.concat([level_test(run, cases, lgbm, dev) for run in [*SEEDS, ENSEMBLE]], ignore_index=True)
    level.to_csv(OUT / "level_test.csv", index=False, float_format="%.4f")
    anatomy = error_anatomy(pd.read_csv(S11_PER_CASE))
    anatomy.to_csv(OUT / "error_anatomy.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 200, "display.precision", 4):
        print(level.to_string(index=False), "\n")
        print(anatomy.to_string(index=False))


if __name__ == "__main__":
    main()
