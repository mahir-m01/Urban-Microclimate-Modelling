"""
S13: the final development-set comparison of the three-seed ensembles.

Averages the out-of-fold predictions of the three seeds of a recipe cell by cell (the ensemble), for
the S11 tuned U-Net (100 epochs) and the 200-epoch U-Net, and scores every single seed and both
ensembles on the 482 development cases. Then compares the two ensembles case by case, with 95%
intervals from 2,000 resamples of whole cities and layout families. The test set is never read.

Writes results/preds/<ensemble>/ (not tracked), outputs/final_summary.csv and outputs/final_vs_s11.csv.

Usage: uv run python -m pipeline.S13_hybrid.s13_final
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from pipeline.S11_unet_baseline.s11_results import bootstrap_vs_s07, score_run, summarise
from urbanwind.dl.train import PREDS, fold_table

OUT = pathlib.Path(__file__).resolve().parent / "outputs"
RECIPES = {"s11_unet_tuned": [f"s11_unet_tuned_seed{s}" for s in range(3)],
           "s13_unet_tuned_e200": [f"s13_unet_tuned_e200_seed{s}" for s in range(3)]}


def ensemble_name(recipe: str) -> str:
    return f"{recipe}_mean3"


def write_ensemble(recipe: str, seeds: list[str], cases: list[str]) -> str:
    """Cell-by-cell mean of the seeds' predictions, saved in the evaluator's format."""
    out = PREDS / ensemble_name(recipe)
    out.mkdir(parents=True, exist_ok=True)
    for case in cases:
        files = [np.load(PREDS / run / f"{case}.npz") for run in seeds]
        truth, mask = files[0]["y_true"], files[0]["mask"]
        for f in files[1:]:
            if not (np.array_equal(f["y_true"], truth, equal_nan=True) and np.array_equal(f["mask"], mask)):
                raise ValueError(f"{case}: the seeds of {recipe} disagree on the truth or the mask")
        pred = np.mean([f["y_pred"].astype(np.float64) for f in files], axis=0).astype(np.float32)
        np.savez_compressed(out / f"{case}.npz", y_true=truth, y_pred=pred, mask=mask)
    return out.name


def main() -> None:
    dev = fold_table("grouped")
    cases = sorted(dev.index)
    ensembles = [write_ensemble(recipe, seeds, cases) for recipe, seeds in RECIPES.items()]
    runs = [*RECIPES["s11_unet_tuned"], *RECIPES["s13_unet_tuned_e200"], *ensembles]
    per_case = pd.concat([score_run(run, dev).assign(split="grouped") for run in runs], ignore_index=True)
    counts = per_case.groupby("run")["case"].nunique()
    if (counts != len(dev)).any():
        raise SystemExit(f"every run needs all {len(dev)} development cases: {counts.to_dict()}")
    summary = summarise(per_case)
    new, ref = ensemble_name("s13_unet_tuned_e200"), ensemble_name("s11_unet_tuned")
    comparison = bootstrap_vs_s07(per_case, new, ref=ref).assign(ref_run=ref)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "final_summary.csv", index=False, float_format="%.4f")
    comparison.to_csv(OUT / "final_vs_s11.csv", index=False, float_format="%.4f")
    cols = ["variant", "subset", "pooled_r2", "median_within_case_r2", "mean_mae", "mean_delta_p95"]
    with pd.option_context("display.width", 200, "display.precision", 4):
        print(summary[summary["subset"].isin(["all", "realistic"])][cols].to_string(index=False), "\n")
        print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
