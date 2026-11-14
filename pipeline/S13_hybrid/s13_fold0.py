"""
S13 fold-0 screen: does a candidate beat the S11 tuned U-Net on fold 0's 97 cases?

Each candidate trains fold 0 with seed 0 (configs/), and its predictions are rebuilt from the
saved weights with `python -m urbanwind.dl.train --config <config> --seed 0 --folds 0 --repredict`.
The yardstick is the spread of the three S11 tuned seeds on the same cases: a candidate counts as
better only if it beats the best of them, and only then does it get all five folds.

Writes outputs/fold0_screen.csv.

Usage: uv run python -m pipeline.S13_hybrid.s13_fold0
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd

from pipeline.S11_unet_baseline.s11_results import RUNS as S11_RUNS
from pipeline.S11_unet_baseline.s11_results import pooled_r2, score_run
from urbanwind.dl.train import PREDS, fold_table

STAGE = pathlib.Path(__file__).resolve().parent
OUT, RUNS = STAGE / "outputs", STAGE / "outputs" / "runs"
S11_SEEDS = [f"s11_unet_tuned_seed{s}" for s in range(3)]


def candidates() -> list[str]:
    """S13 runs whose fold 0 has finished and whose fold-0 predictions exist."""
    return sorted(p.parent.name for p in RUNS.glob("*/fold0.json") if (PREDS / p.parent.name).is_dir())


def screen_row(run: str, scored: pd.DataFrame, record: dict) -> dict:
    sub = scored[scored["run"] == run]
    return {"run": run, "n_cases": len(sub), "pooled_r2": pooled_r2(sub), "median_case_r2": sub["r2"].median(),
            "median_case_r2_idealized": sub.loc[sub["subset"] == "idealized", "r2"].median(),
            "median_case_r2_realistic": sub.loc[sub["subset"] == "realistic", "r2"].median(),
            "realistic_pooled_r2": pooled_r2(sub[sub["subset"] == "realistic"]), "mean_mae": sub["mae"].mean(),
            "inner_val_rmse": record.get("best_val"), "epochs_run": record.get("epochs_run"),
            "best_epoch": record.get("best_epoch"), "train_minutes": record.get("minutes"),
            "device": record.get("device")}


def main() -> None:
    dev = fold_table("grouped")
    fold0 = dev.index[dev["fold"] == 0]
    runs = [*S11_SEEDS, *candidates()]
    scored = pd.concat([score_run(run, dev) for run in runs], ignore_index=True)
    scored = scored[scored["case"].isin(fold0)]
    rows = []
    for run in runs:
        folder = S11_RUNS if run in S11_SEEDS else RUNS
        rows.append(screen_row(run, scored, json.loads((folder / run / "fold0.json").read_text())))
    table = pd.DataFrame(rows)
    best = table[table["run"].isin(S11_SEEDS)][["pooled_r2", "median_case_r2"]].max()
    is_s11 = table["run"].isin(S11_SEEDS)
    for metric in ("pooled_r2", "median_case_r2"):  # left blank for the S11 seeds themselves
        table[f"beats_best_s11_seed_{metric}"] = (table[metric] > best[metric]).astype(object).where(~is_s11, "")
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "fold0_screen.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 220, "display.precision", 4):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
