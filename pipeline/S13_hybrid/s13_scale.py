"""
S13: at what length scale is the remaining error?

The networks predict every open-ground cell. Here the truth and the prediction are both averaged over
square blocks of 1, 2, 4, 8 and 16 m (only blocks that are at least half open ground count), and the
pooled R² is recomputed over the blocks. If the error were fine-scale detail or simulation noise, the
R² would rise quickly with the block size; if it lies in the street-scale pattern, it would barely move.
Idealized cases are stored at 0.5 m and realistic ones at 1 m, so a block holds twice as many cells
per side for the idealized cases. The 1 m row is therefore not the headline pooled R², whose idealized
cells are 0.5 m.

Writes outputs/scale_test.csv.

Usage: uv run python -m pipeline.S13_hybrid.s13_scale [run ...]  (default: the S11 tuned seed 0 and the 200-epoch U-Net)
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

from urbanwind.dl.train import PREDS, fold_table

OUT = pathlib.Path(__file__).resolve().parent / "outputs"
BLOCKS_M = (1, 2, 4, 8, 16)
CELL_M = {"idealized": 0.5, "realistic": 1.0}
MIN_OPEN = 0.5  # a block counts if at least half of it is open ground
SUBSETS = ("idealized", "realistic")


def block_means(truth: np.ndarray, pred: np.ndarray, open_: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Open-ground means of truth and prediction over k x k blocks (the domain is cropped to a multiple of k)."""
    ny, nx = (truth.shape[0] // k) * k, (truth.shape[1] // k) * k

    def sums(a: np.ndarray) -> np.ndarray:
        return a[:ny, :nx].reshape(ny // k, k, nx // k, k).sum(axis=(1, 3))

    n = sums(open_)
    keep = n >= MIN_OPEN * k * k
    return sums(truth * open_)[keep] / n[keep], sums(pred * open_)[keep] / n[keep]


def r2_from_sums(s: dict) -> float:
    return float(1 - s["sse"] / (s["sum_t2"] - s["sum_t"] ** 2 / s["n"]))


def scale_test(run: str, dev: pd.DataFrame) -> list[dict]:
    sums = {(b, sub): {"sum_t": 0.0, "sum_t2": 0.0, "sse": 0.0, "n": 0} for b in BLOCKS_M for sub in SUBSETS}
    for case in sorted(dev.index):
        sub = dev.loc[case, "subset"]
        with np.load(PREDS / run / f"{case}.npz") as f:
            truth = np.nan_to_num(f["y_true"]).astype(np.float64)
            pred = np.nan_to_num(f["y_pred"]).astype(np.float64)
            open_ = (~f["mask"]).astype(np.float64)
        for b in BLOCKS_M:
            t, p = block_means(truth, pred, open_, max(int(round(b / CELL_M[sub])), 1))
            s = sums[(b, sub)]
            s["sum_t"] += t.sum()
            s["sum_t2"] += (t * t).sum()
            s["sse"] += ((p - t) ** 2).sum()
            s["n"] += t.size
    rows = []
    for b in BLOCKS_M:
        both = {k: sums[(b, "idealized")][k] + sums[(b, "realistic")][k] for k in ("sum_t", "sum_t2", "sse", "n")}
        rows.append({"run": run, "block_m": b, "pooled_r2_all": r2_from_sums(both),
                     "pooled_r2_idealized": r2_from_sums(sums[(b, "idealized")]),
                     "pooled_r2_realistic": r2_from_sums(sums[(b, "realistic")]),
                     "n_blocks_realistic": sums[(b, "realistic")]["n"]})
    return rows


def main(runs: list[str]) -> None:
    dev = fold_table("grouped")
    table = pd.DataFrame([row for run in runs for row in scale_test(run, dev)])
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "scale_test.csv", index=False, float_format="%.4f")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:] or ["s11_unet_tuned_seed0", "s13_unet_tuned_e200_seed0"])
