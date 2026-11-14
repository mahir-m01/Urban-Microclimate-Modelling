"""S14: End-to-end pipeline for Deep Ensemble, MC Dropout, and Conformal Calibration.

Usage:
    python pipeline/S14_dl_uncertainty/run_s14_pipeline.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from urbanwind.dl.uq import (
    load_s13_checkpoint,
    group_conformal_calibrate,
    compute_spearman,
    selective_prediction_error,
)

REPO = Path(__file__).resolve().parents[2]
CHECKPOINTS_BASE = Path(os.environ.get("S13_CKPT_DIR", REPO / "checkpoints" / "s13"))
CACHE_DIR = Path(os.environ.get("S10_CACHE_DIR", REPO / "data" / "s10_cache"))
PREDS_BASE = REPO / "results" / "preds"
FIG_DIR = REPO / "pipeline" / "S14_dl_uncertainty" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def run_mc_dropout_passes(model: nn.Module, x_tensor: torch.Tensor, n_passes: int = 30) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()
    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            pred = model(x_tensor)
            preds.append(pred.squeeze(0).squeeze(0).cpu().numpy())
    model.eval()
    stack = np.stack(preds, axis=0)
    return stack.mean(axis=0), stack.std(axis=0)


def undo_transform(field_2d: np.ndarray, k: int, transposed: bool, dx_native: float) -> np.ndarray:
    arr = field_2d.T if transposed else field_2d
    arr = np.rot90(arr, -int(k))
    if dx_native == 0.5:
        arr = np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1)
    return arr


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[S14] Running inference on device: {device}")

    # Check for weights and cache
    if not CHECKPOINTS_BASE.exists() or not CACHE_DIR.exists():
        print(f"[S14] Note: Checkpoints ({CHECKPOINTS_BASE}) or Cache ({CACHE_DIR}) not found locally.")
        print("[S14] This script runs in the Kaggle GPU environment where both datasets are mounted.")
        return

    # Load 15 checkpoints
    models = {seed: {} for seed in (0, 1, 2)}
    for seed in (0, 1, 2):
        for fold in range(5):
            ckpt = CHECKPOINTS_BASE / f"s13_unet_tuned_e200_seed{seed}" / f"fold{fold}.pt"
            models[seed][fold] = load_s13_checkpoint(str(ckpt), device=device)
    print("[S14] Loaded 15 checkpoints into memory.")

    mc_results, ens_results = {}, {}
    case_files = sorted(list(CACHE_DIR.glob("*.npz")))
    print(f"[S14] Processing {len(case_files)} cases...")

    t0 = time.time()
    for f in tqdm(case_files):
        c_name = f.stem
        d = np.load(str(f))
        fold = int(d["fold"])
        H = d["H"].astype(np.float32)
        dist = d["dist"].astype(np.float32)
        angle_deg = float(d["angle_deg"])
        k = int(d["k"])
        transposed = bool(d["transposed"])
        dx = float(d["dx_native"])
        u_native = d["U_native"].astype(np.float32)

        x = np.stack([
            H / 16.0,
            (H > 0.0).astype(np.float32),
            dist / 16.0,
            np.full_like(H, np.cos(np.radians(angle_deg))),
            np.full_like(H, np.sin(np.radians(angle_deg))),
        ], axis=0)

        h_orig, w_orig = H.shape
        pad_h = (16 - (h_orig % 16)) % 16
        pad_w = (16 - (w_orig % 16)) % 16
        x_pad = np.pad(x, ((0, 0), (0, pad_h), (0, pad_w)), mode="wrap") if (pad_h or pad_w) else x
        x_t = torch.from_numpy(x_pad).unsqueeze(0).to(device)

        # MC Dropout (Seed 0)
        m0 = models[0][fold]
        mc_mean_p, mc_std_p = run_mc_dropout_passes(m0, x_t, n_passes=30)
        mc_mean = undo_transform(mc_mean_p[:h_orig, :w_orig], k, transposed, dx)
        mc_std = undo_transform(mc_std_p[:h_orig, :w_orig], k, transposed, dx)

        # Ensemble (Seeds 0, 1, 2)
        with torch.no_grad():
            p0 = m0(x_t).squeeze().cpu().numpy()[:h_orig, :w_orig]
            p1 = models[1][fold](x_t).squeeze().cpu().numpy()[:h_orig, :w_orig]
            p2 = models[2][fold](x_t).squeeze().cpu().numpy()[:h_orig, :w_orig]
        stack = np.stack([p0, p1, p2], axis=0)
        ens_mean = undo_transform(stack.mean(axis=0), k, transposed, dx)
        ens_std = undo_transform(stack.std(axis=0), k, transposed, dx)

        mask = np.isnan(u_native)
        mc_results[c_name] = {"fold": fold, "y_true": u_native, "y_pred": mc_mean, "unc": mc_std, "mask": mask, "subset": str(d["subset"])}
        ens_results[c_name] = {"fold": fold, "y_true": u_native, "y_pred": ens_mean, "unc": ens_std, "mask": mask, "subset": str(d["subset"])}

    print(f"[S14] Completed inference in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
