"""S14: Uncertainty quantification and conformal calibration for urban wind maps.

Core functions
--------------
- S13UNet            : UNet architecture matching the S13 checkpoint layout
- load_s13_checkpoint: Load an S13 checkpoint into S13UNet
- deep_ensemble_predict   : mean + std across ensemble seed predictions
- mc_dropout_predict      : 30 stochastic passes with dropout active
- group_conformal_calibrate: split-conformal threshold (90% interval) from OOF errors
- compute_coverage        : overall / per-case / per-bin coverage
- reliability_bins        : (confidence, empirical_coverage) pairs for the reliability diagram
- compute_spearman        : rank correlation between uncertainty and |error|
- selective_prediction_error: MAE after dropping the most uncertain cells
- plot_reliability_diagram: save a calibration plot to disk

Architecture note (S13)
-----------------------
The S13 checkpoints were trained by Mahir with a UNet whose layer names differ from
the `nets.UNet` class (which is the S11 baseline).  S13 weights use:
  down.{0-3}.{0,3}: Conv2d  (circular padding)
  down.{0-3}.{1,4}: GroupNorm
  mid.{0,3}: Conv2d  /  mid.{1,4}: GroupNorm
  up_conv.{0-3}: Conv2d  (transposes channels: 384→192→96→48→24)
  up_block.{0-3}.{0,3}: Conv2d  /  up_block.{0-3}.{1,4}: GroupNorm
  head: 1×1 Conv2d → Softplus

Input channels (5, from S10): H/16, building_mask, dist_to_wall/16, cos(angle), sin(angle)
Widths: [24, 48, 96, 192], bottleneck 384, dropout 0.08.

GroupNorm: groups = 8 throughout (divides all widths and 384).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# S13 UNet — matches the checkpoint key layout exactly
# ---------------------------------------------------------------------------

def _conv_block(in_ch: int, out_ch: int, groups: int = 8, dropout: float = 0.0) -> nn.Sequential:
    """Two Conv3×3 (circular) + GroupNorm + ReLU, with optional spatial dropout."""
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="circular"),
        nn.GroupNorm(groups, out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="circular"),
        nn.GroupNorm(groups, out_ch),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout2d(p=dropout))
    return nn.Sequential(*layers)


class S13UNet(nn.Module):
    """UNet whose state-dict layout matches the S13 / S11 Kaggle checkpoints.

    Layer names:  down.{0-3}, mid, up_conv.{0-3}, up_block.{0-3}, head.
    Dropout (p=0.08) is appended to each down and mid conv-block so the
    state-dict index still matches the trained weights (dropout has no
    learnable parameters, so the conv/GN indices are unaffected).
    """

    def __init__(
        self,
        in_ch: int = 5,
        widths: tuple[int, ...] = (24, 48, 96, 192),
        bottleneck: int = 384,
        gn_groups: int = 8,
        dropout: float = 0.08,
    ):
        super().__init__()
        w = widths

        # Encoder — four levels, each a conv-block followed by 2× max-pool
        self.down = nn.ModuleList([
            _conv_block(in_ch,  w[0], gn_groups, dropout),
            _conv_block(w[0],   w[1], gn_groups, dropout),
            _conv_block(w[1],   w[2], gn_groups, dropout),
            _conv_block(w[2],   w[3], gn_groups, dropout),
        ])
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.mid = _conv_block(w[3], bottleneck, gn_groups, dropout)

        # Decoder — 3×3 convolutions (circular) halve channels before the skip concat
        self.up_conv = nn.ModuleList([
            nn.Conv2d(bottleneck, w[3], 3, padding=1, padding_mode="circular"),
            nn.Conv2d(w[3],       w[2], 3, padding=1, padding_mode="circular"),
            nn.Conv2d(w[2],       w[1], 3, padding=1, padding_mode="circular"),
            nn.Conv2d(w[1],       w[0], 3, padding=1, padding_mode="circular"),
        ])
        # After concat(up_conv_out, skip): 2× channels → out channels
        self.up_block = nn.ModuleList([
            _conv_block(w[3] * 2, w[3], gn_groups),
            _conv_block(w[2] * 2, w[2], gn_groups),
            _conv_block(w[1] * 2, w[1], gn_groups),
            _conv_block(w[0] * 2, w[0], gn_groups),
        ])

        self.head = nn.Conv2d(w[0], 1, 1)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skips: list[torch.Tensor] = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.mid(x)

        # Decoder
        for i, (up, block) in enumerate(zip(self.up_conv, self.up_block)):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = up(x)
            skip = skips[-(i + 1)]
            # Pad if needed (domain sizes that are not multiples of 16)
            dh = skip.shape[2] - x.shape[2]
            dw = skip.shape[3] - x.shape[3]
            if dh or dw:
                x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
            x = torch.cat([skip, x], dim=1)
            x = block(x)

        return self.softplus(self.head(x))


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_s13_checkpoint(
    ckpt_path: str | object,
    in_ch: int = 5,
    widths: tuple[int, ...] = (24, 48, 96, 192),
    bottleneck: int = 384,
    dropout: float = 0.08,
    device: str = "cpu",
) -> S13UNet:
    """Load a saved S13 checkpoint into S13UNet and return it in eval mode."""
    model = S13UNet(in_ch=in_ch, widths=widths, bottleneck=bottleneck, dropout=dropout)
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Legacy loader kept for backward-compatibility with the existing stub
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path, widths=(24, 48, 96, 192), bottleneck=384, device="cpu"):
    """Alias: load an S13 checkpoint (5-channel S13UNet)."""
    return load_s13_checkpoint(ckpt_path, widths=widths, bottleneck=bottleneck, device=device)


# ---------------------------------------------------------------------------
# Ensemble and MC-dropout prediction
# ---------------------------------------------------------------------------

def deep_ensemble_predict(
    ckpt_paths: list,
    input_data: torch.Tensor,
    device: str = "cpu",
    in_ch: int = 5,
    widths: tuple[int, ...] = (24, 48, 96, 192),
    bottleneck: int = 384,
    dropout: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """Run all ensemble checkpoints and return per-cell mean and std.

    Parameters
    ----------
    ckpt_paths:
        Paths to saved S13 checkpoints (one per seed × fold).
    input_data:
        Tensor of shape [B, C, H, W].
    device:
        'cpu' or 'cuda'.
    in_ch:
        Number of input channels (5 for S13).
    widths, bottleneck, dropout:
        Architecture hyper-parameters; must match the checkpoint.

    Returns
    -------
    mean: ndarray [B, 1, H, W]
    std : ndarray [B, 1, H, W]
    """
    preds: list[np.ndarray] = []
    for ckpt in ckpt_paths:
        model = load_s13_checkpoint(ckpt, in_ch=in_ch, widths=widths, bottleneck=bottleneck, dropout=dropout, device=device)
        x = input_data.to(device) if isinstance(input_data, torch.Tensor) else torch.as_tensor(input_data).to(device)
        with torch.no_grad():
            pred = model(x).cpu().numpy()
        preds.append(pred)
    stack = np.stack(preds, axis=0)          # [n_members, B, 1, H, W]
    return stack.mean(axis=0), stack.std(axis=0)


def mc_dropout_predict(
    model: nn.Module,
    input_data: torch.Tensor,
    n_passes: int = 30,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """MC dropout: n_passes forward passes with dropout layers active.

    Only Dropout2d layers switch to training mode; GroupNorm layers keep no
    running statistics, so their output is unchanged.

    Parameters
    ----------
    model:
        A loaded S13UNet (or any model with Dropout2d).
    input_data:
        Tensor [B, C, H, W].
    n_passes:
        Number of stochastic forward passes (30 is the S14 default).
    device:
        Torch device string.

    Returns
    -------
    mean: ndarray [B, 1, H, W]
    std : ndarray [B, 1, H, W]
    """
    model.to(device)
    x = input_data.to(device) if isinstance(input_data, torch.Tensor) else torch.as_tensor(input_data).to(device)

    # Set the whole model to eval, then flip only Dropout layers to train mode.
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

    preds: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(n_passes):
            pred = model(x).cpu().numpy()
            preds.append(pred)

    model.eval()   # restore
    stack = np.stack(preds, axis=0)
    return stack.mean(axis=0), stack.std(axis=0)


# ---------------------------------------------------------------------------
# Split-conformal calibration
# ---------------------------------------------------------------------------

def group_conformal_calibrate(oof_scores: np.ndarray, alpha: float = 0.1) -> float:
    """Return the (1−α) quantile of |calibration errors|.

    Call once per fold k, passing the absolute errors from ALL OTHER folds.
    The returned threshold is the half-width of the symmetric conformal interval:
    prediction interval = [ŷ − q, ŷ + q].

    Parameters
    ----------
    oof_scores:
        1-D array of residuals (y_true − y_pred) or absolute errors from the
        calibration folds.  The sign is ignored.
    alpha:
        Miscoverage level.  alpha=0.1 → 90% intervals.

    Returns
    -------
    threshold: float
    """
    errors = np.abs(np.asarray(oof_scores, dtype=float))
    n = len(errors)
    if n == 0:
        return float("inf")
    # Finite-sample correction: ceil((n+1)(1−α)) / n
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(errors, level))


def compute_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    mask: np.ndarray | None = None,
) -> dict:
    """Measure coverage of the symmetric conformal interval [ŷ−q, ŷ+q].

    Parameters
    ----------
    y_true, y_pred:
        Arrays of true and predicted values (any shape, flattened internally).
    threshold:
        Conformal half-width q.
    mask:
        Boolean array; True = cell is *excluded* (building / invalid).
        If None, all cells are included.

    Returns
    -------
    dict with keys: coverage (float), n_cells (int), n_covered (int)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if mask is not None:
        valid = ~np.asarray(mask).ravel()
    else:
        valid = np.ones(len(y_true), dtype=bool)
    valid &= np.isfinite(y_true) & np.isfinite(y_pred)
    errors = np.abs(y_true[valid] - y_pred[valid])
    covered = errors <= threshold
    n_cells = int(valid.sum())
    n_covered = int(covered.sum())
    coverage = n_covered / n_cells if n_cells > 0 else float("nan")
    return {"coverage": coverage, "n_cells": n_cells, "n_covered": n_covered}


def per_case_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Coverage per case (expects a 2-D array [n_cases, n_cells] or a list of arrays)."""
    coverages = []
    for i in range(len(y_true)):
        m = mask[i] if mask is not None else None
        c = compute_coverage(y_true[i], y_pred[i], threshold, m)
        coverages.append(c["coverage"])
    return np.array(coverages)


def reliability_bins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainties: np.ndarray,
    mask: np.ndarray | None = None,
    n_bins: int = 10,
    alpha_grid: np.ndarray | None = None,
) -> list[tuple[float, float]]:
    """Compute (expected_coverage, empirical_coverage) pairs for the reliability diagram.

    For each alpha level in alpha_grid, compute the conformal threshold from the
    uncertainties and measure the fraction of cells actually covered.

    Parameters
    ----------
    y_true, y_pred, uncertainties:
        Flat arrays of true values, predictions, and per-cell uncertainty (std).
    mask:
        Boolean mask; True = excluded.
    n_bins:
        Number of confidence levels (ignored when alpha_grid is supplied).
    alpha_grid:
        Array of alpha values to sweep (default: linspace from 0.05 to 0.5).

    Returns
    -------
    List of (confidence_level, empirical_coverage) tuples.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    unc = np.asarray(uncertainties).ravel()

    if mask is not None:
        valid = ~np.asarray(mask).ravel()
    else:
        valid = np.ones(len(y_true), dtype=bool)
    valid &= np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(unc)

    errors = np.abs(y_true[valid] - y_pred[valid])

    if alpha_grid is None:
        alpha_grid = np.linspace(0.05, 0.50, n_bins)

    pairs = []
    for alpha in alpha_grid:
        threshold = group_conformal_calibrate(errors, alpha=alpha)
        cov = (errors <= threshold).mean()
        pairs.append((float(1 - alpha), float(cov)))
    return pairs


# ---------------------------------------------------------------------------
# Spearman correlation
# ---------------------------------------------------------------------------

def compute_spearman(uncertainty: np.ndarray, absolute_error: np.ndarray) -> float:
    """Rank correlation between per-cell uncertainty and |error|.

    Uses a fast rank-based Pearson formula (equivalent to scipy's spearmanr
    but avoids the scipy dependency for this small utility).

    Parameters
    ----------
    uncertainty:
        Per-cell uncertainty (e.g. ensemble std or MC-dropout std).
    absolute_error:
        Per-cell |y_true − y_pred|.

    Returns
    -------
    float: Spearman ρ, NaN-safe.
    """
    u = np.asarray(uncertainty).ravel().astype(float)
    e = np.asarray(absolute_error).ravel().astype(float)
    valid = np.isfinite(u) & np.isfinite(e)
    if valid.sum() < 3:
        return float("nan")
    u, e = u[valid], e[valid]
    # Rank both arrays
    rank_u = np.argsort(np.argsort(u)).astype(float)
    rank_e = np.argsort(np.argsort(e)).astype(float)
    # Pearson on ranks
    ru, re = rank_u - rank_u.mean(), rank_e - rank_e.mean()
    denom = np.sqrt((ru ** 2).sum() * (re ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((ru * re).sum() / denom)


# ---------------------------------------------------------------------------
# Selective prediction
# ---------------------------------------------------------------------------

def selective_prediction_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    mask: np.ndarray | None = None,
    drop_pct: float = 0.10,
) -> float:
    """MAE after dropping the most uncertain cells.

    Parameters
    ----------
    y_true, y_pred:
        True and predicted wind speeds.
    uncertainty:
        Per-cell uncertainty (std).
    mask:
        Boolean; True = cell excluded (building / invalid).  None → keep all.
    drop_pct:
        Fraction of *valid* cells to drop (sorted by decreasing uncertainty).

    Returns
    -------
    Mean absolute error on the retained cells.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    unc = np.asarray(uncertainty).ravel()

    if mask is not None:
        valid = ~np.asarray(mask).ravel()
    else:
        valid = np.ones(len(y_true), dtype=bool)
    valid &= np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(unc)

    errors = np.abs(y_true[valid] - y_pred[valid])
    flat_unc = unc[valid]

    n_drop = int(len(flat_unc) * drop_pct)
    # Sort ascending; drop the last n_drop (highest uncertainty)
    keep_idx = np.argsort(flat_unc)[: len(flat_unc) - n_drop]
    return float(errors[keep_idx].mean()) if len(keep_idx) > 0 else float("nan")


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    coverage_by_bin: list[tuple[float, float]],
    output_path: str,
    title: str = "Group conformal calibration — reliability diagram",
) -> None:
    """Save a reliability diagram to *output_path*.

    Parameters
    ----------
    coverage_by_bin:
        List of (expected_coverage, empirical_coverage) pairs.
    output_path:
        File path for the saved figure (.png recommended).
    title:
        Figure title.
    """
    import matplotlib.pyplot as plt

    confs = [b[0] for b in coverage_by_bin]
    covs = [b[1] for b in coverage_by_bin]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(confs, covs, "o-", label="Empirical coverage", color="#4c72b0")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.fill_between([0, 1], [0, 0], [1, 1], alpha=0.05, color="grey")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Expected coverage (1 − α)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
