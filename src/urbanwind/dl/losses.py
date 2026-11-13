"""
losses.py: training losses. Every loss has the signature ``loss(pred, y, mask) -> scalar tensor``.

The mean is taken over the valid cells of the whole batch, so a patch full of buildings does not
count less per cell than an open one, and a batch without any valid cell gives 0 instead of NaN.
Losses are computed in float32 even when the network runs in half precision.

Public API
----------
masked_l1(pred, y, mask)                   : mean absolute error over valid cells
masked_mse(pred, y, mask)                  : mean squared error over valid cells
masked_sobel_l1(pred, y, mask)             : Sobel-gradient term added to masked L1 (circular padding)
masked_tail_l1(pred, y, mask, gamma, clip) : tail-weighted L1 emphasizing peak wind speeds
masked_tail_l1_gamma1(pred, y, mask)       : tail-weighted L1 with gamma=1
masked_tail_l1_gamma2(pred, y, mask)       : tail-weighted L1 with gamma=2
masked_fft_l1(pred, y, mask)               : FFT-magnitude term added to masked L1
masked_mean_consistency_l1(pred, y, mask)  : per-case/patch mean-consistency term added to masked L1
LOSSES                                     : name -> loss, for configs
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Normalized 3x3 Sobel kernels
SOBEL_Y = torch.tensor([[-1.0, -2.0, -1.0],
                        [ 0.0,  0.0,  0.0],
                        [ 1.0,  2.0,  1.0]], dtype=torch.float32) / 8.0

SOBEL_X = torch.tensor([[-1.0,  0.0,  1.0],
                        [-2.0,  0.0,  2.0],
                        [-1.0,  0.0,  1.0]], dtype=torch.float32) / 8.0


def masked_l1(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over valid open-ground cells."""
    mask = mask.float()
    error = (pred.float() - y.float()).abs() * mask
    return error.sum() / mask.sum().clamp(min=1.0)


def masked_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over valid open-ground cells."""
    mask = mask.float()
    error = (pred.float() - y.float()) ** 2 * mask
    return error.sum() / mask.sum().clamp(min=1.0)


def masked_sobel_l1(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
                    alpha: float = 1.0) -> torch.Tensor:
    """Sobel-gradient term added to masked L1, using circular padding.

    Gradients are counted only where both neighbouring cells across the difference axis
    are valid open ground.
    """
    m = (mask > 0).float()
    l1 = masked_l1(pred, y, m)
    if m.sum() == 0:
        return l1

    # Ensure 4D shape [B, 1, H, W] for convolution
    p4 = pred.float() * m
    y4 = y.float() * m
    m4 = m
    while p4.ndim < 4:
        p4 = p4.unsqueeze(0)
        y4 = y4.unsqueeze(0)
        m4 = m4.unsqueeze(0)

    device = pred.device
    ky = SOBEL_Y.view(1, 1, 3, 3).to(device)
    kx = SOBEL_X.view(1, 1, 3, 3).to(device)

    # Circular padding for periodic domain boundary
    p_pad = F.pad(p4, (1, 1, 1, 1), mode="circular")
    y_pad = F.pad(y4, (1, 1, 1, 1), mode="circular")

    gy_p = F.conv2d(p_pad, ky)
    gy_y = F.conv2d(y_pad, ky)
    gx_p = F.conv2d(p_pad, kx)
    gx_y = F.conv2d(y_pad, kx)

    # Both neighbours along Y (top and bottom) must be valid
    m_top = torch.roll(m4, shifts=1, dims=-2)
    m_bot = torch.roll(m4, shifts=-1, dims=-2)
    valid_y = m4 * m_top * m_bot

    # Both neighbours along X (left and right) must be valid
    m_left = torch.roll(m4, shifts=1, dims=-1)
    m_right = torch.roll(m4, shifts=-1, dims=-1)
    valid_x = m4 * m_left * m_right

    err_y = (gy_p - gy_y).abs() * valid_y
    err_x = (gx_p - gx_y).abs() * valid_x

    n_grad = valid_y.sum() + valid_x.sum()
    if n_grad == 0:
        return l1

    grad_loss = (err_y.sum() + err_x.sum()) / n_grad
    return l1 + alpha * grad_loss


def masked_tail_l1(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
                   gamma: float = 1.0, clip: float = 5.0) -> torch.Tensor:
    """Tail-weighted L1 with weights w = (U / U_mean)^gamma, clipped to prevent outlier instability."""
    m = (mask > 0).float()
    if m.sum() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=torch.float32)

    # Mean speed of valid open-ground cells per item
    dims = tuple(range(1, y.ndim)) if y.ndim > 1 else (0,)
    y_valid = y.float() * m
    valid_count = m.sum(dim=dims, keepdim=True)
    u_mean = (y_valid.sum(dim=dims, keepdim=True) / valid_count.clamp(min=1.0)).clamp(min=1e-4)

    ratio = y.float() / u_mean
    weights = torch.clamp(ratio, min=0.0, max=clip) ** gamma
    error = (pred.float() - y.float()).abs() * weights * m
    return error.sum() / m.sum().clamp(min=1.0)


def masked_tail_l1_gamma1(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Tail-weighted L1 with gamma=1."""
    return masked_tail_l1(pred, y, mask, gamma=1.0)


def masked_tail_l1_gamma2(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Tail-weighted L1 with gamma=2."""
    return masked_tail_l1(pred, y, mask, gamma=2.0)


def masked_fft_l1(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
                  alpha: float = 1.0) -> torch.Tensor:
    """FFT-magnitude term added to masked L1.

    Building cells are zeroed before transform to provide a full rectangular field.
    """
    m = (mask > 0).float()
    l1 = masked_l1(pred, y, m)
    if m.sum() == 0:
        return l1

    p_clean = pred.float() * m
    y_clean = y.float() * m

    fft_p = torch.fft.rfft2(p_clean, dim=(-2, -1), norm="ortho")
    fft_y = torch.fft.rfft2(y_clean, dim=(-2, -1), norm="ortho")

    mag_diff = (fft_p.abs() - fft_y.abs()).abs()
    fft_term = mag_diff.sum() / m.sum().clamp(min=1.0)
    return l1 + alpha * fft_term


def masked_mean_consistency_l1(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
                               alpha: float = 1.0) -> torch.Tensor:
    """Per-case/patch mean-consistency term added to masked L1."""
    m = (mask > 0).float()
    l1 = masked_l1(pred, y, m)
    if m.sum() == 0:
        return l1

    dims = tuple(range(1, y.ndim)) if y.ndim > 1 else (0,)
    valid_count = m.sum(dim=dims, keepdim=True)
    has_valid = (valid_count > 0).float()

    mean_p = (pred.float() * m).sum(dim=dims, keepdim=True) / valid_count.clamp(min=1.0)
    mean_y = (y.float() * m).sum(dim=dims, keepdim=True) / valid_count.clamp(min=1.0)

    mean_diff = (mean_p - mean_y).abs() * has_valid
    mean_term = (mean_diff * valid_count).sum() / m.sum().clamp(min=1.0)
    return l1 + alpha * mean_term


LOSSES = {
    "l1": masked_l1,
    "mse": masked_mse,
    "sobel": masked_sobel_l1,
    "gradient": masked_sobel_l1,
    "sobel_l1": masked_sobel_l1,
    "tail_w1": masked_tail_l1_gamma1,
    "tail_w2": masked_tail_l1_gamma2,
    "tail1": masked_tail_l1_gamma1,
    "tail2": masked_tail_l1_gamma2,
    "fft": masked_fft_l1,
    "spectral": masked_fft_l1,
    "fft_l1": masked_fft_l1,
    "mean_consistency": masked_mean_consistency_l1,
    "mean": masked_mean_consistency_l1,
    "mean_l1": masked_mean_consistency_l1,
}
