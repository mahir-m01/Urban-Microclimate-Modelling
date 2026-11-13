"""
test_losses.py: hand-checked tests for candidate training losses (S12b).

Every test runs in its own process on macOS and checks:
1. Exact hand-calculated numbers on a 4x4 tensor with building cells.
2. Zero loss when pred == y.
3. Zero loss (and no NaN) when no cells are valid (mask == 0).
4. Normalisation by the number of valid open-ground cells.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch
from urbanwind.dl.losses import (
    LOSSES,
    masked_fft_l1,
    masked_l1,
    masked_mean_consistency_l1,
    masked_sobel_l1,
    masked_tail_l1_gamma1,
    masked_tail_l1_gamma2,
)


@pytest.fixture
def example_4x4():
    """A 4x4 test patch with 3 building cells (13 valid cells)."""
    y = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0],
        [4.0, 5.0, 6.0, 7.0],
    ], dtype=torch.float32).view(1, 1, 4, 4)

    pred = torch.tensor([
        [1.5, 2.0, 2.5, 4.0],
        [2.0, 3.5, 4.0, 5.5],
        [3.0, 4.0, 5.0, 6.0],
        [4.5, 5.0, 5.5, 7.0],
    ], dtype=torch.float32).view(1, 1, 4, 4)

    # 3 building cells: (0, 0), (1, 2), (3, 3) -> 13 valid cells
    mask = torch.tensor([
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0],
    ], dtype=torch.float32).view(1, 1, 4, 4)

    return pred, y, mask


def test_masked_l1_hand_checked(example_4x4):
    pred, y, mask = example_4x4
    # Valid errors: 0.5, 0.5, 0.5, 0.5, 0.5 -> sum = 2.5 over 13 cells
    expected = 2.5 / 13.0
    loss = masked_l1(pred, y, mask)
    assert loss.item() == pytest.approx(expected, rel=1e-5)


def test_masked_mean_consistency_hand_checked(example_4x4):
    pred, y, mask = example_4x4
    # y valid sum = 52, mean = 4.0
    # pred valid sum = 52.5, mean = 52.5 / 13
    # |pred_mean - y_mean| = 0.5 / 13
    # Total loss = L1 (2.5/13) + 1.0 * (0.5/13) = 3.0 / 13.0
    expected = 3.0 / 13.0
    loss = masked_mean_consistency_l1(pred, y, mask, alpha=1.0)
    assert loss.item() == pytest.approx(expected, rel=1e-5)


def test_masked_tail_l1_hand_checked(example_4x4):
    pred, y, mask = example_4x4
    # U_mean = 52 / 13 = 4.0
    # gamma = 1: weights w = y / 4.0
    # errors * weights:
    # (0, 2): 0.5 * 3/4 = 0.375
    # (1, 1): 0.5 * 3/4 = 0.375
    # (1, 3): 0.5 * 5/4 = 0.625
    # (3, 0): 0.5 * 4/4 = 0.500
    # (3, 2): 0.5 * 6/4 = 0.750
    # sum = 2.625 -> loss = 2.625 / 13
    expected_gamma1 = 2.625 / 13.0
    loss_g1 = masked_tail_l1_gamma1(pred, y, mask)
    assert loss_g1.item() == pytest.approx(expected_gamma1, rel=1e-5)

    # gamma = 2: weights w = (y / 4.0)^2
    # (0, 2): 0.5 * (3/4)^2 = 0.28125
    # (1, 1): 0.5 * (3/4)^2 = 0.28125
    # (1, 3): 0.5 * (5/4)^2 = 0.78125
    # (3, 0): 0.5 * (4/4)^2 = 0.50000
    # (3, 2): 0.5 * (6/4)^2 = 1.12500
    # sum = 2.96875 -> loss = 2.96875 / 13
    expected_gamma2 = 2.96875 / 13.0
    loss_g2 = masked_tail_l1_gamma2(pred, y, mask)
    assert loss_g2.item() == pytest.approx(expected_gamma2, rel=1e-5)


def test_masked_sobel_l1(example_4x4):
    pred, y, mask = example_4x4
    # Identical predictions should have zero gradient loss
    perfect = masked_sobel_l1(y, y, mask, alpha=1.0)
    assert perfect.item() == pytest.approx(0.0, abs=1e-6)

    loss = masked_sobel_l1(pred, y, mask, alpha=1.0)
    assert loss.item() > masked_l1(pred, y, mask).item()
    assert torch.isfinite(loss)


def test_masked_fft_l1(example_4x4):
    pred, y, mask = example_4x4
    # Identical predictions should have zero FFT loss
    perfect = masked_fft_l1(y, y, mask, alpha=1.0)
    assert perfect.item() == pytest.approx(0.0, abs=1e-6)

    loss = masked_fft_l1(pred, y, mask, alpha=1.0)
    assert loss.item() > masked_l1(pred, y, mask).item()
    assert torch.isfinite(loss)


def test_empty_mask_returns_zero_and_no_nan():
    """A patch with no valid cells must return 0.0, not NaN."""
    pred = torch.randn(2, 1, 16, 16)
    y = torch.randn(2, 1, 16, 16)
    mask = torch.zeros(2, 1, 16, 16)

    for name, fn in LOSSES.items():
        loss = fn(pred, y, mask)
        assert torch.isfinite(loss), f"{name} returned non-finite on empty mask"
        assert loss.item() == 0.0, f"{name} did not return 0.0 on empty mask"


def test_losses_registry_coverage():
    """All required loss keys from S11, S12b, and S15 are registered."""
    required = ["l1", "mse", "gradient", "sobel", "tail_w1", "tail_w2", "fft", "mean_consistency"]
    for key in required:
        assert key in LOSSES, f"Missing loss key: {key}"


def test_losses_optimize_and_stay_finite():
    """Smoke optimization test: each candidate loss trains a model, decreases, and stays finite."""
    torch.manual_seed(42)
    # 2 small patches with random buildings
    B, H, W = 2, 16, 16
    x = torch.randn(B, 4, H, W)
    y = torch.abs(torch.randn(B, 1, H, W)) + 0.5
    mask = (torch.rand(B, 1, H, W) > 0.25).float()  # 75% open ground

    losses_to_test = [
        "l1",
        "sobel",
        "tail_w1",
        "tail_w2",
        "fft",
        "mean_consistency",
    ]

    for loss_name in losses_to_test:
        loss_fn = LOSSES[loss_name]
        # Simple convolutional predictor
        model = torch.nn.Sequential(
            torch.nn.Conv2d(4, 16, 3, padding=1, padding_mode="circular"),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 1, 1),
            torch.nn.Softplus(),
        )
        opt = torch.optim.Adam(model.parameters(), lr=0.05)

        initial_loss = None
        final_loss = None
        for step in range(35):
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y, mask)
            assert torch.isfinite(loss), f"{loss_name} produced non-finite loss at step {step}"
            loss.backward()
            opt.step()

            if step == 0:
                initial_loss = loss.item()
            if step == 34:
                final_loss = loss.item()

        assert final_loss < initial_loss, (
            f"{loss_name} did not decrease after optimization: initial={initial_loss}, final={final_loss}"
        )

