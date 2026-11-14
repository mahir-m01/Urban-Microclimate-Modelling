"""Tests for the S13 ConvNeXt U-Net (urbanwind.dl.nets.ConvNeXtUNet)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # PyTorch has no wheels for Intel Macs; CI and Apple silicon have it

from urbanwind.dl.losses import masked_l1  # noqa: E402
from urbanwind.dl.nets import ConvNeXtUNet, LayerNorm2d, build_model  # noqa: E402


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def test_convnext_unet_keeps_the_grid_and_predicts_positive_speeds():
    model = build_model("convnext_unet", 5).eval()
    out = model(torch.randn(2, 5, 64, 96))
    assert out.shape == (2, 1, 64, 96) and (out > 0).all()
    with pytest.raises(ValueError):
        model(torch.randn(1, 5, 64, 72))  # 72 is not a multiple of 16


def test_convnext_unet_size_is_within_a_quarter_of_the_tuned_unet():
    unet = build_model("unet", 5, widths=[24, 48, 96, 192], bottleneck=384, dropout=0.08)
    ratio = n_params(build_model("convnext_unet", 5)) / n_params(unet)
    assert 0.75 < ratio < 1.25


def test_convnext_unet_is_periodic():
    # a shift by a multiple of 16 cells must shift the output by the same amount; any zero padding
    # or clamped upsampling would break this at the edges (float64, so rounding cannot hide it).
    # The layer scale is set to 1 so the ConvNeXt blocks actually contribute to the output.
    torch.manual_seed(0)
    model = ConvNeXtUNet(widths=(8, 16, 16, 32), bottleneck=32, depths=(1, 1, 1, 1), mid_depth=1).double().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("gamma"):
                p.fill_(1.0)
        x = torch.randn(1, 5, 64, 96, dtype=torch.float64)
        shifted = model(torch.roll(x, shifts=(16, 48), dims=(2, 3)))
        expected = torch.roll(model(x), shifts=(16, 48), dims=(2, 3))
    assert torch.allclose(shifted, expected, atol=1e-10)


def test_layer_norm_normalises_each_cell_over_its_channels():
    x = torch.randn(2, 6, 4, 5) * 3 + 2
    out = LayerNorm2d(6)(x)
    assert torch.allclose(out.mean(dim=1), torch.zeros(2, 4, 5), atol=1e-5)
    assert torch.allclose(out.var(dim=1, unbiased=False), torch.ones(2, 4, 5), atol=1e-3)


def test_a_few_steps_on_one_batch_drive_the_loss_down():
    torch.manual_seed(0)
    model = ConvNeXtUNet(widths=(8, 16, 16, 32), bottleneck=32, depths=(1, 1, 1, 1), mid_depth=1, dropout=0.0)
    x, y = torch.randn(2, 5, 64, 64), torch.rand(2, 1, 64, 64) + 0.5  # the 7x7 filters need at least 4 cells at the bottom level
    mask = torch.ones_like(y, dtype=torch.bool)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(150):
        loss = masked_l1(model(x), y, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
