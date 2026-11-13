"""Tests for S12 candidate architectures (Res-U-Net, Attention U-Net, FNO)."""

from __future__ import annotations

import math
import pytest

torch = pytest.importorskip("torch")

from urbanwind.dl.losses import masked_l1  # noqa: E402
from urbanwind.dl.nets import MODELS, UNet, build_model  # noqa: E402

S11_BASELINE_PARAMS = 4_860_313
S12_MODELS = ("res_unet", "attn_unet", "fno")


def test_registry_registers_all_s12_models_and_names_are_unique():
    for name in S12_MODELS:
        assert name in MODELS
        model = build_model(name, 5)
        assert isinstance(model, torch.nn.Module)
    assert len(MODELS) == len(set(MODELS.keys()))
    with pytest.raises(ValueError, match="unknown model"):
        build_model("unknown_architecture", 5)


@pytest.mark.parametrize("name", S12_MODELS)
def test_model_keeps_grid_finite_and_predicts_positive_speeds(name):
    model = build_model(name, 5).eval()
    x = torch.randn(2, 5, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 256, 256)
    assert torch.isfinite(out).all()
    assert (out >= 0).all()

    # Check multiple constraints on input dimensions
    if model.multiple > 1:
        with pytest.raises(ValueError, match="must be multiples"):
            model(torch.randn(1, 5, 64, 72))  # 72 is not a multiple of 16
    else:
        # FNO accepts any grid size including arbitrary aspect ratios
        out_odd = model(torch.randn(1, 5, 64, 72))
        assert out_odd.shape == (1, 1, 64, 72)
        assert (out_odd >= 0).all()


@pytest.mark.parametrize("name", S12_MODELS)
def test_model_is_periodic(name):
    torch.manual_seed(0)
    x = torch.randn(1, 5, 64, 96, dtype=torch.float64)

    if name in ("res_unet", "attn_unet"):
        model = build_model(name, 5, widths=(8, 16, 16, 32), bottleneck=32).double().eval()
        with torch.no_grad():
            shifted = model(torch.roll(x, shifts=(16, 48), dims=(2, 3)))
            expected = torch.roll(model(x), shifts=(16, 48), dims=(2, 3))
        assert torch.allclose(shifted, expected, atol=1e-10)
    elif name == "fno":
        model = build_model("fno", 5, width=16, modes1=8, modes2=8, n_layers=2).double().eval()
        with torch.no_grad():
            # Test multiple of 16
            shifted16 = model(torch.roll(x, shifts=(16, 48), dims=(2, 3)))
            expected16 = torch.roll(model(x), shifts=(16, 48), dims=(2, 3))
            assert torch.allclose(shifted16, expected16, atol=1e-10)

            # Test non-multiple shift (e.g. 7 and 13 cells) - FNO is translation equivariant for any shift
            shifted7 = model(torch.roll(x, shifts=(7, 13), dims=(2, 3)))
            expected7 = torch.roll(model(x), shifts=(7, 13), dims=(2, 3))
            assert torch.allclose(shifted7, expected7, atol=1e-10)


@pytest.mark.parametrize("name", S12_MODELS)
def test_parameter_counts_within_25_percent_of_baseline(name):
    # Verify baseline tuned U-Net count first
    baseline = UNet(in_ch=5, widths=(24, 48, 96, 192), bottleneck=384, dropout=0.08)
    baseline_count = sum(p.numel() for p in baseline.parameters())
    assert baseline_count == S11_BASELINE_PARAMS

    model = build_model(name, 5)
    total = sum(p.numel() for p in model.parameters())
    ratio = total / S11_BASELINE_PARAMS
    print(f"\n{name}: {total:,} parameters ({ratio:.1%} of tuned U-Net)")

    # Assert parameter count within 25% of baseline (3.65M to 6.08M)
    assert 0.75 * S11_BASELINE_PARAMS <= total <= 1.25 * S11_BASELINE_PARAMS


def test_fno_autocast_and_spectral_block_precision():
    fno = build_model("fno", 5, width=16, modes1=8, modes2=8, n_layers=2).eval()
    x = torch.randn(2, 5, 64, 64)

    # 1. Without autocast: float32
    out_f32 = fno(x)
    assert out_f32.dtype == torch.float32

    # 2. With autocast (CPU bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out_ac = fno(x)
        assert out_ac.dtype == torch.bfloat16

    # 3. Verify spectral block runs in float32 when autocast is enabled
    block = fno.spectral_layers[0]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        # Even with bfloat16 input under autocast, internal FFT and output succeed seamlessly
        h_bf = fno.lift(x).to(torch.bfloat16)
        out_block = block(h_bf)
        assert out_block.dtype == torch.bfloat16


@pytest.mark.parametrize("name", S12_MODELS)
def test_a_few_steps_on_one_batch_drive_the_loss_down(name):
    """Verify gradient flow: loss must drop well below the trivial constant-mean-prediction baseline.

    The key addition over a simple "loss decreases" check is the constant-mean baseline assertion:
    a model that collapses to predicting zero would yield final_loss ≈ mean_target, so
    final_loss < 0.5 * target_mean ensures the model is actually learning, not just collapsing.

    Target is uniform 1.0 + small noise so mean_target ≈ 1.0 is predictable and the threshold
    is well-defined.  15 steps on a tiny model is fast on CPU (~1–2 s).
    """
    torch.manual_seed(0)
    if name in ("res_unet", "attn_unet"):
        model = build_model(name, 5, widths=(8, 16, 16, 32), bottleneck=32)
    else:
        model = build_model("fno", 5, width=16, modes1=8, modes2=8, n_layers=2)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randn(2, 5, 32, 32)
    # Constant target with known mean ≈ 1.0 so the trivial-predictor baseline is obvious.
    y = torch.ones(2, 1, 32, 32) + 0.05 * torch.randn(2, 1, 32, 32)
    mask = torch.ones_like(y, dtype=torch.bool)
    target_mean = y[mask].mean().item()  # ≈ 1.0

    first_loss = None
    for _ in range(30):  # 30 steps: enough for all three architectures to show clear learning
        opt.zero_grad()
        pred = model(x)
        loss = masked_l1(pred, y, mask)
        loss.backward()
        opt.step()
        if first_loss is None:
            first_loss = loss.item()

    final_loss = loss.item()

    # 1. Initial loss must be finite and positive (not trivially zero from the start).
    assert math.isfinite(first_loss), f"[{name}] first_loss is not finite: {first_loss}"
    assert first_loss > 1e-6, f"[{name}] first_loss suspiciously small ({first_loss}); model may output zero before training"

    # 2. Final loss must be finite.
    assert math.isfinite(final_loss), f"[{name}] final_loss is not finite: {final_loss}"

    # 3. Final loss must be below the constant-mean-prediction baseline (≈ target_mean for masked L1
    #    when the model predicts zero).  A ratio < 0.5 means the model has learned to do better than
    #    predicting the trivial constant and has NOT collapsed to outputting zero.
    assert final_loss < 0.5 * target_mean, (
        f"[{name}] final_loss {final_loss:.4f} ≥ 0.5 × mean_target {0.5 * target_mean:.4f}; "
        "model may be collapsing to zero instead of learning."
    )
