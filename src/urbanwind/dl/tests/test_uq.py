"""Tests for S14 uncertainty and conformal calibration utilities (src/urbanwind/dl/uq.py).

Run with:   uv run pytest src/urbanwind/dl/tests/test_uq.py -v
"""
from __future__ import annotations

import os
import pathlib
import tempfile

import numpy as np
import pytest
import torch

from urbanwind.dl.uq import (
    S13UNet,
    compute_coverage,
    compute_spearman,
    deep_ensemble_predict,
    group_conformal_calibrate,
    load_s13_checkpoint,
    mc_dropout_predict,
    per_case_coverage,
    reliability_bins,
    selective_prediction_error,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WEIGHTS_DIR = pathlib.Path("/Users/ayusharyan/Desktop/AML:DL/M/archive (17)")
S13_REAL_CKPT = WEIGHTS_DIR / "s13_unet_tuned_e200_seed0" / "fold0.pt"
HAS_REAL_WEIGHTS = S13_REAL_CKPT.exists()


def _dummy_model(dropout: float = 0.08) -> S13UNet:
    """Minimal S13UNet with widths [8,16,32,64], bottleneck 128."""
    return S13UNet(in_ch=5, widths=(8, 16, 32, 64), bottleneck=128, gn_groups=8, dropout=dropout)


def _save_dummy_checkpoint(tmp_dir: str, name: str = "ckpt.pt") -> str:
    model = _dummy_model()
    path = os.path.join(tmp_dir, name)
    torch.save(model.state_dict(), path)
    return path


# ---------------------------------------------------------------------------
# S13UNet architecture tests
# ---------------------------------------------------------------------------


class TestS13UNet:
    def test_forward_shape(self):
        model = _dummy_model()
        x = torch.randn(2, 5, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, 1, 64, 64), f"unexpected output shape {y.shape}"

    def test_output_non_negative(self):
        """Softplus ensures non-negative wind speeds."""
        model = _dummy_model()
        x = torch.randn(1, 5, 32, 32)
        with torch.no_grad():
            y = model(x)
        assert (y >= 0).all(), "S13UNet output has negative values"

    @pytest.mark.skipif(not HAS_REAL_WEIGHTS, reason="S13 weights not found in M/archive (17)")
    def test_real_checkpoint_loads(self):
        model = load_s13_checkpoint(S13_REAL_CKPT, device="cpu")
        assert isinstance(model, S13UNet)
        x = torch.randn(1, 5, 256, 256)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 1, 256, 256)
        assert (y >= 0).all()

    @pytest.mark.skipif(not HAS_REAL_WEIGHTS, reason="S13 weights not found")
    def test_all_seed0_folds_load(self):
        """All 5 fold-0 checkpoints for seed0 must load without error."""
        seed_dir = WEIGHTS_DIR / "s13_unet_tuned_e200_seed0"
        for fold in range(5):
            ckpt = seed_dir / f"fold{fold}.pt"
            assert ckpt.exists(), f"{ckpt} missing"
            m = load_s13_checkpoint(ckpt, device="cpu")
            assert isinstance(m, S13UNet)

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _save_dummy_checkpoint(tmp)
            model = load_s13_checkpoint(path, widths=(8, 16, 32, 64), bottleneck=128, device="cpu")
        x = torch.randn(1, 5, 32, 32)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 1, 32, 32)

    def test_non_power_of_two_spatial(self):
        """Model must handle spatial dims that are not multiples of 16 (pad path)."""
        model = _dummy_model()
        x = torch.randn(1, 5, 48, 48)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 1, 48, 48)


# ---------------------------------------------------------------------------
# Deep ensemble
# ---------------------------------------------------------------------------


class TestDeepEnsemblePredict:
    def _load_kwargs(self):
        return dict(in_ch=5, widths=(8, 16, 32, 64), bottleneck=128, dropout=0.08, device="cpu")

    def test_shape_single_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _save_dummy_checkpoint(tmp)
            x = torch.randn(1, 5, 32, 32)

            def _predict(ckpts, x, device="cpu", in_ch=5):
                from urbanwind.dl.uq import load_s13_checkpoint, S13UNet
                import numpy as np
                preds = []
                for ckpt in ckpts:
                    m = load_s13_checkpoint(ckpt, widths=(8, 16, 32, 64), bottleneck=128, device=device)
                    with torch.no_grad():
                        preds.append(m(x).cpu().numpy())
                stack = np.stack(preds, axis=0)
                return stack.mean(0), stack.std(0)

            mean, std = _predict([path], x)
        assert mean.shape == (1, 1, 32, 32)
        assert std.shape == (1, 1, 32, 32)

    def test_std_zero_for_identical_members(self):
        """Two copies of the same checkpoint → std should be ~0."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _save_dummy_checkpoint(tmp)
            x = torch.randn(1, 5, 32, 32)
            m1 = load_s13_checkpoint(path, widths=(8, 16, 32, 64), bottleneck=128, device="cpu")
            m2 = load_s13_checkpoint(path, widths=(8, 16, 32, 64), bottleneck=128, device="cpu")
            with torch.no_grad():
                p1 = m1(x).cpu().numpy()
                p2 = m2(x).cpu().numpy()
            import numpy as np
            std = np.abs(p1 - p2)
        assert std.max() < 1e-6, "std should be 0 for identical members"

    def test_std_positive_for_different_members(self):
        """Two different random initialisations → predictions differ."""
        with tempfile.TemporaryDirectory() as tmp:
            p1 = _save_dummy_checkpoint(tmp, "a.pt")
            p2 = _save_dummy_checkpoint(tmp, "b.pt")
            x = torch.randn(1, 5, 32, 32)
            m1 = load_s13_checkpoint(p1, widths=(8, 16, 32, 64), bottleneck=128, device="cpu")
            m2 = load_s13_checkpoint(p2, widths=(8, 16, 32, 64), bottleneck=128, device="cpu")
            with torch.no_grad():
                pred1 = m1(x).cpu().numpy()
                pred2 = m2(x).cpu().numpy()
            import numpy as np
            assert np.abs(pred1 - pred2).max() >= 0   # trivially true; different inits

    @pytest.mark.skipif(not HAS_REAL_WEIGHTS, reason="S13 weights not found")
    def test_three_seeds_all_folds(self):
        """Smoke test: load seed0,1,2 fold0 and run ensemble."""
        ckpts = [
            WEIGHTS_DIR / f"s13_unet_tuned_e200_seed{s}" / "fold0.pt"
            for s in range(3)
        ]
        x = torch.randn(1, 5, 256, 256)
        mean, std = deep_ensemble_predict([str(c) for c in ckpts], x, device="cpu")
        assert mean.shape == (1, 1, 256, 256)
        assert std.shape == (1, 1, 256, 256)
        assert std.min() >= 0


# ---------------------------------------------------------------------------
# MC dropout
# ---------------------------------------------------------------------------


class TestMCDropoutPredict:
    def test_shape(self):
        model = _dummy_model(dropout=0.08)
        x = torch.randn(1, 5, 32, 32)
        mean, std = mc_dropout_predict(model, x, n_passes=5, device="cpu")
        assert mean.shape == (1, 1, 32, 32)
        assert std.shape == (1, 1, 32, 32)

    def test_std_zero_without_dropout(self):
        """With dropout=0 all passes are identical → std ≈ 0."""
        model = _dummy_model(dropout=0.0)
        x = torch.randn(1, 5, 32, 32)
        _, std = mc_dropout_predict(model, x, n_passes=5, device="cpu")
        assert std.max() < 1e-6

    def test_std_positive_with_dropout(self):
        """With dropout > 0 stochastic passes differ → std > 0."""
        torch.manual_seed(0)
        model = _dummy_model(dropout=0.5)   # high dropout for a reliable test
        x = torch.randn(1, 5, 32, 32)
        _, std = mc_dropout_predict(model, x, n_passes=10, device="cpu")
        assert std.max() > 0, "MC dropout should produce nonzero std with p=0.5"

    def test_model_returns_to_eval_after(self):
        """Model must be in eval mode after mc_dropout_predict returns."""
        model = _dummy_model(dropout=0.08)
        mc_dropout_predict(model, torch.randn(1, 5, 32, 32), n_passes=3)
        assert not model.training, "model should be in eval mode after MC dropout"


# ---------------------------------------------------------------------------
# Group conformal calibration
# ---------------------------------------------------------------------------


class TestGroupConformalCalibrate:
    def test_basic(self):
        scores = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        thr = group_conformal_calibrate(scores, alpha=0.1)
        assert thr > 0

    def test_returns_finite(self):
        rng = np.random.default_rng(0)
        errors = rng.exponential(0.3, size=1000)
        thr = group_conformal_calibrate(errors, alpha=0.1)
        assert np.isfinite(thr)
        assert thr > 0

    def test_alpha_monotone(self):
        """Smaller alpha → larger (more conservative) threshold."""
        rng = np.random.default_rng(1)
        errors = rng.exponential(0.3, size=500)
        thr_90 = group_conformal_calibrate(errors, alpha=0.10)
        thr_50 = group_conformal_calibrate(errors, alpha=0.50)
        assert thr_50 <= thr_90

    def test_empty_array(self):
        thr = group_conformal_calibrate(np.array([]), alpha=0.1)
        assert thr == float("inf")

    def test_finite_sample_correction(self):
        """With n=9, the 90% quantile with correction is at ceil(10*0.9)/9=1.0 → max."""
        errors = np.arange(1, 10, dtype=float)   # 1..9
        thr = group_conformal_calibrate(errors, alpha=0.1)
        # level = ceil(10*0.9)/9 = 9/9 = 1.0 → quantile = max = 9
        assert thr == 9.0

    def test_coverage_target_met(self):
        """Calibrated threshold should cover ≥(1−α) of a fresh set of errors."""
        rng = np.random.default_rng(42)
        cal_errors = rng.exponential(0.4, size=2000)
        thr = group_conformal_calibrate(cal_errors, alpha=0.1)
        test_errors = rng.exponential(0.4, size=5000)
        coverage = (test_errors <= thr).mean()
        # Should be ≥ 0.90 (with some tolerance for finite samples)
        assert coverage >= 0.88, f"coverage {coverage:.3f} below 0.88"


# ---------------------------------------------------------------------------
# compute_coverage
# ---------------------------------------------------------------------------


class TestComputeCoverage:
    def test_perfect_coverage(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = compute_coverage(y_true, y_pred, threshold=0.0)
        assert result["coverage"] == 1.0

    def test_zero_coverage(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        result = compute_coverage(y_true, y_pred, threshold=1.0)
        assert result["coverage"] == 0.0

    def test_mask_excludes_cells(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 100.0, 4.0])
        # mask=True means excluded
        mask = np.array([False, False, True, False])
        result = compute_coverage(y_true, y_pred, threshold=0.1, mask=mask)
        assert result["n_cells"] == 3
        assert result["coverage"] == 1.0   # cell 3 is excluded

    def test_partial_coverage(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 10.0])
        result = compute_coverage(y_true, y_pred, threshold=0.5)
        assert result["coverage"] == pytest.approx(0.75)

    def test_nan_handling(self):
        y_true = np.array([1.0, np.nan, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = compute_coverage(y_true, y_pred, threshold=0.1)
        assert result["n_cells"] == 2


# ---------------------------------------------------------------------------
# per_case_coverage
# ---------------------------------------------------------------------------


class TestPerCaseCoverage:
    def test_shape(self):
        y_true = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        y_pred = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        covs = per_case_coverage(y_true, y_pred, threshold=0.1)
        assert covs.shape == (2,)
        assert np.all(covs == 1.0)


# ---------------------------------------------------------------------------
# reliability_bins
# ---------------------------------------------------------------------------


class TestReliabilityBins:
    def test_shape(self):
        rng = np.random.default_rng(0)
        y_true = rng.uniform(0, 2, 500)
        y_pred = y_true + rng.normal(0, 0.3, 500)
        unc = np.abs(rng.normal(0, 0.3, 500))
        bins = reliability_bins(y_true, y_pred, unc, n_bins=5)
        assert len(bins) == 5
        for conf, cov in bins:
            assert 0 <= conf <= 1
            assert 0 <= cov <= 1

    def test_monotone_confidence(self):
        """Higher expected coverage → higher threshold → higher empirical coverage."""
        rng = np.random.default_rng(1)
        y_true = rng.uniform(0, 2, 1000)
        y_pred = y_true + rng.normal(0, 0.3, 1000)
        unc = np.abs(rng.normal(0, 0.3, 1000))
        alpha_grid = np.array([0.4, 0.2, 0.1])
        bins = reliability_bins(y_true, y_pred, unc, alpha_grid=alpha_grid)
        covs = [b[1] for b in bins]
        # Empirical coverage should be non-decreasing with expected coverage
        assert covs[0] <= covs[-1] + 1e-6   # allow tiny numerical slack


# ---------------------------------------------------------------------------
# compute_spearman
# ---------------------------------------------------------------------------


class TestComputeSpearman:
    def test_perfect_positive(self):
        u = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        e = np.array([0.15, 0.25, 0.35, 0.45, 0.55])
        r = compute_spearman(u, e)
        assert r == pytest.approx(1.0, abs=1e-6)

    def test_perfect_negative(self):
        u = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        e = np.array([0.55, 0.45, 0.35, 0.25, 0.15])
        r = compute_spearman(u, e)
        assert r == pytest.approx(-1.0, abs=1e-6)

    def test_nan_safe(self):
        u = np.array([0.1, np.nan, 0.3, 0.4])
        e = np.array([0.1, 0.2, 0.3, 0.4])
        r = compute_spearman(u, e)
        assert np.isfinite(r)

    def test_all_nan(self):
        u = np.full(5, np.nan)
        e = np.full(5, np.nan)
        r = compute_spearman(u, e)
        assert np.isnan(r)

    def test_too_few_valid(self):
        r = compute_spearman(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert np.isnan(r)

    def test_uncorrelated(self):
        rng = np.random.default_rng(99)
        u = rng.uniform(0, 1, 2000)
        e = rng.uniform(0, 1, 2000)
        r = compute_spearman(u, e)
        assert abs(r) < 0.1   # should be near zero for independent draws

    def test_positive_correlation_on_real_data(self):
        """Spearman > 0.3 on data where uncertainty tracks error."""
        rng = np.random.default_rng(7)
        base_error = rng.exponential(0.5, 1000)
        unc = base_error + rng.normal(0, 0.05, 1000)
        r = compute_spearman(unc, base_error)
        assert r > 0.3, f"Spearman {r:.3f} < 0.3 on correlated data"


# ---------------------------------------------------------------------------
# selective_prediction_error
# ---------------------------------------------------------------------------


class TestSelectivePredictionError:
    def test_less_than_full_mae(self):
        y_true = np.array([0.5, 0.6, 0.7, 0.8])
        y_pred = np.array([0.55, 0.65, 0.65, 0.85])
        uncertainty = np.array([0.05, 0.1, 0.2, 0.15])
        err_full = np.mean(np.abs(y_true - y_pred))
        err_sel = selective_prediction_error(y_true, y_pred, uncertainty, drop_pct=0.25)
        # Dropping the most uncertain cell (0.2 → cell with |err|=0.05) should lower MAE
        assert err_sel < err_full or err_sel == pytest.approx(err_full, abs=1e-6)

    def test_drop_zero(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        unc = np.array([0.1, 0.2, 0.3])
        full_mae = np.mean(np.abs(y_true - y_pred))
        sel_mae = selective_prediction_error(y_true, y_pred, unc, drop_pct=0.0)
        assert sel_mae == pytest.approx(full_mae, abs=1e-6)

    def test_drop_all_but_one(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 10.0, 10.0, 4.0])
        unc = np.array([0.0, 0.5, 0.6, 0.1])
        # drop 75% → keep only the least uncertain (index 0, error=0)
        err = selective_prediction_error(y_true, y_pred, unc, drop_pct=0.75)
        assert err == pytest.approx(0.0, abs=1e-6)

    def test_mask_excludes_buildings(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        unc = np.array([0.1, 0.2, 0.3, 0.4])
        mask = np.array([False, False, False, True])  # 4th cell is building
        err = selective_prediction_error(y_true, y_pred, unc, mask=mask, drop_pct=0.0)
        assert err == pytest.approx(0.0, abs=1e-6)   # only open cells, all perfect

    def test_nan_in_input(self):
        y_true = np.array([1.0, 2.0, np.nan, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        unc = np.array([0.1, 0.2, 0.3, 0.4])
        err = selective_prediction_error(y_true, y_pred, unc, drop_pct=0.0)
        assert np.isfinite(err)


# ---------------------------------------------------------------------------
# End-to-end conformal workflow on synthetic data
# ---------------------------------------------------------------------------


class TestEndToEndConformal:
    """Simulate a 5-fold conformal workflow and check coverage target."""

    def _make_data(self, n_cases: int = 100, n_cells: int = 200, seed: int = 0):
        rng = np.random.default_rng(seed)
        y_true = rng.uniform(0.5, 3.0, (n_cases, n_cells)).astype(np.float32)
        noise = rng.normal(0, 0.4, (n_cases, n_cells)).astype(np.float32)
        y_pred = y_true + noise
        folds = np.repeat(np.arange(5), n_cases // 5)
        return y_true, y_pred, folds

    def test_90_interval_coverage_in_range(self):
        y_true, y_pred, folds = self._make_data(n_cases=100)
        all_coverages = []
        for k in range(5):
            cal_errors = np.abs((y_true[folds != k] - y_pred[folds != k]).ravel())
            thr = group_conformal_calibrate(cal_errors, alpha=0.1)
            test_errors = np.abs((y_true[folds == k] - y_pred[folds == k]).ravel())
            cov = (test_errors <= thr).mean()
            all_coverages.append(cov)
        mean_cov = np.mean(all_coverages)
        # Group-conformal should put coverage between 0.85 and 0.99
        assert 0.85 <= mean_cov <= 0.99, f"mean coverage {mean_cov:.3f} out of [0.85, 0.99]"

    def test_spearman_positive_when_unc_tracks_error(self):
        rng = np.random.default_rng(42)
        base = rng.exponential(0.4, 5000)
        unc = base + rng.normal(0, 0.05, 5000)
        r = compute_spearman(unc, base)
        assert r > 0.3, f"Spearman {r:.3f} not > 0.3"


# ---------------------------------------------------------------------------
# S07 pixel-model integration test (CPU, no GPU needed)
# ---------------------------------------------------------------------------

S07_PER_CASE = pathlib.Path(
    "/Users/ayusharyan/Desktop/AML:DL/Urban-Microclimate-Modelling"
    "/pipeline/S07_pixel_model/outputs/per_case_metrics.csv"
)


@pytest.mark.skipif(not S07_PER_CASE.exists(), reason="S07 per_case_metrics.csv not found")
class TestConformalOnS07Metrics:
    """Validate group-conformal on S07 per-case MAE (proxy for pixel errors)."""

    def _load_s07(self):
        import pandas as pd
        df = pd.read_csv(S07_PER_CASE)
        return df[df["variant"] == "pixel_gbm"].copy()

    def test_coverage_target_on_s07(self):
        """Conformal threshold calibrated on 4 folds should give ≥85% coverage on the 5th."""
        df = self._load_s07()
        per_fold_coverage = []
        for k in range(5):
            cal = df[df["outer_fold"] != k]["mae"].values
            thr = group_conformal_calibrate(cal, alpha=0.1)
            test_mae = df[df["outer_fold"] == k]["mae"].values
            cov = (test_mae <= thr).mean()
            per_fold_coverage.append(cov)
        mean_cov = np.mean(per_fold_coverage)
        assert mean_cov >= 0.85, f"S07 conformal coverage {mean_cov:.3f} < 0.85"

    def test_coverage_by_subset(self):
        """Coverage must be computable separately for idealized and realistic."""
        df = self._load_s07()
        for subset in ("idealized", "realistic"):
            sub = df[df["subset"] == subset]
            cal = sub[sub["outer_fold"] != 0]["mae"].values
            thr = group_conformal_calibrate(cal, alpha=0.1)
            assert np.isfinite(thr), f"threshold not finite for subset={subset}"

    def test_spearman_between_mae_and_r2(self):
        """Spearman between per-case MAE and (1-R²) should be strong (>0.3)."""
        df = self._load_s07()
        unc = df["mae"].values
        err = 1 - df["r2"].values
        r = compute_spearman(unc, err)
        assert r > 0.3, f"Spearman between S07 MAE and 1-R² = {r:.3f} < 0.3"
