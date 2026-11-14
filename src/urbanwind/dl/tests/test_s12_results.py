"""Tests for S12 results scoring, S11 baseline reconstruction fallback, and output safety."""

from __future__ import annotations

import pathlib
import sys
import pytest

pd = pytest.importorskip("pandas")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
STAGE_PATH = str(REPO_ROOT / "pipeline" / "S12_architecture_search")
if STAGE_PATH not in sys.path:
    sys.path.insert(0, STAGE_PATH)

from s12_results import S11_METRICS_CSV, load_s11_baseline_per_case, pooled_r2  # noqa: E402


def test_s11_baseline_reconstruction_reproduces_summary_pooled_r2():
    """Verify that reconstructed metrics from per_case_metrics.csv reproduce S11 summary.csv pooled R2."""
    if not S11_METRICS_CSV.exists():
        pytest.skip(f"S11 per_case_metrics.csv not found at {S11_METRICS_CSV}")

    expected = {
        "s11_unet_tuned_seed0": 0.7092,
        "s11_unet_tuned_seed1": 0.7122,
        "s11_unet_tuned_seed2": 0.7111,
    }

    required_cols = [
        "case",
        "r2",
        "delta_p95",
        "mae",
        "subset",
        "split_group",
        "outer_fold",
        "flagged",
        "n",
        "run",
        "sum_t",
        "sum_t2",
        "sse",
    ]

    for run_name, exp_r2 in expected.items():
        df = load_s11_baseline_per_case(ref_run=run_name)
        assert len(df) == 482
        for col in required_cols:
            assert col in df.columns, f"Missing required column '{col}' in reconstructed DataFrame"
        actual_r2 = round(pooled_r2(df), 4)
        assert actual_r2 == exp_r2, f"{run_name}: reconstructed pooled R2 {actual_r2} != expected {exp_r2}"
