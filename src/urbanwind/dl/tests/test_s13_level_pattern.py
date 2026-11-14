"""Tests for the S13 level test (pipeline/S13_hybrid/s13_level_pattern.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")  # the script imports the trainer for the fold table

from pipeline.S13_hybrid.s13_level_pattern import pooled_r2, rescale_to_level  # noqa: E402


def test_rescaling_sets_the_open_ground_mean_and_keeps_the_pattern():
    pred = np.array([[1.0, 2.0], [3.0, 100.0]])
    valid = np.array([[True, True], [True, False]])  # the 100 is a building cell
    out = rescale_to_level(pred, valid, level=4.0)
    assert out[valid].mean() == pytest.approx(4.0)
    assert out[0, 0] / out[0, 1] == pytest.approx(0.5)  # one factor for the whole case


def test_pooled_r2_from_sums_matches_r2_over_all_cells():
    rng = np.random.default_rng(0)
    cases = [(rng.normal(3, 1, 50), rng.normal(5, 2, 30))]
    truth = [t for pair in cases for t in pair]
    pred = [t + rng.normal(0, 0.5, t.size) for t in truth]
    rows = pd.DataFrame([{"n": t.size, "sum_t": t.sum(), "sum_t2": (t * t).sum(), "sse": ((p - t) ** 2).sum()}
                         for t, p in zip(truth, pred)])
    t_all, p_all = np.concatenate(truth), np.concatenate(pred)
    expected = 1 - ((p_all - t_all) ** 2).sum() / ((t_all - t_all.mean()) ** 2).sum()
    assert pooled_r2(rows) == pytest.approx(expected)
