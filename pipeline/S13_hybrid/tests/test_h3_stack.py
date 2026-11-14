"""Tests for the S13 H3 residual stacker (pipeline/S13_hybrid/s13_h3_stack.py)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lightgbm")

from pipeline.S13_hybrid.s13_h3_stack import training_rows  # noqa: E402


def test_a_fold_never_trains_on_its_own_cases():
    case_fold = np.array([0, 0, 1, 2, 2, 3, 4, 4])
    for k in range(5):
        rows = training_rows(case_fold, k)
        assert not rows[case_fold == k].any()
        assert rows[case_fold != k].all()
