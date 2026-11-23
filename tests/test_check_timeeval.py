"""Helpers of the second-harness checks (scripts/check_timeeval.py) and the TimeEval contract records."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.check_timeeval import _auc, _to_points
from scripts.make_contract_records import SCHEMA, SCOPES, timeeval_records


def test_reverse_windowing_averages_covering_windows() -> None:
    w, n = 3, 6
    windows = np.arange(n - w + 1, dtype=float)          # window k starts at point k
    pts = _to_points(windows, n)
    assert pts.size == n
    assert pts[0] == pytest.approx(0.0) and pts[2] == pytest.approx(1.0) and pts[-1] == pytest.approx(3.0)
    assert np.array_equal(_to_points(np.ones(n), n), np.ones(n))


def test_auc_matches_definition() -> None:
    y = np.array([0, 0, 1, 1])
    assert _auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert _auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert _auc(y, np.ones(4)) == pytest.approx(0.5)


def test_timeeval_records_follow_the_schema() -> None:
    recs = timeeval_records()
    assert len(recs) == 21
    for r in recs:
        assert set(SCHEMA["required"]) <= set(r) and r["scope"] in SCOPES and r["results_commit"] is None
    assert sum(r["reads_labels"] for r in recs) == 10
