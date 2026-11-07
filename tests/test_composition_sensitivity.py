"""Tests for the composition range, flip radius and error budget (scripts/composition_sensitivity.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.composition_sensitivity import analyse, analyse_with_pool, error_budget, flip_radius, main


def _make_split(counts: dict[str, int], means: dict[str, float]) -> tuple[pd.Series, pd.Series]:
    items, src, diff = [], [], []
    for s, n in counts.items():
        for i in range(n):
            items.append(f"{s}_{i}")
            src.append(s)
            diff.append(means[s])
    return pd.Series(diff, index=items), pd.Series(src, index=items)


def test_flip_radius_matches_closed_form() -> None:
    # Two sources: mean = 0.8*0.2 + 0.2*(-0.4) = 0.08; moving mass t from A to B gives
    # 0.08 - 0.6 t = 0 at t = 0.1333...
    r = flip_radius(pd.Series({"A": 0.2, "B": -0.4}), pd.Series({"A": 0.8, "B": 0.2}))
    assert r == pytest.approx(0.08 / 0.6, abs=1e-9)


def test_flip_radius_none_without_reversing_source() -> None:
    assert flip_radius(pd.Series({"A": 0.2, "B": 0.1}), pd.Series({"A": 0.5, "B": 0.5})) is None


def test_flip_radius_negative_mean_and_cap() -> None:
    m, w = pd.Series({"A": -0.3, "B": 0.1}), pd.Series({"A": 0.5, "B": 0.5})
    # mean -0.1; B needs weight 0.75 -> radius 0.25
    assert flip_radius(m, w) == pytest.approx(0.25, abs=1e-9)
    # capping B at 0.6 makes the reversal impossible
    assert flip_radius(m, w, pd.Series({"A": 1.0, "B": 0.6})) is None


def test_analyse_reports_range_and_reversing_sources() -> None:
    diff, src = _make_split({"A": 8, "B": 2}, {"A": 0.2, "B": -0.4})
    out = analyse(diff, src)
    assert out["mean"] == pytest.approx(0.08)
    assert out["range"] == pytest.approx([-0.4, 0.2])
    assert out["reversing_sources"] == ["B"]
    assert out["flip_radius_items"] == pytest.approx(10 * 0.08 / 0.6)


def test_realizable_radius_respects_pool_size() -> None:
    # Pool: A 20 items (+0.2), B 3 items (-0.4); split of 10 has 8 A, 2 B.
    pool_diff, pool_src = _make_split({"A": 20, "B": 3}, {"A": 0.2, "B": -0.4})
    split = pd.Index([f"A_{i}" for i in range(8)] + ["B_0", "B_1"])
    out = analyse_with_pool(pool_diff, pool_src, split, n_split=10)
    # B can fill at most 3 of 10 items: mean at w_B = 0.3 is 0.7*0.2 - 0.3*0.4 = 0.02 > 0
    assert out["flip_radius"] is None
    out = analyse_with_pool(pool_diff, pool_src, split, n_split=5)  # cap 0.6 now allows it
    assert out["flip_radius"] == pytest.approx(0.08 / 0.6, abs=1e-9)


def test_error_budget_terms() -> None:
    rng = np.random.default_rng(0)
    vals = pd.Series(rng.normal(0.5, 0.1, 200), index=[f"i{k}" for k in range(200)])
    src = pd.Series(["s" + str(k % 10) for k in range(200)], index=vals.index)
    out = error_budget(vals, src, {"normalization": -0.2, "split": 0.05, "missing": None})
    assert out["systematic"] == {"normalization": 0.2, "split": 0.05}
    assert out["largest_systematic"] == "normalization"
    assert 0 < out["items_half"] < 0.03
    assert out["largest_over_items_half"] > 5


def test_cli_long_format(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rows = []
    for s, n, d in (("A", 8, 0.2), ("B", 2, -0.4)):
        for i in range(n):
            rows += [{"item": f"{s}{i}", "source": s, "system": "x", "value": 0.5 + d},
                     {"item": f"{s}{i}", "source": s, "system": "y", "value": 0.5}]
    path = tmp_path / "r.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    main([str(path), "--a", "x", "--b", "y"])
    out = json.loads(capsys.readouterr().out)
    assert out["flip_radius"] == pytest.approx(0.08 / 0.6, abs=1e-9)
