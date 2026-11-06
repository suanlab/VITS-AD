"""Regression tests for the paper's GPU-free pipeline: released numbers -> tables.

The tables the manuscript includes are generated from artifacts/official_protocol/paper_numbers.json;
regenerating them must give the committed files byte for byte. The normalization swap that the
toggle applies to TSB-AD's detectors is tested on a synthetic series.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NUMBERS = ROOT / "artifacts/official_protocol/paper_numbers.json"
TABLES = ROOT / "paper/pvldb2027/tables"


@pytest.mark.skipif(not NUMBERS.exists() or not TABLES.exists(), reason="paper artifacts absent")
def test_tables_regenerate_byte_identical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.make_paper_tables as mpt

    monkeypatch.chdir(ROOT)
    monkeypatch.setattr("sys.argv", ["make_paper_tables", "--out", str(tmp_path)])
    mpt.main()
    for f in sorted(TABLES.glob("*.tex")):
        assert (tmp_path / f.name).read_bytes() == f.read_bytes(), f.name


def test_window_variants_normalize_as_declared() -> None:
    pytest.importorskip("TSB_AD")
    from scripts.run_detector_normalization import _make_window_class

    x = 5.0 + 2.0 * np.sin(np.arange(200) / 7.0)
    per_window = _make_window_class("window")(window=20).convert(x)
    global_ = _make_window_class("global")(window=20).convert(x)
    rows = np.asarray(per_window, dtype=float)
    assert np.allclose(rows.mean(axis=1), 0.0, atol=1e-9)
    assert np.allclose(rows.std(axis=1, ddof=1), 1.0, atol=1e-9)
    # the global variant leaves windows as they are (the series is standardized before windowing)
    assert np.allclose(np.asarray(global_, dtype=float)[0], x[:20])


@pytest.mark.skipif(not NUMBERS.exists(), reason="paper artifacts absent")
def test_no_duplicate_series_conflicts() -> None:
    numbers = json.loads(NUMBERS.read_text())
    assert numbers.get("key_conflicts", {}).get("n", 0) == 0
