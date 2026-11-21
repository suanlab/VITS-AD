"""Scoring of the registered P (paper_numbers.gutentag) on synthetic stand-in results."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.paper_numbers as pn


def _fake_results(index: pd.Index, gain: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {}
    for d in pn.SUB_DETECTORS:
        base = rng.uniform(0.1, 0.5, len(index))
        cols[f"{d}|window"] = base
        cols[f"{d}|global"] = base + gain + 0.01 * rng.standard_normal(len(index))
        cols[f"{d}|window|test"] = base
        cols[f"{d}|global|test"] = base + gain
    return pd.DataFrame(cols, index=index)


def test_pending_until_every_series_has_run(monkeypatch: pytest.MonkeyPatch) -> None:
    lst = pd.read_csv(pn.ART / "gutentag_listing.csv")
    monkeypatch.setattr(pn, "keyed", lambda pattern, value="VUS-PR": _fake_results(pd.Index(lst["file_name"][:10]), 0.1))
    assert pn.gutentag()["status"] == "pending"


def test_scores_every_registered_part(monkeypatch: pytest.MonkeyPatch) -> None:
    lst = pd.read_csv(pn.ART / "gutentag_listing.csv")
    gt = _fake_results(pd.Index(lst["file_name"]), 0.1)
    eva = _fake_results(pd.Index([f"e{i}_SRC{i % 5}_x" for i in range(50)]), 0.2, seed=1)
    monkeypatch.setattr(pn, "keyed", lambda pattern, value="VUS-PR": gt if "gutentag" in pattern else eva)
    res = pn.gutentag()
    assert res["status"] == "complete" and res["series"] == len(lst)
    assert res["P1"]["by_detector"]["Sub_KNN"]["observed"] == pytest.approx(0.1, abs=0.01)
    assert res["P2"]["detectors_holding"] in range(7) and res["P2"]["decision"] in ("supported", "refuted", "inconclusive")
    assert res["P3"]["O1"]["n"] == 330 and res["P3"]["decision"] in ("supported", "refuted", "inconclusive")
