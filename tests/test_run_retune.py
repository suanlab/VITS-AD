"""Selection rule of the registered re-tuning (scripts/run_retune.py)."""

from __future__ import annotations

from typing import Any

import pytest

import scripts.run_retune as rt


def _make_records(values: dict[str, list[float | None]]) -> list[dict[str, Any]]:
    n = len(next(iter(values.values())))
    return [{"series": f"s{i}", **{k: v[i] for k, v in values.items()}} for i in range(n)]


def test_select_failure_scores_zero_and_ties_go_to_grid_order(monkeypatch: pytest.MonkeyPatch) -> None:
    g = [{"k": 1}, {"k": 2}, {"k": 3}]
    monkeypatch.setattr(rt, "grid", lambda det: g)
    monkeypatch.setitem(rt.TUNED, "Sub_X", {"k": 2})
    key = lambda v, k: f"Sub_X|{v}|{rt.config_key({'k': k})}"  # noqa: E731
    recs = _make_records({
        key("window", 1): [0.5, 0.5], key("window", 2): [0.5, 0.5], key("window", 3): [1.0, None],
        key("global", 1): [0.2, 0.2], key("global", 2): [0.4, 0.4], key("global", 3): [0.4, 0.4],
    })
    sel = rt.select(recs, ["Sub_X"])
    # window: k=3 has a failure scored 0 -> mean 0.5, tied with k=1 and k=2; first in grid order wins
    assert sel["Sub_X|window"]["params"] == {"k": 1}
    assert sel["Sub_X|window"]["equals_tsbad_tuned"] is False
    # global: k=2 and k=3 tie at 0.4; k=2 comes first and equals TSB-AD's tuned config
    assert sel["Sub_X|global"]["params"] == {"k": 2}
    assert sel["Sub_X|global"]["equals_tsbad_tuned"] is True
    assert sel["Sub_X|global"]["tsbad_tuned_mean"] == pytest.approx(0.4)
