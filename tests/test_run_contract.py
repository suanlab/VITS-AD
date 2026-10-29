"""Negative tests for the run contract of the per-series experiment scripts and the aggregation.

A missing input, a resumed output with a different configuration, conflicting duplicate values and
an incomplete registered run must stop with an error instead of being aggregated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_contract import fingerprint_path, open_run, require_inputs, save_records


def _make_inputs(tmp_path: Path, names: list[str]) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    for n in names:
        (data / n).write_text("x,Label\n0,0\n")
    return data


def test_missing_input_raises(tmp_path: Path) -> None:
    data = _make_inputs(tmp_path, ["a.csv"])
    with pytest.raises(FileNotFoundError, match="1 of 2"):
        require_inputs(["a.csv", "b.csv"], data)
    present, missing = require_inputs(["a.csv", "b.csv"], data, allow_missing=True)
    assert present == ["a.csv"] and missing == ["b.csv"]


def test_changed_configuration_refuses_resume(tmp_path: Path) -> None:
    out = tmp_path / "res_shard0.json"
    script = tmp_path / "runner.py"
    script.write_text("# runner\n")
    assert open_run(out, {"detectors": ["Sub_KNN"]}, ["a.csv"], script) == []
    save_records(out, [{"series": "a.csv", "Sub_KNN|window": {"VUS-PR": 0.5}}])
    assert len(open_run(out, {"detectors": ["Sub_KNN"]}, ["a.csv"], script)) == 1
    with pytest.raises(RuntimeError, match="config"):
        open_run(out, {"detectors": ["Sub_LOF"]}, ["a.csv"], script)
    with pytest.raises(RuntimeError, match="series_sha256"):
        open_run(out, {"detectors": ["Sub_KNN"]}, ["a.csv", "b.csv"], script)


def test_source_change_is_recorded_not_fatal(tmp_path: Path) -> None:
    out = tmp_path / "res_shard0.json"
    script = tmp_path / "runner.py"
    script.write_text("# v1\n")
    open_run(out, {}, ["a.csv"], script)
    script.write_text("# v2\n")
    open_run(out, {}, ["a.csv"], script)
    assert len(json.loads(fingerprint_path(out).read_text())["code_sha256_seen"]) == 2


def test_legacy_output_gets_marked_fingerprint(tmp_path: Path) -> None:
    out = tmp_path / "res_shard0.json"
    script = tmp_path / "runner.py"
    script.write_text("# runner\n")
    out.write_text(json.dumps([{"series": "a.csv"}]))
    assert len(open_run(out, {}, ["a.csv"], script)) == 1
    assert "predate" in json.loads(fingerprint_path(out).read_text())["note"]


def test_conflicting_duplicates_are_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.paper_numbers as pn

    sub = tmp_path / "TSB-AD-U"
    sub.mkdir()
    rec = {"series": "a.csv", "raw_flatten": {"VUS-PR": 0.5}}
    (sub / "per_series_raw_shard0.json").write_text(json.dumps([rec]))
    (sub / "per_series_raw_shard0b.json").write_text(
        json.dumps([{**rec, "raw_flatten": {"VUS-PR": 0.6}}]))
    monkeypatch.setattr(pn, "ART", tmp_path)
    monkeypatch.setattr(pn, "KEY_CONFLICTS", [])
    pn.arms("TSB-AD-U", "raw", ("raw_flatten",))
    assert len(pn.KEY_CONFLICTS) == 1
    monkeypatch.setattr(pn, "KEY_CONFLICTS", [])
    pn.keyed("TSB-AD-U/per_series_raw_shard*.json")
    assert len(pn.KEY_CONFLICTS) == 1


def test_threshold_sub_requires_every_eligible_series(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.paper_numbers as pn

    sub = tmp_path / "TSB-AD-U"
    sub.mkdir()
    monkeypatch.setattr(pn, "ART", tmp_path)
    (sub / "threshold_sub_shard0.json").write_text(json.dumps([{"series": "a.csv"}]))
    assert pn.threshold_sub()["reason"] == "no list of eligible series"
    (sub / "threshold_sub_expected.json").write_text(
        json.dumps({"eligible": ["a.csv", "b.csv"], "skipped": {"c.csv": "no anomalous test point"}}))
    assert pn.threshold_sub() == {"status": "incomplete", "series": 1, "absent": 1}
    (sub / "threshold_sub_shard1.json").write_text(json.dumps([{"series": "c.csv"}]))
    with pytest.raises(RuntimeError, match="not eligible"):
        pn.threshold_sub()
    (sub / "threshold_sub_shard1.json").unlink()
    monkeypatch.setattr(pn, "KEY_CONFLICTS", [])
    (sub / "threshold_sub_shard0b.json").write_text(json.dumps([{"series": "a.csv", "none|Sub_KNN|window": 1}]))
    (sub / "threshold_sub_shard0c.json").write_text(json.dumps([{"series": "a.csv", "none|Sub_KNN|window": 2}]))
    pn.threshold_sub()
    assert len(pn.KEY_CONFLICTS) == 1
