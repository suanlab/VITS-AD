"""Tests for the preprocessing-contract records (scripts/make_contract_records.py)."""

from __future__ import annotations

import pandas as pd

from scripts.make_contract_records import SCOPES, _scope, records


def test_scope_mapping_keeps_composite_steps() -> None:
    assert _scope("per-window") == ("window", [])
    assert _scope("per-window (+ column-wise with train stats)") == ("window", ["column-wise with train stats"])
    assert _scope("per-time-step across channels (degenerate: D=1)")[0] == "time_step_across_channels"
    assert _scope("n/a - detector raises before scoring")[0] == "not_run"


def test_records_use_the_schema_vocabulary() -> None:
    audit = pd.DataFrame([{"detector": "Sub-KNN", "subset": "U", "state": "A(35ba9c3b)", "normalization_class": "per-window",
                           "train_score_mismatch": "no", "detail": "", "evidence_file_lines": "", "changed_by_commit": "",
                           "confidence": "high"}])
    rec = records(audit)["A(35ba9c3b)"][0]
    assert rec["scope"] in SCOPES and rec["audited_commit"] == "35ba9c3b" and rec["statistics_from"] == "each window"
    assert rec["results_commit"] is None and rec["fit_score_mismatch"] == "no"
