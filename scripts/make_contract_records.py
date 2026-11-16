"""Preprocessing contracts for TSB-AD's entries, as machine-readable records.

Writes artifacts/preprocessing_contract/:
  schema.json       JSON Schema of one contract record (the fields Sec. 8 of the paper proposes)
  tsbad_<state>.json  one record per detector entry of TSB-AD at each audited code state, filled in
                    from artifacts/normalization_audit/norm_audit.csv (the audit's wording is kept in
                    `detail` and `fit_score`; `scope` maps the audit class onto the schema's vocabulary)
  summary.json      counts of `scope` per state and subset

    python -m scripts.make_contract_records
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)
AUDIT = Path("artifacts/normalization_audit/norm_audit.csv")
OUT = Path("artifacts/preprocessing_contract")

SCOPES = ("window", "time_step_across_channels", "series_per_channel", "across_windows", "model_internal",
          "none", "unclear", "not_run")
# The first matching prefix wins; a composite class keeps its first (outermost) step in `scope` and
# the rest in `additional`.
CLASS_TO_SCOPE = (
    ("per-window", "window"),
    ("per-time-step across channels", "time_step_across_channels"),
    ("per-channel over the series", "series_per_channel"),
    ("column-wise per channel", "series_per_channel"),
    ("column-wise across windows", "across_windows"),
    ("model-internal", "model_internal"),
    ("none", "none"),
    ("unclear", "unclear"),
    ("n/a", "not_run"),
)

SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Preprocessing contract of one benchmark entry",
    "type": "object",
    "required": ["entry", "subset", "code_commit", "applied_by", "scope", "statistics_from", "fit_score"],
    "properties": {
        "entry": {"type": "string", "description": "detector or method name as on the leaderboard"},
        "subset": {"type": "string", "description": "table the entry belongs to (e.g. U, M, both)"},
        "code_commit": {"type": "string", "description": "commit of the code that produced the entry's results"},
        "applied_by": {"enum": ["harness", "detector", "library", "none"],
                       "description": "where normalization happens"},
        "scope": {"enum": list(SCOPES), "description": "unit over which statistics are computed"},
        "additional": {"type": "array", "items": {"type": "string"},
                       "description": "further normalization steps, in order"},
        "statistics_from": {"type": "string",
                            "description": "data the statistics come from (window, training part, whole series, ...)"},
        "fit_score": {"type": "string",
                      "description": "whether fitting and scoring apply the same transform with the same statistics"},
        "detail": {"type": "string"},
        "evidence": {"type": "string", "description": "file and line references"},
        "changed_by_commit": {"type": "string"},
        "confidence": {"type": "string"},
    },
}


def _scope(cls: str) -> tuple[str, list[str]]:
    parts = [p.strip() for p in cls.replace(", then ", " + ").replace(" (+ ", " + ").rstrip(")").split(" + ")]
    for prefix, scope in CLASS_TO_SCOPE:
        if cls.startswith(prefix):
            return scope, parts[1:]
    return "unclear", parts


def _statistics_from(cls: str, scope: str) -> str:
    if "fit: train part; score: whole series" in cls:
        return "fit: training part; score: whole series"
    if "whole series" in cls:
        return "whole series"
    if "train stats" in cls:
        return "own unit, then training part"
    return {"window": "each window", "time_step_across_channels": "each time step", "model_internal": "model input",
            "none": "-", "unclear": "unclear", "not_run": "-"}.get(scope, "unclear")


def records(audit: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for _, r in audit.iterrows():
        cls = str(r["normalization_class"])
        scope, extra = _scope(cls)
        state = str(r["state"])
        out.setdefault(state, []).append({
            "entry": r["detector"], "subset": r["subset"], "code_commit": state.split("(")[-1].rstrip(")"),
            "applied_by": "none" if scope == "none" else "library" if "library" in cls else "detector",
            "scope": scope, "additional": extra, "statistics_from": _statistics_from(cls, scope),
            "fit_score": str(r["train_score_mismatch"]), "detail": str(r["detail"]),
            "evidence": str(r["evidence_file_lines"]), "changed_by_commit": str(r["changed_by_commit"]),
            "confidence": str(r["confidence"])})
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "schema.json").write_text(json.dumps(SCHEMA, indent=1) + "\n")
    recs = records(pd.read_csv(AUDIT))
    summary: dict[str, Any] = {}
    for state, rows in recs.items():
        tag = state.split("(")[0]
        (OUT / f"tsbad_{tag}.json").write_text(json.dumps(rows, indent=1) + "\n")
        summary[state] = {sub: dict(Counter(r["scope"] for r in rows if r["subset"] in (sub, "both")))
                          for sub in ("U", "M")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    LOGGER.info("wrote %s (%s)", OUT, ", ".join(f"{k}: {len(v)}" for k, v in recs.items()))


if __name__ == "__main__":
    main()
