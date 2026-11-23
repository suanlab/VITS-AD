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
    "required": ["entry", "subset", "audited_commit", "results_commit", "applied_by", "scope", "statistics_from",
                 "fit_score_mismatch"],
    "properties": {
        "entry": {"type": "string", "description": "detector or method name as on the leaderboard"},
        "subset": {"type": "string", "description": "table the entry belongs to (e.g. U, M, both)"},
        "audited_commit": {"type": "string", "description": "commit whose code this record describes"},
        "results_commit": {"type": ["string", "null"],
                           "description": "commit of the code that produced the entry's published results; null if "
                                          "not established"},
        "applied_by": {"enum": ["harness", "detector", "library", "none"],
                       "description": "where normalization happens"},
        "scope": {"enum": list(SCOPES), "description": "unit over which statistics are computed"},
        "additional": {"type": "array", "items": {"type": "string"},
                       "description": "further normalization steps, in order"},
        "statistics_from": {"type": "string",
                            "description": "data the statistics come from (window, training part, whole series, ...)"},
        "fit_score_mismatch": {"type": "string",
                               "description": "whether fitting and scoring DIFFER in transform or statistics: starts "
                                              "with 'yes' (they differ), 'no' (they do not), 'n/a' (one pass) or 'unclear', then "
                                              "the audit's explanation"},
        "detail": {"type": "string"},
        "evidence": {"type": "string", "description": "file and line references"},
        "changed_by_commit": {"type": "string"},
        "confidence": {"type": "string"},
        "reads_labels": {"type": "boolean",
                         "description": "whether the scoring step reads the labels of the data it scores"},
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
            "entry": r["detector"], "subset": r["subset"], "audited_commit": state.split("(")[-1].rstrip(")"),
            # The audit could not establish which code state produced each published value (the paper,
            # Sec. 4.1), so no record names one.
            "results_commit": None,
            "applied_by": "none" if scope == "none" else "library" if "library" in cls else "detector",
            "scope": scope, "additional": extra, "statistics_from": _statistics_from(cls, scope),
            "fit_score_mismatch": str(r["train_score_mismatch"]), "detail": str(r["detail"]),
            "evidence": str(r["evidence_file_lines"]), "changed_by_commit": str(r["changed_by_commit"]),
            "confidence": str(r["confidence"])})
    return out


# Second harness (added 2026-11-24 after simulated review round 19): TimeEval's algorithm scripts at
# github.com/TimeEval/TimeEval-algorithms commit 54da80b, audited by reading each algorithm's
# algorithm.py and helper modules. TimeEval's harness passes the dataset's values unchanged, so
# "applied_by" refers to the algorithm script.
TIMEEVAL_COMMIT = "54da80b41fb8afb56f247d58f22412d803b7cbd5"
_PYOD = ("subsequence_lof", "subsequence_if", "subsequence_knn", "knn", "lof", "hbos", "iforest", "cof", "copod",
         "cblof")
TIMEEVAL_AUDIT: dict[str, dict[str, Any]] = {
    **{a: {"applied_by": "none", "scope": "none", "statistics_from": "-", "reads_labels": True,
           "fit_score_mismatch": "n/a (unsupervised; PyOD decision_scores_ from the fit on the scored series)",
           "detail": "raw values (sliding windows for subsequence_*); the labels of the scored file set PyOD's "
                     "contamination, which affects the threshold, not decision_scores_",
           "evidence": f"{a}/algorithm.py: load_data (labels = df.iloc[:, -1]; contamination)"} for a in _PYOD},
    **{a: {"applied_by": "none", "scope": "none", "statistics_from": "-", "reads_labels": False,
           "fit_score_mismatch": "n/a (unsupervised)", "detail": "raw values; no input scaling",
           "evidence": f"{a}/ (algorithm.py and helper modules: no input scaling found)"}
       for a in ("kmeans", "pci", "phasespace_svm", "fft", "dwt_mlead", "if_lof")},
    "median_method": {"applied_by": "detector", "scope": "window", "statistics_from": "each window", "reads_labels": False,
                      "fit_score_mismatch": "n/a (unsupervised)",
                      "detail": "score = distance to the window median divided by the window's standard deviation "
                                "(score-level normalization by local spread)",
                      "evidence": "median_method/median_method.py:36"},
    "triple_es": {"applied_by": "detector", "scope": "series_per_channel", "statistics_from": "whole series",
                  "reads_labels": False, "fit_score_mismatch": "n/a (unsupervised)",
                  "detail": "MinMaxScaler(0.1, 1.1) over the whole scored series (the model needs positive values)",
                  "evidence": "triple_es/algorithm.py:45"},
    **{a: {"applied_by": "none", "scope": "none", "statistics_from": "-", "reads_labels": False,
           "fit_score_mismatch": "no (raw sliding windows in train and execute; the model fitted on the training "
                                 "file is applied to the execute file)",
           "detail": "raw values; no input scaling", "evidence": f"{a}/algorithm.py: train, execute"}
       for a in ("subsequence_fast_mcd", "fast_mcd", "robust_pca")},
}


def timeeval_records() -> list[dict[str, Any]]:
    return [{"entry": a, "subset": "univariate input", "audited_commit": TIMEEVAL_COMMIT, "results_commit": None,
             "additional": [], "changed_by_commit": "n/a", "confidence": "high", **v}
            for a, v in sorted(TIMEEVAL_AUDIT.items())]


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
    te = timeeval_records()
    (OUT / "timeeval.json").write_text(json.dumps(te, indent=1) + "\n")
    summary["TimeEval-algorithms@54da80b"] = {"scope": dict(Counter(r["scope"] for r in te)),
                                              "reads_labels": sum(r["reads_labels"] for r in te)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    LOGGER.info("wrote %s (%s)", OUT, ", ".join(f"{k}: {len(v)}" for k, v in recs.items()))


if __name__ == "__main__":
    main()
