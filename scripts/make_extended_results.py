"""Write artifacts/EXTENDED_RESULTS.md: results moved out of the PVLDB manuscript.

Every number comes from artifacts/official_protocol/paper_numbers.json (or, for the table
fragments, from scripts/make_paper_tables.py, which reads the same file); the prose holds no
numbers. Long per-series lists are left out; the JSON key of each section is given so that the
full values can be looked up.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from scripts import make_paper_tables as mpt

ART = Path("artifacts/official_protocol")
MAX_LIST = 12
MAX_ROWS = 160

# (title, explanation, paper_numbers key, sub-keys to keep or None for all)
SECTIONS: list[tuple[str, str, str, list[str] | None]] = [
    ("Training-free baselines: cost per series", "Median fitting-plus-scoring time per series (seconds) over repeats.",
     "cost_median_repeat", None),
    ("Factorial from Sub-MCD to Raw-LW", "Cell means of normalization x estimator x window on the series complete in "
     "all cells, the anchor cell that re-implements Sub-MCD, and order-averaged (Shapley) contributions.",
     "factorial", ["n_series", "n_complete", "cell_means", "anchor", "total", "shapley"]),
    ("Amplitude terciles on the 472 held-out series (exploratory)", "Global minus per-window by tercile of the "
     "amplitude ratio, cut at the evaluation split's boundaries; counts and median ratio by source.",
     "held_out_472_regime", None),
    ("Drift: ranking", "Raw-1NN VUS-PR under injected benign drift, and differences to no drift.", "drift", None),
    ("Drift: false-alarm rate (registered H5)", "FAR at a threshold calibrated on out-of-sample training scores.",
     "far", None),
    ("Follow-ups to B1-B4 (registered F1, F2)", "Interior AUC against far-away normal points (F1); AUC-PR deficit by "
     "duration at fixed prevalence (F2).", "boundary_followup", None),
    ("Registered H1, H2, H4, H8", "Statistics behind the registered decisions.", "preregistered", None),
    ("H10 families on both series sets", "Matrix Profile and KMeansAD, global minus per-window.", "family_toggle", None),
    ("Other measures on held-out sets (not registered)", "Global minus per-window under AUC-PR, AUC-ROC and VUS-ROC.",
     "other_measures_held_out", None),
    ("Chance floor", "VUS-PR of a constant score by source.", "chance_floor", None),
    ("Census controls (exploratory, after review round 18)", "Reversals on sources with at least five series, a "
     "permutation null over source labels, and the total-variation distance between TSB-AD's split compositions.",
     "census_controls", None),
    ("Toggle by anomaly type (exploratory, after review round 18)", "Global minus per-window by point, sequence or "
     "mixed anomaly series, and a type-stratified prediction of the held-out means.", "anomaly_type", None),
    ("Rank uncertainty under source resampling", "Rank intervals and top-k shares.", "rank_bootstrap", None),
    ("Seeds", "Randomized detectors rerun with two other seeds.", "seed_robustness", None),
    ("Contaminated training parts", "Main comparisons without series whose training part has labels.",
     "contamination", None),
    ("TSB-AD's later code", "Ranking with the later-code values replacing the published ones.",
     "later_code_ranking", None),
    ("Raw-1NN on the 472 held-out series (exploratory)", "Global minus per-window, by source and source-weighted.",
     "baselines_held_out", ["series", "failed", "means", "raw_1nn_global_minus_window", "by_source",
                            "source_weighted_mean", "ci_source_cluster"]),
]

TABLES: list[tuple[str, str, list[str]]] = [
    ("Mean VUS-PR on the evaluation split (Raw-1NN minus row, paired, 95% CI)", "bench",
     ["method", "VUS-PR", "Delta"]),
    ("Factorial cell means (dagger: re-implementation of Sub-MCD)", "factorial",
     ["normalization", "Ledoit-Wolf, W=100", "Ledoit-Wolf, period", "MCD, W=100", "MCD, period"]),
    ("Per-window minus global by amplitude tercile (evaluation split; per-window wins in parentheses)", "regime",
     ["detector", "low z_r", "mid", "high z_r", "Spearman rho"]),
    ("Raw-1NN at thresholds (calibrated: FAR / segment recall; at FAR 1%: segment recall, uses test labels)",
     "threshold", ["drift", "calibrated global", "calibrated per-window", "calibrated trailing",
                   "FAR 1% global", "FAR 1% per-window", "FAR 1% trailing"]),
    ("Scaling grid", "scaling", ["method", "reference windows", "scored windows", "channels", "seconds",
                                 "peak RSS (GB)"]),
]


def _fmt(v: Any) -> str:
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.3g}" if abs(v) < 1e-3 and v != 0 else f"{v:.3f}"
    if isinstance(v, list):
        shown = ", ".join(_fmt(x) for x in v[:MAX_LIST])
        return f"[{shown}{', ...' if len(v) > MAX_LIST else ''}]"
    return str(v)


def _flatten(x: Any, prefix: str = "") -> list[tuple[str, str]]:
    if isinstance(x, dict):
        out: list[tuple[str, str]] = []
        for k, v in x.items():
            out += _flatten(v, f"{prefix}.{k}" if prefix else str(k))
        return out
    if isinstance(x, list) and any(isinstance(v, (dict, list)) for v in x):
        if len(x) > MAX_LIST:
            return [(prefix, f"({len(x)} entries; see JSON)")]
        out = []
        for i, v in enumerate(x):
            out += _flatten(v, f"{prefix}[{i}]")
        return out
    if isinstance(x, list) and len(x) > MAX_LIST:
        return [(prefix, f"({len(x)} values; see JSON)")]
    return [(prefix, _fmt(x))]


def _tex_to_md(rows: str) -> list[str]:
    out = []
    for ln in rows.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("\\midrule"):
            continue
        ln = ln.rstrip("\\").strip()
        for a, b in (("$", ""), ("\\to", "->"), ("\\,", " "), ("{,}", ","), ("\\pm", "+/-"), ("\\%", "%"),
                     ("^{\\dagger}", " (dagger)"), ("^{*}", " (*)"), ("\\multicolumn{2}{c}", "")):
            ln = ln.replace(a, b)
        ln = re.sub(r"\\multicolumn\{\d+\}\{(?:[^{}]|\{\})*\}\{(.*)\}$", r"\1", ln)
        ln = re.sub(r"\\emph\{([^}]*)\}", r"\1", ln).replace("\\times", "x").replace("{", "").replace("}", "")
        out.append("| " + " | ".join(c.strip() for c in ln.split("&")) + " |")
    return out


def build(n: dict[str, Any]) -> str:
    lines = ["# Extended results", "",
             "Results referred to in the PVLDB manuscript as being in the artifact. Generated by",
             "`scripts/make_extended_results.py` from `artifacts/official_protocol/paper_numbers.json`;",
             "do not edit by hand. Registered outcomes and their decision statistics are in",
             "`artifacts/preregistration/results.md`.", ""]
    frags = {
        "bench": mpt.table_bench(n).replace("TSB-AD-U|", "U: ").replace("TSB-AD-M|", "M: "),
        "factorial": mpt.table_factorial(n),
        "regime": mpt.table_regime(n),
        "threshold": mpt.table_threshold(n),
        "scaling": mpt.table_scaling(n) if "scaling" in n else "",
    }
    lines += ["## Tables", ""]
    for title, key, head in TABLES:
        rows = _tex_to_md(frags[key])
        if not rows:
            continue
        lines += [f"### {title}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + rows + [""]
    lines += ["## Values by analysis", ""]
    for title, text, key, keep in SECTIONS:
        if key not in n:
            continue
        v = n[key]
        if keep is not None and isinstance(v, dict):
            v = {k: v[k] for k in keep if k in v}
        flat = _flatten(v)
        lines += [f"### {title}", "", f"{text} JSON key: `{key}`.", "", "| key | value |", "|---|---|"]
        lines += [f"| {k.replace('|', chr(92) + '|')} | {val.replace('|', chr(92) + '|')} |" for k, val in flat[:MAX_ROWS]]
        if len(flat) > MAX_ROWS:
            lines.append(f"| ... | {len(flat) - MAX_ROWS} more; see JSON |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("artifacts/EXTENDED_RESULTS.md"))
    args = ap.parse_args()
    n = json.loads((ART / "paper_numbers.json").read_text())
    args.out.write_text(build(n))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
