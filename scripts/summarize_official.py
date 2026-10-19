"""Compare our arms with every TSB-AD detector on TSB-AD's evaluation split, same protocol.

Inputs: artifacts/official_protocol/<subset>/per_series_{raw,vision}_shard*.json (our arms; put
there by collect_official_artifacts.py) and
data/tsb_ad/official/{uni,multi}_mergedTable_VUS-PR.csv (TSB-AD's published per-series VUS-PR
for 32 univariate / 23 multivariate detectors, computed with the same get_metrics code).

Analysis plan, fixed before looking at the merged results:
  1. Leaderboard: mean VUS-PR and mean per-series rank of every method, per subset, on the
     series where all our arms produced a score.
  2. Primary family (Holm-corrected, 4 tests per subset = 8 tests): rendered vs raw_flatten,
     rendered vs raw_meanpool, raw_flatten vs raw_meanpool, raw_flatten vs the best TSB-AD
     detector by mean VUS-PR on that subset. Paired Wilcoxon + 95% bootstrap CI of the mean.
  3. Secondary (reported uncorrected, labelled exploratory): rendered and raw_flatten vs each
     TSB-AD detector.
  4. Later leaderboard submissions (benchmark_exp/leaderboard_results), descriptive: mean VUS-PR
     and raw_flatten minus each, on the series every entry scored.
Writes artifacts/official_protocol/summary_<variant>.json. Reads nothing outside artifacts/ and
data/tsb_ad/official/, so it runs on a clean clone after fetch_tsbad_official.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

LOGGER = logging.getLogger(__name__)
RES = Path("artifacts/official_protocol")  # per-series inputs (see collect_official_artifacts)
OFF = Path("data/tsb_ad/official")
ART = Path("artifacts/official_protocol")
META = {"ts_len", "anomaly_len", "num_anomaly", "avg_anomaly_len", "anomaly_ratio",
        "point_anomaly", "seq_anomaly"}
TABLES = {"TSB-AD-U": "uni_mergedTable_VUS-PR.csv", "TSB-AD-M": "multi_mergedTable_VUS-PR.csv"}
OURS = ("raw_meanpool", "raw_flatten", "vision")


def _ours(subset: str, tags: tuple[str, ...]) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    files = [f for tag in tags for f in sorted((RES / subset).glob(f"per_series_{tag}_shard*.json"))]
    for f in files:
        for r in json.loads(f.read_text()):
            for arm in OURS:
                if r.get(arm):
                    rows.setdefault(r["series"], {})[arm] = r[arm]["VUS-PR"]
    return pd.DataFrame.from_dict(rows, orient="index")


def _paired(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    d = a - b
    rng = np.random.default_rng(0)
    boot = [d[rng.integers(0, d.size, d.size)].mean() for _ in range(10000)]
    return {"n": int(d.size), "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "delta": float(d.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "wins": int((d > 0).sum()), "ties": int((d == 0).sum()),
            "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0}


def _holm(tests: dict[str, dict[str, Any]]) -> None:
    items = sorted(tests.items(), key=lambda kv: kv[1]["wilcoxon_p"])
    running, m = 0.0, len(items)
    for i, (_, t) in enumerate(items):
        running = min(1.0, max(running, (m - i) * t["wilcoxon_p"]))
        t["p_holm"] = running


LEADERBOARD_PREFIX = {"TSB-AD-U": "Uni", "TSB-AD-M": "Multi"}


def _leaderboard(subset: str) -> pd.DataFrame:
    """Per-series VUS-PR of later TSB-AD leaderboard submissions (heterogeneous file formats)."""
    cols: dict[str, pd.Series] = {}
    for f in sorted((OFF / "leaderboard").glob(f"{LEADERBOARD_PREFIX[subset]}_*.csv")):
        d = pd.read_csv(f)
        d = d.set_index("file" if "file" in d.columns else "filename")
        if "VUS-PR" in d.columns:
            cols[f.stem.split("_", 1)[1]] = d["VUS-PR"]
            continue
        names = [c for c in d.columns if c not in META]
        if any(c.startswith("TSPulse") for c in names):
            names = [c for c in names if c.startswith("TSPulse")]
        for c in names:
            cols[c] = d[c]
    return pd.DataFrame(cols).apply(pd.to_numeric, errors="coerce")


def analyse(subset: str, tags: tuple[str, ...]) -> dict[str, Any]:
    off = pd.read_csv(OFF / TABLES[subset]).set_index("file")
    detectors = [c for c in off.columns if c not in META]
    ours = _ours(subset, tags)
    present = [a for a in OURS if a in ours.columns]
    ours = ours[present].dropna()
    merged = ours.join(off[detectors], how="inner").dropna()
    methods = present + detectors
    ranks = merged[methods].rank(axis=1, ascending=False, method="average")
    board = pd.DataFrame({"mean_vus_pr": merged[methods].mean(), "mean_rank": ranks.mean()})
    board = board.sort_values("mean_vus_pr", ascending=False)
    best = board.loc[detectors].sort_values("mean_vus_pr", ascending=False).index[0]
    plan = [("vision", "raw_flatten"), ("vision", "raw_meanpool"), ("raw_flatten", "raw_meanpool"),
            ("raw_flatten", best)]
    primary = {f"{a} vs {b}" + (" (best TSB-AD)" if b == best else ""):
               _paired(merged[a].values, merged[b].values)
               for a, b in plan if a in merged.columns and b in merged.columns}
    secondary = {f"{arm} vs {det}": _paired(merged[arm].values, merged[det].values)
                 for arm in ("vision", "raw_flatten") if arm in merged.columns for det in detectors}
    lb = _leaderboard(subset).reindex(merged.index)
    lb = lb.loc[:, lb.notna().all()]          # only entries scored on every series of the split
    later = {}
    for name in lb.columns:
        later[name] = {"mean_vus_pr": float(lb[name].mean()),
                       "raw_flatten_minus": _paired(merged["raw_flatten"].values, lb[name].values)}
    return {"n_series": int(len(merged)), "n_detectors": len(detectors), "best_tsbad": best,
            "later_leaderboard": later,
            "leaderboard": board.round(4).reset_index(names="method").to_dict(orient="records"),
            "primary": primary, "secondary_exploratory": secondary}


VARIANTS = {"default": ("raw", "vision"), "crossfit": ("rawcf", "visioncf")}


def stride_check() -> None:
    """Default (interpolated) scoring vs scoring every window at stride 1, raw_flatten, univariate."""
    sub = "TSB-AD-U"
    base = _ours(sub, ("raw",))["raw_flatten"]
    s1 = _ours(sub, ("stride1",))
    if s1.empty:
        return
    both = pd.concat([base, s1["raw_flatten"]], axis=1, keys=["default", "stride1"]).dropna()
    d = both["stride1"] - both["default"]
    res = {"n": int(len(both)), "mean_default": float(both["default"].mean()),
           "mean_stride1": float(both["stride1"].mean()), "mean_diff": float(d.mean()),
           "max_abs_diff": float(d.abs().max()),
           "spearman": float(stats.spearmanr(both["default"], both["stride1"]).correlation)}
    (ART / "stride_check.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("stride check (%s, n=%d): mean diff %+.4f, Spearman %.3f", sub, res["n"],
                res["mean_diff"], res["spearman"])


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="default",
                    help="default: in-sample train part (TSB-AD protocol); crossfit: train part "
                         "scored out of sample")
    args = ap.parse_args()
    tags = VARIANTS[args.variant]
    out = {s: analyse(s, tags) for s in TABLES}
    family = {f"{s}: {k}": v for s in out for k, v in out[s]["primary"].items()}
    _holm(family)
    ART.mkdir(parents=True, exist_ok=True)
    (ART / f"summary_{args.variant}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    stride_check()
    for s, res in out.items():
        LOGGER.info("== %s: %d series, best TSB-AD detector %s", s, res["n_series"], res["best_tsbad"])
        for row in res["leaderboard"][:8]:
            LOGGER.info("   %-20s VUS-PR %.4f  rank %.2f", row["method"], row["mean_vus_pr"], row["mean_rank"])
        for name, row in sorted(res["later_leaderboard"].items(), key=lambda kv: -kv[1]["mean_vus_pr"]):
            d = row["raw_flatten_minus"]
            LOGGER.info("   later: %-14s %.4f  flatten-minus %+.4f CI[%+.3f,%+.3f]", name,
                        row["mean_vus_pr"], d["delta"], d["ci95"][0], d["ci95"][1])
        for k, t in res["primary"].items():
            LOGGER.info("   %-45s d=%+.4f CI[%+.3f,%+.3f] p=%.2g p_holm=%.2g", k, t["delta"],
                        t["ci95"][0], t["ci95"][1], t["wilcoxon_p"], t["p_holm"])


if __name__ == "__main__":
    main()
