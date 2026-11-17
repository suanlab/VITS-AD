"""Exploratory follow-ups to the registered replication prediction M (not registered; written
2026-11-17 after simulated review round 18, with M's results known).

On the same instances as scripts/replication_prediction.py (outcome O1), this scores:
  * variants that separate p_new's two changes from p_iid:
      p_mu_novar  centre: random-effects mean mu; variance: se_mu^2 + sigma_w^2/n (no tau^2)
      p_d_revar   centre: |d| (series mean, as p_iid); variance: tau^2 + se_mu^2 + sigma_w^2/n
      p_new_t     p_new with a t distribution on k-2 degrees of freedom (k discovery sources with
                  at least two series; the normal when k < 3)
  * every prediction per held-out source (TimeEval: per collection), and the instances where mu
    and d have opposite signs.
p_new and p_iid reproduce the registered O1 scores, which serves as a check.
The registered decision is unaffected; this script does not write replication_prediction.json.

Output: artifacts/official_protocol/replication_prediction_explore.json

    python -m scripts.replication_prediction_explore --timeeval data/timeeval_results/timeeval_results.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from scripts.replication_prediction import (
    ART,
    N_DISCOVERY_COLLECTIONS,
    fit,
    predict,
    rt_tables,
    source_of,
    timeeval_table,
)

LOGGER = logging.getLogger(__name__)
OUT = ART / "replication_prediction_explore.json"
PREDICTIONS = ("p_new", "p_iid", "p_mu_novar", "p_d_revar", "p_new_t")


def variants(f: dict[str, float], n_new: int) -> dict[str, float]:
    s = 1.0 if f["d"] >= 0 else -1.0
    base = predict(f, n_new, 1.0)
    within = f["se_mu"] ** 2 + f["sigma2_w"] / max(1, n_new)
    full = f["tau2"] + within
    dof = f["k_sources"] - 2
    return {"p_new": base["p_new"], "p_iid": base["p_iid"],
            "p_mu_novar": float(stats.norm.cdf(s * f["mu"] / np.sqrt(within))) if within > 0 else float(s * f["mu"] > 0),
            "p_d_revar": float(stats.norm.cdf(abs(f["d"]) / np.sqrt(full))) if full > 0 else 1.0,
            "p_new_t": (float(stats.t.cdf(s * f["mu"] / np.sqrt(full), dof)) if dof >= 1 else base["p_new"])
            if full > 0 else base["p_new"],
            "sign": s, "mu_sign_differs": bool(np.sign(f["mu"]) != s)}


def o1_rows(disc: pd.DataFrame, hold: pd.DataFrame, src_d: pd.Series, src_h: pd.Series,
            pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """O1 instances as in replication_prediction.instances, with the variant predictions."""
    rows = []
    for a, b in pairs:
        dd = (disc[a] - disc[b]).dropna()
        if len(dd) < 3:
            continue
        f = fit(dd, src_d)
        if f["d"] == 0:
            continue
        hd = (hold[a] - hold[b]).dropna()
        if hd.empty:
            continue
        hs = src_h.reindex(hd.index)
        for src, grp in hd.groupby(hs):
            v = variants(f, len(grp))
            rows.append({"source": str(src), **v, "y": int(np.sign(grp.mean()) == v["sign"])})
    return rows


def _scores(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"n": int(len(df)), "share_held": float(df["y"].mean())}
    for col in PREDICTIONS:
        p = df[col].clip(1e-6, 1 - 1e-6)
        r = {"brier": float(((p - df["y"]) ** 2).mean())}
        if df["y"].nunique() == 2:
            r["auc"] = float(roc_auc_score(df["y"], p))
        out[col] = r
    flip = df[df["mu_sign_differs"]]
    out["mu_sign_differs"] = {"n": int(len(flip)), "share_held": float(flip["y"].mean()) if len(flip) else None}
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    return {"all": _scores(df), "by_heldout_source": {s: _scores(g) for s, g in df.groupby("source")}}


def run(timeeval: Path) -> dict[str, Any]:
    res: dict[str, Any] = {}
    t = rt_tables()
    pairs = list(itertools.combinations(t["tuning"].columns, 2))
    for split in ("eva", "rest"):
        res[f"rt_{split}"] = summarize(o1_rows(t["tuning"], t[split], source_of(t["tuning"].index),
                                               source_of(t[split].index), pairs))
    wide, src = timeeval_table(timeeval, "ROC_AUC")
    cols = sorted(src.unique())
    pairs = list(itertools.combinations(wide.columns, 2))
    rows = []
    for disc in itertools.combinations(cols, N_DISCOVERY_COLLECTIONS):
        m = src.isin(disc)
        rows += o1_rows(wide[m], wide[~m], src[m], src[~m], pairs)
    res["timeeval"] = summarize(rows)
    return res


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeeval", type=Path, required=True, help="TimeEval's published result table (CSV)")
    args = ap.parse_args()
    OUT.write_text(json.dumps(run(args.timeeval), indent=1) + "\n")
    LOGGER.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
