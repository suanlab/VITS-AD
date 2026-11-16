"""Registered M: does between-source heterogeneity predict which benchmark conclusions replicate on
new sources? (artifacts/preregistration/replication_prediction.md)

A conclusion is a pairwise order "a above b" read from a discovery set (mean over its series). For
each, three predicted probabilities that the order holds on new data:

  p_new  random-effects prediction for a new source (DerSimonian-Laird over discovery sources with
         at least two series): Phi(s * mu / sqrt(tau^2 + se_mu^2 + sigma_w^2 / n_new))
  p_iid  same-population replication (no between-source term): Phi(|d| / sqrt(se_iid^2 + sigma^2 / n_new))
  p_sig  the leaderboard's significance view: Phi(|d| / se_iid)

with d the discovery mean difference, s = sign(d), se_iid its i.i.d. standard error, sigma_w^2 the
pooled within-source variance and sigma^2 the series variance. Outcomes:

  O1  per (conclusion, held-out source): the source's mean difference has the sign s (n_new = its series)
  O2  per conclusion: the held-out set's series-weighted mean difference has the sign s
      (for p_new the between-source term is tau^2 * sum_k w_k^2 over held-out sources, n_new = all series)

Targets:
  rt_eva, rt_rest  TSB-AD-U: discovery = the 48 tuning series with the configurations RT selected;
                   held out = the evaluation split (350) and the 472 series in neither split
  timeeval         TimeEval's published univariate results (ROC_AUC): algorithms with a result on at
                   least 90% of the univariate datasets; every split of the 9 collections into 5
                   discovery and 4 held-out collections (126 splits)

Scores: Brier score and ROC-AUC of each prediction against the outcomes, and the calibration slope
(logistic regression of the outcome on logit p).

    python -m scripts.replication_prediction --timeeval data/timeeval_results/timeeval_results.csv
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

LOGGER = logging.getLogger(__name__)
ART = Path("artifacts/official_protocol")
OUT = ART / "replication_prediction.json"
COVERAGE = 0.9
N_DISCOVERY_COLLECTIONS = 5


def fit(diff: pd.Series, src: pd.Series) -> dict[str, float]:
    """Discovery statistics of one conclusion: series mean and i.i.d. SE, and a DerSimonian-Laird
    random-effects fit over sources with at least two series (tau^2 = 0 if fewer than two)."""
    d = diff.dropna()
    s = src.reindex(d.index)
    n = len(d)
    out = {"d": float(d.mean()), "n": n, "sigma2": float(d.var(ddof=1)) if n > 1 else 0.0}
    out["se_iid"] = float(np.sqrt(out["sigma2"] / n)) if n > 1 else np.inf
    g = d.groupby(s)
    m, v, k = g.mean(), g.var(ddof=1), g.size()
    keep = k >= 2
    m, v, k = m[keep], v[keep], k[keep]
    out["k_sources"] = int(keep.sum())
    out["sigma2_w"] = float((v * (k - 1)).sum() / max(1, (k - 1).sum())) if len(k) else out["sigma2"]
    if len(m) < 2:
        out.update(mu=out["d"], tau2=0.0, se_mu=out["se_iid"])
        return out
    se2 = (v / k).clip(lower=1e-12)
    w = 1 / se2
    fixed = float((w * m).sum() / w.sum())
    q = float((w * (m - fixed) ** 2).sum())
    tau2 = max(0.0, (q - (len(m) - 1)) / float(w.sum() - (w ** 2).sum() / w.sum()))
    wr = 1 / (se2 + tau2)
    out.update(mu=float((wr * m).sum() / wr.sum()), tau2=tau2, se_mu=float(np.sqrt(1 / wr.sum())))
    return out


def predict(f: dict[str, float], n_new: int, between: float) -> dict[str, float]:
    """Predicted probabilities that the discovery order (sign of d) holds on new data of n_new series;
    ``between`` multiplies tau^2 (1 for one new source, sum of squared weights for a pooled set)."""
    s = 1.0 if f["d"] >= 0 else -1.0
    var_new = f["tau2"] * between + f["se_mu"] ** 2 + f["sigma2_w"] / max(1, n_new)
    var_iid = f["se_iid"] ** 2 + f["sigma2"] / max(1, n_new)
    return {"p_new": float(stats.norm.cdf(s * f["mu"] / np.sqrt(var_new))) if var_new > 0 else float(s * f["mu"] > 0),
            "p_iid": float(stats.norm.cdf(abs(f["d"]) / np.sqrt(var_iid))) if var_iid > 0 else 1.0,
            "p_sig": float(stats.norm.cdf(abs(f["d"]) / f["se_iid"])) if f["se_iid"] > 0 else 1.0,
            "sign": s}


def instances(disc: pd.DataFrame, hold: pd.DataFrame, src_d: pd.Series, src_h: pd.Series,
              pairs: list[tuple[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """O1 and O2 instances (predictions and outcomes) for the given conclusions."""
    o1, o2 = [], []
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
        sizes = hs.value_counts()
        w = sizes / sizes.sum()
        p = predict(f, len(hd), float((w ** 2).sum()))
        o2.append({"a": a, "b": b, **p, "y": int(np.sign(hd.mean()) == p["sign"])})
        for src, grp in hd.groupby(hs):
            q = predict(f, len(grp), 1.0)
            o1.append({"a": a, "b": b, "source": str(src), **q, "y": int(np.sign(grp.mean()) == q["sign"])})
    return o1, o2


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    out: dict[str, Any] = {"n": int(len(df)), "share_replicated": float(df["y"].mean()) if len(df) else None}
    if df.empty:
        return out
    for col in ("p_new", "p_iid", "p_sig"):
        p = df[col].clip(1e-6, 1 - 1e-6)
        res = {"brier": float(((p - df["y"]) ** 2).mean())}
        if df["y"].nunique() == 2:
            res["auc"] = float(roc_auc_score(df["y"], p))
            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(C=1e6).fit(np.log(p / (1 - p)).to_numpy()[:, None], df["y"])
            res["calibration_slope"] = float(lr.coef_[0][0])
        out[col] = res
    return out


def decision(per_target: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Registered rule on O1: p_new against p_iid (primary baseline)."""
    better, worse = [], []
    for t, r in per_target.items():
        s = r["O1"]
        if "auc" not in s.get("p_new", {}) or "auc" not in s.get("p_iid", {}):
            continue
        db = s["p_new"]["brier"] - s["p_iid"]["brier"]
        da = s["p_new"]["auc"] - s["p_iid"]["auc"]
        if db <= -0.02 and da >= 0.05:
            better.append(t)
        if db >= -0.005 and da < 0.02:
            worse.append(t)
    n = len(per_target)
    verdict = "supported" if len(better) >= 2 else "refuted" if len(worse) == n else "inconclusive"
    return {"targets_better": better, "targets_no_gain": worse, "decision": verdict}


def rt_tables() -> dict[str, pd.DataFrame]:
    """Per-series VUS-PR of the twelve re-tuned variants (detector|variant) on tuning, eva and rest."""
    from scripts.paper_numbers import SUB_DETECTORS, keyed
    from scripts.run_retune import config_key

    sel = json.loads((ART / "TSB-AD-U" / "retune_selection.json").read_text())
    grid: dict[str, dict[str, Any]] = {}
    for f in sorted((ART / "TSB-AD-U").glob("retune_grid_shard*.json")):
        for r in json.loads(f.read_text()):
            grid.setdefault(r["series"], {}).update(r)
    tun = pd.DataFrame({f"{d}|{v}": {s: r.get(f"{d}|{v}|{config_key(sel[f'{d}|{v}']['params'])}")
                                       for s, r in grid.items()}
                        for d in SUB_DETECTORS for v in ("window", "global")}).astype(float)
    out = {"tuning": tun}
    for split, base_pat in (("eva", "TSB-AD-U/detnorm_shard*.json"), ("rest", "TSB-AD-U/detnorm_rest_shard*.json")):
        base, new = keyed(base_pat), keyed(f"TSB-AD-U/retune_{split}_shard*.json")
        cols = {}
        for d in SUB_DETECTORS:
            for v in ("window", "global"):
                k = f"{d}|{v}"
                cols[k] = base[k] if sel[k]["equals_tsbad_tuned"] else new[k].reindex(base.index)
        out[split] = pd.DataFrame(cols)
    return out


def source_of(index: pd.Index) -> pd.Series:
    return pd.Series([s.split("_")[1] for s in index], index=index)


def run_rt() -> dict[str, Any]:
    t = rt_tables()
    pairs = list(itertools.combinations(t["tuning"].columns, 2))
    res = {}
    for split in ("eva", "rest"):
        o1, o2 = instances(t["tuning"], t[split], source_of(t["tuning"].index), source_of(t[split].index), pairs)
        res[f"rt_{split}"] = {"O1": score(o1), "O2": score(o2), "n_conclusions": len(o2)}
    return res


def timeeval_table(path: Path, metric: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    uni = df[df.dataset_input_dimensionality == "UNIVARIATE"]
    ok = uni[uni.status == "Status.OK"].dropna(subset=[metric])
    wide = ok.pivot_table(index=["collection", "dataset"], columns="algorithm", values=metric, aggfunc="first")
    n_ds = uni.drop_duplicates(["collection", "dataset"]).shape[0]
    keep = [a for a in wide.columns if wide[a].notna().sum() >= COVERAGE * n_ds]
    wide = wide[keep]
    wide.index = [f"{c}::{d}" for c, d in wide.index]
    src = pd.Series([i.split("::")[0] for i in wide.index], index=wide.index)
    return wide, src


def run_timeeval(path: Path, metric: str = "ROC_AUC") -> dict[str, Any]:
    wide, src = timeeval_table(path, metric)
    cols = sorted(src.unique())
    pairs = list(itertools.combinations(wide.columns, 2))
    o1_all, o2_all = [], []
    for disc in itertools.combinations(cols, N_DISCOVERY_COLLECTIONS):
        dmask = src.isin(disc)
        o1, o2 = instances(wide[dmask], wide[~dmask], src[dmask], src[~dmask], pairs)
        o1_all += o1
        o2_all += o2
    return {"timeeval": {"metric": metric, "n_algorithms": int(wide.shape[1]), "n_datasets": int(wide.shape[0]),
                         "n_collections": len(cols), "n_splits": int(len(list(itertools.combinations(cols, N_DISCOVERY_COLLECTIONS)))),
                         "O1": score(o1_all), "O2": score(o2_all), "n_conclusions": len(pairs)}}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--timeeval", type=Path, required=True, help="TimeEval's published result table (CSV)")
    args = ap.parse_args()
    per = {**run_rt(), **run_timeeval(args.timeeval)}
    secondary = run_timeeval(args.timeeval, "PR_AUC")["timeeval"]
    out = {"targets": per, "decision": decision(per), "secondary_timeeval_pr_auc": secondary}
    OUT.write_text(json.dumps(out, indent=1, default=float) + "\n")
    LOGGER.info("wrote %s: %s", OUT, out["decision"])


if __name__ == "__main__":
    main()
