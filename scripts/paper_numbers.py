"""Every number reported in the ICDE paper, computed from artifacts/ only.

Reads per-series results under artifacts/official_protocol/ (put there by
collect_official_artifacts.py) and TSB-AD's published tables under data/tsb_ad/official/
(fetched and hash-checked by fetch_tsbad_official.py). The amplitude ratio z_r needs the raw
TSB-AD series; it is computed once and cached in artifacts/official_protocol/z_ratio.json, so a
clean clone regenerates everything without the raw data.

Sections of the output (artifacts/official_protocol/paper_numbers.json):
  toggle        TSB-AD's subsequence detectors, per-window vs global (Sec. IV, Table I)
  pinned        the same detectors run unmodified at TSB-AD's pinned commit (Sec. IV-A)
  baselines     Raw-1NN / Raw-LW vs published and later leaderboard entries (Sec. V, Table II)
  factorial     3x2x2 design, anchor reproduction, Shapley contributions (Sec. VI-A)
  regime        per-window minus global by amplitude tercile, with CIs (Sec. VI-B)
  render        rendered pipeline case study and tuning-split margins (Sec. VII)
  drift         benign-drift injection (Sec. VIII)
  held_out      pre-registered H6 on TSB-AD's tuning split (Sec. IV)
  stride, cost  copied from stride_check.json and cost.json

    python -m scripts.paper_numbers
"""

from __future__ import annotations

import glob
import itertools
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from scripts.summarize_official import _holm, _leaderboard, _paired

LOGGER = logging.getLogger(__name__)
ART = Path("artifacts/official_protocol")
OFF = Path("data/tsb_ad/official")
DATA = Path("data/tsb_ad")
TABLES = {"TSB-AD-U": "uni_mergedTable_VUS-PR.csv", "TSB-AD-M": "multi_mergedTable_VUS-PR.csv"}
META = {"ts_len", "anomaly_len", "num_anomaly", "avg_anomaly_len", "anomaly_ratio", "period",
        "point_anomaly", "seq_anomaly"}
SUB_DETECTORS = ("Sub_KNN", "Sub_MCD", "Sub_OCSVM", "Sub_HBOS", "Sub_IForest", "Sub_LOF")
H9_SERIES = 472  # univariate series in neither TSB-AD split (heldout472_hypothesis.md)
MULTI_DETECTORS = ("KNN", "MCD", "OCSVM", "HBOS", "IForest", "LOF", "PCA")
MULTI_REPRODUCED = ("KNN", "LOF", "HBOS", "OCSVM")


# ----------------------------------------------------------------------------------- loading
def _records(pattern: str) -> list[dict[str, Any]]:
    return [r for f in sorted(glob.glob(str(ART / pattern))) for r in json.loads(Path(f).read_text())]


def arms(subset: str, tag: str, names: tuple[str, ...], prefix: str = "") -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    for r in _records(f"{subset}/per_series_{tag}_shard*.json"):
        for a in names:
            if r.get(a):
                rows.setdefault(r["series"], {})[prefix + a] = r[a]["VUS-PR"]
    return pd.DataFrame.from_dict(rows, orient="index")


KEY_CONFLICTS: list[dict[str, Any]] = []


def keyed(pattern: str, value: str = "VUS-PR") -> pd.DataFrame:
    """Per-series values from all files matching pattern. A series in several files (a shard and its
    continuation) must carry the same value; any disagreement is recorded in KEY_CONFLICTS."""
    rows: dict[str, dict[str, float]] = {}
    for r in _records(pattern):
        for k, v in r.items():
            if k in ("series", "n", "n_train", "sliding_window", "period3", "n_channels"):
                continue
            if isinstance(v, dict):
                new = float(v[value])
            elif isinstance(v, (int, float)) and v is not None:
                new = float(v)
            else:
                continue
            old = rows.setdefault(r["series"], {}).get(k)
            if old is not None and not (np.isnan(old) and np.isnan(new)) and abs(old - new) > 1e-9:
                KEY_CONFLICTS.append({"pattern": pattern, "series": r["series"], "key": k,
                                      "values": [old, new]})
            rows[r["series"]][k] = new
    return pd.DataFrame.from_dict(rows, orient="index")


def published(subset: str) -> pd.DataFrame:
    pub = pd.read_csv(OFF / TABLES[subset]).set_index("file")
    return pub[[c for c in pub.select_dtypes("number").columns if c not in META]]


def pair(df: pd.DataFrame, a: str, b: str) -> dict[str, Any] | None:
    if a not in df or b not in df:
        return None
    sub = df[[a, b]].dropna()
    return _paired(sub[a].to_numpy(), sub[b].to_numpy()) if len(sub) else None


def _boot_ci(d: np.ndarray, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    boot = [d[rng.integers(0, d.size, d.size)].mean() for _ in range(10000)]
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


# ----------------------------------------------------------------------------------- sections
def toggle(subset: str) -> dict[str, Any]:
    det = keyed(f"{subset}/detnorm_shard*.json")
    if det.empty:
        return {}
    pub = published(subset)
    names = SUB_DETECTORS + ("Sub_PCA",) if subset == "TSB-AD-U" else MULTI_DETECTORS
    # Holm families as reported: the six per-window Sub- detectors (univariate), and the four
    # multivariate detectors whose per-window variant reproduces TSB-AD's published values.
    holm_set = set(SUB_DETECTORS) if subset == "TSB-AD-U" else set(MULTI_REPRODUCED)
    out: dict[str, Any] = {"detectors": {}}
    family: dict[str, dict[str, Any]] = {}
    for d in names:
        col = d.replace("_", "-")
        w, g = f"{d}|window", f"{d}|global"
        row: dict[str, Any] = {"published_mean_all": float(pub[col].mean())}
        for variant, key in (("window", w), ("global", g)):
            if key in det:
                s = det[key].dropna()
                ix = s.index.intersection(pub.index)
                row[f"{variant}_n"] = int(len(s))
                row[f"{variant}_mean"] = float(s.mean())
                row[f"{variant}_vs_published"] = {
                    "published_mean_same_series": float(pub.loc[ix, col].mean()),
                    "median_abs_diff": float((s[ix] - pub.loc[ix, col]).abs().median()),
                    "spearman": float(stats.spearmanr(s[ix], pub.loc[ix, col]).statistic)}
        t = pair(det, g, w)
        if t:
            row["global_minus_window"] = t
            if d in holm_set:
                family[d] = t
        out["detectors"][d] = row
    _holm(family)
    # Re-ranking: replace published values by the global variant (where it ran).
    new = pub.copy()
    for d in names:
        if subset == "TSB-AD-M" and d not in MULTI_REPRODUCED:
            continue
        if f"{d}|global" in det and d != "Sub_PCA" and d != "PCA":
            col = d.replace("_", "-")
            new[col] = det[f"{d}|global"].reindex(new.index).fillna(pub[col])
    m0, m1 = pub.mean(), new.mean()
    r0, r1 = m0.rank(ascending=False, method="min"), m1.rank(ascending=False, method="min")
    out["rank_change"] = {d.replace("_", "-"): [int(r0[d.replace("_", "-")]),
                                                int(r1[d.replace("_", "-")])]
                          for d in names if d.replace("_", "-") in r0}
    out["kendall_tau"] = float(stats.kendalltau(r0[m0.index], r1[m0.index]).statistic)
    out["n_published_detectors"] = int(pub.shape[1])
    out["first_to_twentieth"] = float(m0.sort_values(ascending=False).iloc[0]
                                      - m0.sort_values(ascending=False).iloc[19])
    return out


def pinned(subset: str = "TSB-AD-U", tag: str = "pinned") -> dict[str, Any]:
    """TSB-AD's detectors run unmodified from a later commit vs. the published values."""
    df = keyed(f"{subset}/{tag}_shard*.json")
    if df.empty:
        return {}
    pub = published(subset)
    out: dict[str, Any] = {}
    new = pub.copy()
    for d in df.columns:
        col = d.replace("_", "-")
        if col in new and df[d].notna().sum() >= 0.9 * len(df):
            new[col] = df[d].reindex(new.index).fillna(pub[col])
    m0, m1 = pub.mean(), new.mean()
    r0, r1 = m0.rank(ascending=False, method="min"), m1.rank(ascending=False, method="min")
    out["_ranking"] = {"kendall_tau": float(stats.kendalltau(r0[m0.index], r1[m0.index]).statistic),
                       "rank_change": {c.replace("_", "-"): [int(r0[c.replace("_", "-")]),
                                                             int(r1[c.replace("_", "-")])]
                                       for c in df.columns if c.replace("_", "-") in r0}}
    for d in df.columns:
        col = d.replace("_", "-")
        s = df[d].dropna()
        ix = s.index.intersection(pub.index)
        both = pd.concat([s[ix], pub.loc[ix, col]], axis=1, keys=["current", "published"])
        out[d] = {"n_ok": int(len(s)), "n_failed": int(df[d].isna().sum()),
                  "current_mean": float(s.mean()) if len(s) else None,
                  "published_mean_same_series": float(pub.loc[ix, col].mean()) if len(ix) else None,
                  "current_minus_published": (_paired(both["current"].to_numpy(),
                                                      both["published"].to_numpy())
                                              if len(both) else None)}
    return out


def baselines(subset: str) -> dict[str, Any]:
    df = (arms(subset, "knn", ("raw_knn", "raw_knn_wz"))
          .join(arms(subset, "knncf", ("raw_knn",), "cf:"), how="outer")
          .join(arms(subset, "raw", ("raw_flatten", "raw_meanpool")), how="outer")
          .join(arms(subset, "rawcf", ("raw_flatten",), "cf:"), how="outer"))
    pub, later = published(subset), _leaderboard(subset)
    df = df.join(pub, how="left").join(later, how="left")
    out: dict[str, Any] = {"means": {c: float(df[c].mean()) for c in
                                     ("raw_knn", "raw_knn_wz", "raw_flatten", "raw_meanpool",
                                      "cf:raw_knn", "cf:raw_flatten") if c in df}}
    best = pub.mean().idxmax()
    out["best_published"] = {"name": best, "mean": float(pub[best].mean())}
    top = list(pub.mean().sort_values(ascending=False).index[:6])
    fms = [c for c in pub.columns if c.split(" ")[0] in ("MOMENT", "TimesFM", "Chronos", "Lag-Llama")]
    for ref in ("raw_knn", "raw_flatten", "cf:raw_knn", "cf:raw_flatten"):
        out[f"{ref}_minus"] = {c: pair(df, ref, c) for c in
                               dict.fromkeys(top + fms + list(pub.columns) + list(later.columns)
                                             + ["raw_knn", "raw_knn_wz", "raw_flatten"])
                               if c != ref and c in df}
    out["later_means"] = {c: float(later[c].reindex(df.index).mean()) for c in later.columns}
    # Per-series rank of Raw-LW / Raw-1NN among the published detectors.
    for ref in ("raw_flatten", "raw_knn"):
        sub = pub.join(df[[ref]], how="inner")
        r = sub.rank(axis=1, ascending=False, method="min")[ref]
        out[f"{ref}_rank"] = {"n": int(len(sub)), "first": int((r == 1).sum()),
                              "median": float(r.median()), "pool": int(sub.shape[1])}
    return out


def factorial() -> dict[str, Any]:
    a = keyed("TSB-AD-U/ablation_shard*.json")
    if a.empty:
        return {}
    comp = a.dropna()
    pub = published("TSB-AD-U")["Sub-MCD"]
    anc = comp["window_z|mcd|period3"]
    ix = anc.index.intersection(pub.index)
    excluded = a.index.difference(comp.index)
    start = {"norm": "window_z", "est": "mcd", "win": "period3"}
    end = {"norm": "global", "est": "lw", "win": "w100"}

    def cell(s: dict[str, str]) -> pd.Series:
        return comp[f"{s['norm']}|{s['est']}|{s['win']}"]

    contrib: dict[str, list[pd.Series]] = {k: [] for k in start}
    for order in itertools.permutations(start):
        s = dict(start)
        for f in order:
            before = cell(s)
            s[f] = end[f]
            contrib[f].append(cell(s) - before)
    shapley = {}
    for f, parts in contrib.items():
        d = (sum(parts) / len(parts)).to_numpy()
        shapley[f] = {"mean": float(d.mean()), "ci95": _boot_ci(d)}
    return {"n_series": int(len(a)), "n_complete": int(len(comp)),
            "cell_means": {c: float(comp[c].mean()) for c in comp.columns},
            "anchor": {"n": int(len(ix)), "mean_ours": float(anc[ix].mean()),
                       "mean_published": float(pub[ix].mean()),
                       "median_abs_diff": float((anc[ix] - pub[ix]).abs().median()),
                       "spearman": float(stats.spearmanr(anc[ix], pub[ix]).statistic),
                       "n_diff_gt_0.05": int(((anc[ix] - pub[ix]).abs() > 0.05).sum())},
            "excluded": {"n": int(len(excluded)),
                         "published_sub_mcd_mean": float(pub.reindex(excluded).mean())},
            "total": float((cell(end) - cell(start)).mean()),
            "shapley": shapley,
            "window_effect_by_cell": {f"{n}|{e}": float((comp[f"{n}|{e}|w100"]
                                                          - comp[f"{n}|{e}|period3"]).mean())
                                      for n in ("global", "window_z", "window_minmax")
                                      for e in ("lw", "mcd")}}


def z_ratios() -> dict[str, dict[str, float]]:
    cache = ART / "z_ratio.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from scripts.analyze_mechanism import z_ratio

    out: dict[str, dict[str, float]] = {}
    for subset in TABLES:
        files = pd.read_csv(OFF / f"{subset}-Eva.csv")["file_name"]
        out[subset] = {f: z for f in files if (DATA / subset / f).exists()
                       and (z := z_ratio(DATA / subset / f)) is not None}
    cache.write_text(json.dumps(out, indent=1) + "\n")
    return out


def regime(subset: str, zr: dict[str, float]) -> dict[str, Any]:
    df = (arms(subset, "knn", ("raw_knn", "raw_knn_wz"))
          .join(arms(subset, "raw", ("raw_flatten",)), how="outer")
          .join(arms(subset, "vision", ("vision",)), how="outer")
          .join(arms(subset, "vfixed", ("vision_fixed",)), how="outer"))
    det = keyed(f"{subset}/detnorm_shard*.json")
    if subset == "TSB-AD-U":
        abl = keyed("TSB-AD-U/ablation_shard*.json")
        df = df.join(abl[["global|lw|w100", "window_z|lw|w100", "window_minmax|lw|w100"]],
                     how="left")
    if not det.empty:
        df = df.join(det, how="left")
    df["zr"] = pd.Series(zr)
    df = df.dropna(subset=["zr"])
    q = pd.qcut(df["zr"], 3, labels=["low", "mid", "high"])
    comps = {"raw_knn: per-window minus global": ("raw_knn_wz", "raw_knn"),
             "vision minus raw_flatten": ("vision", "raw_flatten"),
             "vision_fixed minus raw_flatten": ("vision_fixed", "raw_flatten")}
    if subset == "TSB-AD-U":
        comps["LW factorial: per-window z minus global"] = ("window_z|lw|w100", "global|lw|w100")
        comps["LW factorial: per-window min-max minus global"] = ("window_minmax|lw|w100",
                                                                 "global|lw|w100")
        comps["Sub-KNN: per-window minus global"] = ("Sub_KNN|window", "Sub_KNN|global")
    out: dict[str, Any] = {"cutoffs": [float(df["zr"].quantile(1 / 3)),
                                       float(df["zr"].quantile(2 / 3))],
                           "tercile_sizes": [int(x) for x in q.value_counts(sort=False)]}
    for name, (a, b) in comps.items():
        if a not in df or b not in df:
            continue
        sub = df[[a, b]].dropna()
        d = sub[a] - sub[b]
        out[name] = {k: {"n": int(len(g)), "mean": float(g.mean()), "ci95": _boot_ci(g.to_numpy()),
                         "wins": int((g > 0).sum())}
                     for k, g in d.groupby(q.reindex(d.index), observed=True)}
        out[name]["spearman_zr"] = float(stats.spearmanr(df.loc[d.index, "zr"], d).statistic)
    return out


def render() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for subset in TABLES:
        df = (arms(subset, "raw", ("raw_flatten",)).join(arms(subset, "knn", ("raw_knn",)))
              .join(arms(subset, "vision", ("vision",)))
              .join(arms(subset, "randvision", ("random_vision",)))
              .join(arms(subset, "vfixed", ("vision_fixed",))))
        out[subset] = {"means": {c: float(df[c].mean()) for c in df.columns},
                       **{f"{a} - {b}": pair(df, a, b) for a, b in
                          (("vision", "random_vision"), ("vision_fixed", "vision"),
                           ("vision_fixed", "raw_flatten"), ("vision_fixed", "raw_knn"),
                           ("vision", "raw_flatten"))}}
    tune: dict[str, Any] = {}
    for subset in TABLES:
        for m in ("0", "0.25", "0.5", "1.0"):
            rows = _records(f"tuning/{subset}/per_series_tune_m{m}_shard*.json")
            df = pd.DataFrame({a: [r[a]["VUS-PR"] if r.get(a) else np.nan for r in rows]
                               for a in ("vision_fixed", "vision", "raw_flatten")})
            if df.empty:
                continue
            tune[f"{subset}|{m}"] = {"n": int(len(df)), **{c: float(df[c].mean()) for c in df},
                                     "vision_fixed - raw_flatten": _paired(
                                         df["vision_fixed"].to_numpy(),
                                         df["raw_flatten"].to_numpy())}
    out["tuning_margins"] = tune
    return out


def drift() -> dict[str, Any]:
    df = keyed("TSB-AD-U/drift_shard*.json")
    if df.empty:
        return {}
    out: dict[str, Any] = {"n": int(len(df)),
                           "means": {c: float(df[c].mean()) for c in sorted(df.columns)}}
    for dr in ("level1", "level3", "scale2", "trend3"):
        for norm in ("global", "per_window", "rolling"):
            out[f"{dr}|{norm} - none|{norm}"] = pair(df, f"{dr}|{norm}", f"none|{norm}")
        out[f"{dr}: global - per_window"] = pair(df, f"{dr}|global", f"{dr}|per_window")
    return out


def held_out() -> dict[str, Any]:
    """Pre-registered H6: the normalization toggle on TSB-AD's tuning split (disjoint series)."""
    out: dict[str, Any] = {}
    for subset, names in (("TSB-AD-U", SUB_DETECTORS), ("TSB-AD-M", MULTI_DETECTORS)):
        det = keyed(f"tuning/{subset}/detnorm_tuning_shard*.json")
        if det.empty:
            continue
        fam: dict[str, dict[str, Any]] = {}
        for d in names:
            t = pair(det, f"{d}|global", f"{d}|window")
            if t:
                fam[d] = t
        _holm(fam)
        res: dict[str, Any] = {"n_series": int(len(det)), "tests": fam}
        if subset == "TSB-AD-U":
            pos = sum(t["delta"] > 0 for t in fam.values())
            sig = sum(t["delta"] > 0 and t["p_holm"] < 0.05 for t in fam.values())
            neg = sum(t["delta"] < 0 for t in fam.values())
            res["decision"] = ("supported" if pos >= 5 and sig >= 3 else
                               "refuted" if neg >= 2 else "inconclusive")
            res["counts"] = {"positive": pos, "holm_significant_positive": sig, "negative": neg}
        out[subset] = res
    return out


def far() -> dict[str, Any]:
    """Pre-registered H5: false-alarm rate at a calibrated threshold under injected drift."""
    df = keyed("TSB-AD-U/far_shard*.json")
    if df.empty:
        return {}
    out: dict[str, Any] = {"n": int(len(df)),
                           "means": {c: float(df[c].mean()) for c in sorted(df.columns)
                                     if "|" in c}}
    for dr in ("level1", "level3", "scale2", "trend3"):
        for norm in ("global", "per_window"):
            a, b = f"{dr}|{norm}", f"none|{norm}"
            if a in df and b in df:
                out[f"{a} - {b}"] = float((df[a] - df[b]).mean())
    return out


def tercile_all(subset: str = "TSB-AD-U") -> dict[str, Any]:
    """Per-window minus global by amplitude tercile for every per-window Sub- detector."""
    zr = pd.Series(z_ratios()[subset])
    det = keyed(f"{subset}/detnorm_shard*.json")
    q = pd.qcut(zr.reindex(det.index), 3, labels=["low", "mid", "high"])
    out: dict[str, Any] = {}
    for d in SUB_DETECTORS:
        x = (det[f"{d}|window"] - det[f"{d}|global"]).dropna()
        out[d] = {str(k): {"n": int(len(g)), "mean": float(g.mean()),
                           "ci95": _boot_ci(g.to_numpy()), "wins": int((g > 0).sum())}
                  for k, g in x.groupby(q.reindex(x.index), observed=True)}
        out[d]["spearman_zr"] = float(stats.spearmanr(zr.reindex(x.index), x).statistic)
    return out


def contamination() -> dict[str, Any]:
    """Series whose training part contains labelled anomalies, and results without them."""
    cache = ART / "train_contamination.json"
    if cache.exists():
        dirty = json.loads(cache.read_text())
    else:
        dirty = {}
        for subset in TABLES:
            files = pd.read_csv(OFF / f"{subset}-Eva.csv")["file_name"]
            dirty[subset] = sorted(
                f for f in files if (DATA / subset / f).exists()
                and pd.read_csv(DATA / subset / f, usecols=["Label"])["Label"].to_numpy()[
                    : int(Path(f).stem.split("_")[-3])].sum() > 0)
        cache.write_text(json.dumps(dirty, indent=1) + "\n")
    out: dict[str, Any] = {s: {"n_contaminated": len(v)} for s, v in dirty.items()}
    det = keyed("TSB-AD-U/detnorm_shard*.json")
    clean = det.index.difference(dirty["TSB-AD-U"])
    out["TSB-AD-U"]["toggle"] = {d: pair(det.loc[clean], f"{d}|global", f"{d}|window")
                                 for d in SUB_DETECTORS}
    for subset in TABLES:
        df = (arms(subset, "knn", ("raw_knn",)).join(arms(subset, "raw", ("raw_flatten",)))
              .join(published(subset)))
        c = df.loc[df.index.difference(dirty[subset])]
        pub = published(subset).loc[lambda x: x.index.isin(c.index)]
        best = pub.mean().idxmax()
        out[subset]["best_published_clean"] = {"name": best, "mean": float(pub[best].mean())}
        out[subset]["baselines_minus_best"] = {a: pair(c, a, best) for a in ("raw_knn", "raw_flatten")}
    return out


def other_measures() -> dict[str, Any]:
    """The univariate toggle under TSB-AD's other threshold-free measures."""
    out: dict[str, Any] = {}
    for m in ("AUC-PR", "AUC-ROC", "VUS-ROC"):
        det = keyed("TSB-AD-U/detnorm_shard*.json", value=m)
        out[m] = {d: pair(det, f"{d}|global", f"{d}|window") for d in SUB_DETECTORS}
    return out


def stride_both() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for subset in TABLES:
        a = arms(subset, "raw", ("raw_flatten",))["raw_flatten"]
        b = arms(subset, "stride1", ("raw_flatten",))
        if b.empty:
            continue
        both = pd.concat([a, b["raw_flatten"]], axis=1, keys=["default", "stride1"]).dropna()
        d = both["stride1"] - both["default"]
        out[subset] = {"n": int(len(both)), "mean_diff": float(d.mean()),
                       "max_abs_diff": float(d.abs().max()),
                       "spearman": float(stats.spearmanr(both["default"], both["stride1"]).statistic)}
    return out


def preregistered(zr: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Statistics behind H1, H2, H4 and H8 (decision rules in artifacts/preregistration)."""
    out: dict[str, Any] = {}
    det = keyed("TSB-AD-U/detnorm_shard*.json")
    abl = keyed("TSB-AD-U/ablation_shard*.json")
    u = (arms("TSB-AD-U", "knn", ("raw_knn", "raw_knn_wz")).join(arms("TSB-AD-U", "amp", ("raw_amp",)))
         .join(det[["Sub_KNN|global", "Sub_KNN|window"]]).join(abl[["global|lw|w100", "window_z|lw|w100"]]))
    u["P"] = u["raw_amp"] - u["raw_knn_wz"]
    u["zr"] = pd.Series(zr["TSB-AD-U"])

    def pcor(x: pd.Series, y: pd.Series, z: pd.Series) -> float:
        r = pd.concat([x, y, z], axis=1).dropna().rank()
        rxy, rxz, ryz = r.iloc[:, 0].corr(r.iloc[:, 1]), r.iloc[:, 0].corr(r.iloc[:, 2]), r.iloc[:, 1].corr(r.iloc[:, 2])
        return float((rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)))

    h1: dict[str, Any] = {"probe_means": {"level_scale": float(u["raw_amp"].mean()),
                                          "shape": float(u["raw_knn_wz"].mean())}}
    for name, a, b in (("Raw-1NN", "raw_knn", "raw_knn_wz"), ("Sub-KNN", "Sub_KNN|global", "Sub_KNN|window"),
                       ("LW", "global|lw|w100", "window_z|lw|w100")):
        dlt = u[a] - u[b]
        s = pd.concat([u["P"], u["zr"], dlt.rename("d")], axis=1).dropna()
        h1[name] = {"n": int(len(s)), "rho_P": float(stats.spearmanr(s["P"], s["d"]).statistic),
                    "rho_zr": float(stats.spearmanr(s["zr"], s["d"]).statistic),
                    "partial_rho_P_given_shape": pcor(u["P"], dlt, u["raw_knn_wz"])}
    out["H1"] = h1
    h2: dict[str, Any] = {}
    for subset in TABLES:
        df = (arms(subset, "knn", ("raw_knn", "raw_knn_wz")).join(arms(subset, "aug", ("raw_aug",)))).dropna()
        df["zr"] = pd.Series(zr[subset])
        df = df.dropna()
        q = pd.qcut(df["zr"], 3, labels=["low", "mid", "high"])
        terc = {}
        for k, g in df.groupby(q, observed=True):
            better = "raw_knn" if g["raw_knn"].mean() >= g["raw_knn_wz"].mean() else "raw_knn_wz"
            terc[str(k)] = {"better": better, **_paired(g["raw_aug"].to_numpy(), g[better].to_numpy())}
        h2[subset] = {"n": int(len(df)), "aug": float(df["raw_aug"].mean()),
                      "global": float(df["raw_knn"].mean()), "per_window": float(df["raw_knn_wz"].mean()),
                      "terciles": terc}
    out["H2"] = h2
    pub = published("TSB-AD-U")
    new = pub.copy()
    for d in SUB_DETECTORS:
        c = d.replace("_", "-")
        new[c] = det[f"{d}|global"].reindex(new.index).fillna(pub[c])  # failed runs keep published
    src = pd.Series([s.split("_")[1] for s in pub.index], index=pub.index)
    w = 1 / src.map(src.value_counts())
    m0, m1 = pub.mul(w, axis=0).sum() / w.sum(), new.mul(w, axis=0).sum() / w.sum()
    r0, r1 = m0.rank(ascending=False, method="min"), m1.rank(ascending=False, method="min")
    out["H4"] = {"n_sources": int(src.nunique()),
                 "kendall_tau": float(stats.kendalltau(r0[m0.index], r1[m0.index]).statistic),
                 "rank_sub_knn": int(r1["Sub-KNN"]), "convention": "failed runs keep published value"}
    later = _leaderboard("TSB-AD-U")
    allm = pub.join(later, how="left").join(arms("TSB-AD-U", "knn", ("raw_knn",)))
    zs = pd.Series(zr["TSB-AD-U"])
    out["H8"] = {c: float(stats.spearmanr(*pd.concat([zs, allm[c]], axis=1).dropna().T.to_numpy()).statistic)
                 for c in ("MOMENT (ZS)", "MOMENT (FT)", "TimesFM", "TSPulse (ZS)", "TSPulse (FT)",
                           "Chronos", "Lag-Llama", "raw_knn", "Sub-PCA", *[d.replace("_", "-") for d in SUB_DETECTORS])
                 if c in allm}
    return out


def _cluster_ci(d: pd.Series, clusters: pd.Series, seed: int = 0) -> list[float]:
    """95% interval of the mean of d when whole clusters, not series, are resampled."""
    rng = np.random.default_rng(seed)
    groups = [d[clusters == c].to_numpy() for c in clusters.unique()]
    sums, sizes = np.array([g.sum() for g in groups]), np.array([g.size for g in groups])
    boot = []
    for _ in range(10000):
        ix = rng.integers(0, len(groups), len(groups))
        boot.append(sums[ix].sum() / sizes[ix].sum())
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def robustness(zr: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Third review: reproduction agreement in detail, dependence between series, within-source H1."""
    out: dict[str, Any] = {"reproduction": {}}
    for subset, names in (("TSB-AD-U", SUB_DETECTORS), ("TSB-AD-M", MULTI_REPRODUCED)):
        det, pub = keyed(f"{subset}/detnorm_shard*.json"), published(subset)
        for d in names:
            s = det[f"{d}|window"].dropna()
            diff = (s - pub.loc[s.index, d.replace("_", "-")]).abs()
            out["reproduction"][f"{subset}|{d}"] = {
                "n": int(diff.size), "n_differ_1e-6": int((diff > 1e-6).sum()),
                "n_differ_0.01": int((diff > 0.01).sum()), "p95": float(diff.quantile(0.95)),
                "max": float(diff.max()), "argmax": str(diff.idxmax())}
    det = keyed("TSB-AD-U/detnorm_shard*.json")
    src = pd.Series([s.split("_")[1] for s in det.index], index=det.index)
    dup = json.loads((ART / "near_duplicates.json").read_text())["clusters_eva_099"]
    sig = pd.Series({f: i for i, g in enumerate(dup) for f in g}).reindex(det.index)
    clus: dict[str, Any] = {"n_sources": int(src.nunique()), "n_signal_clusters": int(sig.nunique())}
    for d in SUB_DETECTORS:
        dl = (det[f"{d}|global"] - det[f"{d}|window"]).dropna()
        clus[d] = {"n": int(dl.size), "delta": float(dl.mean()), "ci_series": _boot_ci(dl.to_numpy()),
                   "ci_source_cluster": _cluster_ci(dl, src[dl.index]),
                   "ci_signal_cluster": _cluster_ci(dl, sig[dl.index])}
    out["cluster_bootstrap"] = clus
    pub = published("TSB-AD-U")
    lodo: dict[str, Any] = {}
    for s in sorted(src.unique()):
        keep = pub.index[[f.split("_")[1] != s for f in pub.index]]
        new = pub.loc[keep].copy()
        for d in SUB_DETECTORS:
            c = d.replace("_", "-")
            new[c] = det[f"{d}|global"].reindex(keep).fillna(pub.loc[keep, c])
        m0, m1 = pub.loc[keep].mean(), new.mean()
        r0, r1 = m0.rank(ascending=False, method="min"), m1.rank(ascending=False, method="min")
        lodo[s] = {"n_dropped": int(len(pub) - len(keep)),
                   "kendall_tau": float(stats.kendalltau(r0, r1).statistic),
                   "rank_sub_knn": int(r1["Sub-KNN"]), "rank_sub_mcd": int(r1["Sub-MCD"]),
                   "min_delta": float(min((det.loc[det.index.intersection(keep), f"{d}|global"]
                                           - det.loc[det.index.intersection(keep), f"{d}|window"]).mean()
                                          for d in SUB_DETECTORS))}
    out["leave_one_source_out"] = {
        "per_source": lodo,
        "kendall_tau_range": [min(v["kendall_tau"] for v in lodo.values()),
                              max(v["kendall_tau"] for v in lodo.values())],
        "rank_sub_knn_range": [min(v["rank_sub_knn"] for v in lodo.values()),
                               max(v["rank_sub_knn"] for v in lodo.values())],
        "min_delta_over_sources": min(v["min_delta"] for v in lodo.values())}
    # H1 within sources: rank both variables, remove each source's mean rank, correlate residuals.
    u = (arms("TSB-AD-U", "knn", ("raw_knn", "raw_knn_wz")).join(arms("TSB-AD-U", "amp", ("raw_amp",)))
         .join(det[["Sub_KNN|global", "Sub_KNN|window"]]).join(keyed("TSB-AD-U/ablation_shard*.json")[
             ["global|lw|w100", "window_z|lw|w100"]]))
    u["P"] = u["raw_amp"] - u["raw_knn_wz"]
    within: dict[str, Any] = {}
    for name, a, b in (("Sub-KNN", "Sub_KNN|global", "Sub_KNN|window"), ("LW", "global|lw|w100", "window_z|lw|w100")):
        f = pd.concat([u["P"], (u[a] - u[b]).rename("d")], axis=1).dropna().rank()
        g = pd.Series([s.split("_")[1] for s in f.index], index=f.index)
        res = f - f.groupby(g).transform("mean")
        within[name] = {"n": int(len(f)), "rho_within_source": float(res["P"].corr(res["d"]))}
    out["H1_within_source"] = within
    return out


def render_lift() -> dict[str, Any]:
    abl = keyed("TSB-AD-U/ablation_shard*.json").dropna()
    v = arms("TSB-AD-U", "vision", ("vision",))
    df = abl[["window_minmax|lw|w100"]].join(v, how="inner")
    return {"n": int(len(df)), **pair(df, "vision", "window_minmax|lw|w100")}


def held_out_472() -> dict[str, Any]:
    """Pre-registered H9 (artifacts/preregistration/heldout472_hypothesis.md)."""
    det = keyed("TSB-AD-U/detnorm_rest_shard*.json")
    if det.empty:
        return {}
    # No statistic is computed before the run is complete or the registered cutoff has passed.
    cutoff = datetime(2026, 11, 5, tzinfo=timezone(timedelta(hours=9)))
    if len(det) < H9_SERIES and datetime.now(timezone.utc) < cutoff:
        return {"n_series": int(len(det)), "status": "incomplete; no statistic computed"}
    fam = {d: pair(det, f"{d}|global", f"{d}|window") for d in SUB_DETECTORS}
    fam = {k: v for k, v in fam.items() if v}
    _holm(fam)
    pos = sum(t["delta"] > 0 for t in fam.values())
    sig = sum(t["delta"] > 0 and t["p_holm"] < 0.05 for t in fam.values())
    neg = sum(t["delta"] < 0 for t in fam.values())
    src = pd.Series([s.split("_")[1] for s in det.index], index=det.index)
    by_source = {s: {d: float((det.loc[g.index, f"{d}|global"] - det.loc[g.index, f"{d}|window"]).mean())
                     for d in SUB_DETECTORS} for s, g in src.groupby(src)}
    clean = keyed("TSB-AD-U/detnorm_rest_ocsvmclean_shard*.json")
    return {"n_series": int(len(det)), "tests": fam,
            "counts": {"positive": pos, "holm_significant_positive": sig, "negative": neg},
            "decision": ("supported" if pos >= 5 and sig >= 4 else "refuted" if neg >= 2 else "inconclusive"),
            "by_source": by_source,
            "ocsvm_clean": pair(clean, "Sub_OCSVM|global", "Sub_OCSVM|window") if not clean.empty else None,
            "no_partner": _h9_no_partner(det)}


def held_out_472_regime(zr: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Exploratory, not registered: H9's Deltas by amplitude ratio, at the evaluation split's terciles."""
    det = keyed("TSB-AD-U/detnorm_rest_shard*.json")
    if len(det) < H9_SERIES:
        return {}
    cache = ART / "z_ratio_rest.json"
    if cache.exists():
        zrest = pd.Series(json.loads(cache.read_text()))
    else:
        from scripts.analyze_mechanism import z_ratio

        zrest = pd.Series({f: z_ratio(DATA / "TSB-AD-U" / f) for f in det.index}).dropna()
        cache.write_text(json.dumps(zrest.to_dict(), indent=1) + "\n")
    ev = pd.Series(zr["TSB-AD-U"])
    bounds = [-np.inf, float(ev.quantile(1 / 3)), float(ev.quantile(2 / 3)), np.inf]
    cat = pd.cut(zrest, bounds, labels=["low", "mid", "high"])
    src = pd.Series([s.split("_")[1] for s in zrest.index], index=zrest.index)
    out: dict[str, Any] = {"eva_bounds": bounds[1:3], "counts": cat.value_counts().to_dict(),
                           "median_zr_by_source": zrest.groupby(src).median().to_dict()}
    for d in SUB_DETECTORS:
        dl = (det[f"{d}|global"] - det[f"{d}|window"]).reindex(zrest.index)
        out[d] = {"spearman_zr": float(stats.spearmanr(zrest, dl, nan_policy="omit").statistic),
                  **{k: float(dl[cat == k].mean()) for k in ("low", "mid", "high")}}
    return out


def later_code_ranking() -> dict[str, Any]:
    """Published univariate ranking with all seven Sub- detectors replaced by TSB-AD's later code:
    6beac72e for six of them, 303dffbf for Sub-OCSVM (which crashes at 6beac72e)."""
    pin, head = keyed("TSB-AD-U/pinned_shard*.json"), keyed("TSB-AD-U/head_shard*.json")
    if pin.empty or "Sub_OCSVM" not in head:
        return {}
    pub = published("TSB-AD-U")
    new = pub.copy()
    for d in (*[c for c in pin.columns], "Sub_OCSVM"):
        src = head if d == "Sub_OCSVM" else pin
        new[d.replace("_", "-")] = src[d].reindex(new.index).fillna(pub[d.replace("_", "-")])
    m0, m1 = pub.mean(), new.mean()
    r0, r1 = m0.rank(ascending=False, method="min"), m1.rank(ascending=False, method="min")
    return {"kendall_tau": float(stats.kendalltau(r0, r1).statistic),
            "rank_change": {c: [int(r0[c]), int(r1[c])] for c in new.columns if c.startswith("Sub-")},
            "means": {c: float(m1[c]) for c in new.columns if c.startswith("Sub-")}}


def failure_policy() -> dict[str, Any]:
    """Ranking sensitivity to how failed global runs enter it: published value (default), the
    series where every replaced detector ran (common set), or a score of 0 (worst case)."""
    det, pub = keyed("TSB-AD-U/detnorm_shard*.json"), published("TSB-AD-U")
    cols = {d: f"{d}|global" for d in SUB_DETECTORS}
    failed = {d: int(det[c].reindex(pub.index).isna().sum()) for d, c in cols.items()}
    out: dict[str, Any] = {"failed_global_runs": failed}
    common = pub.index[det.reindex(pub.index)[list(cols.values())].notna().all(axis=1)]
    for policy in ("published", "common_set", "zero"):
        idx = common if policy == "common_set" else pub.index
        new, base = pub.loc[idx].copy(), pub.loc[idx]
        for d, c in cols.items():
            g = det[c].reindex(idx)
            new[d.replace("_", "-")] = g.fillna(base[d.replace("_", "-")] if policy == "published" else 0.0)
        m0, m1 = base.mean(), new.mean()
        r0, r1 = m0.rank(ascending=False, method="min"), m1.rank(ascending=False, method="min")
        out[policy] = {"n_series": int(len(idx)), "kendall_tau": float(stats.kendalltau(r0, r1).statistic),
                       "ranks": {d.replace("_", "-"): int(r1[d.replace("_", "-")]) for d in SUB_DETECTORS}}
    return out


def source_decomposition() -> dict[str, Any]:
    """Exploratory, designed after H9: split the difference between the 472 held-out series and
    the evaluation split into source composition and within-source change, on their common sources.

    Delta_H - Delta_E = sum_s (w_H - w_E) d_E  +  sum_s w_H (d_H - d_E), weights summing to 1 over
    the common sources."""
    eva, rest = keyed("TSB-AD-U/detnorm_shard*.json"), keyed("TSB-AD-U/detnorm_rest_shard*.json")
    if len(rest) < H9_SERIES:
        return {}
    dup = set(json.loads((ART / "near_duplicates.json").read_text())["rest_with_partner"]["0.99"])
    out: dict[str, Any] = {}
    for label, held in (("all_472", rest), ("no_partner_418", rest.drop(index=[s for s in dup if s in rest.index]))):
        res: dict[str, Any] = {}
        for d in SUB_DETECTORS:
            de = (eva[f"{d}|global"] - eva[f"{d}|window"]).dropna()
            dh = (held[f"{d}|global"] - held[f"{d}|window"]).dropna()
            se = pd.Series([s.split("_")[1] for s in de.index], index=de.index)
            sh = pd.Series([s.split("_")[1] for s in dh.index], index=dh.index)
            common = sorted(set(se) & set(sh))
            de_c, dh_c = de[se.isin(common)], dh[sh.isin(common)]
            ws_e = se[de_c.index].value_counts(normalize=True).reindex(common)
            ws_h = sh[dh_c.index].value_counts(normalize=True).reindex(common)
            d_e = de_c.groupby(se[de_c.index]).mean().reindex(common)
            d_h = dh_c.groupby(sh[dh_c.index]).mean().reindex(common)
            res[d] = {"common_sources": common,
                      "n_eva_common": int(len(de_c)), "n_held_common": int(len(dh_c)),
                      "delta_eva_all": float(de.mean()), "delta_eva_common": float((ws_e * d_e).sum()),
                      "delta_eva_other_sources": float(de[~se.isin(common)].mean()),
                      "n_eva_other": int((~se.isin(common)).sum()),
                      "ci_eva_common": _boot_ci(de_c.to_numpy()),
                      "ci_eva_other": _boot_ci(de[~se.isin(common)].to_numpy()),
                      "ci_held": _boot_ci(dh.to_numpy()),
                      "delta_held": float((ws_h * d_h).sum()),
                      "composition": float(((ws_h - ws_e) * d_e).sum()),
                      "within_source": float((ws_h * (d_h - d_e)).sum()),
                      "per_source": {s: {"w_eva": float(ws_e[s]), "w_held": float(ws_h[s]),
                                         "d_eva": float(d_e[s]), "d_held": float(d_h[s]),
                                         "n_eva": int((se[de_c.index] == s).sum()),
                                         "n_held": int((sh[dh_c.index] == s).sum())} for s in common}}
        out[label] = res
    return out


def rank_sensitivity() -> dict[str, Any]:
    """Ranks of the six per-window Sub- detectors over all 2^6 global/per-window settings, the
    other 26 detectors at their published values. A sensitivity to a configuration choice, not a
    sampling distribution."""
    det, pub = keyed("TSB-AD-U/detnorm_shard*.json"), published("TSB-AD-U")
    names = [d.replace("_", "-") for d in SUB_DETECTORS]
    variants = {d.replace("_", "-"): {v: det[f"{d}|{v}"].reindex(pub.index).fillna(pub[d.replace("_", "-")]).mean()
                                      for v in ("window", "global")} for d in SUB_DETECTORS}
    base = pub.mean()
    r0 = base.rank(ascending=False, method="min")
    ranks: dict[str, list[int]] = {n: [] for n in names}
    taus, firsts = [], set()
    pair_order: dict[str, set[bool]] = {}
    for combo in itertools.product(("window", "global"), repeat=len(names)):
        m = base.copy()
        for n, v in zip(names, combo):
            m[n] = variants[n][v]
        r = m.rank(ascending=False, method="min")
        for n in names:
            ranks[n].append(int(r[n]))
        taus.append(float(stats.kendalltau(r0, r).statistic))
        firsts.add(str(m.idxmax()))
        for a, b in itertools.combinations(m.index, 2):
            if a in names or b in names:
                pair_order.setdefault(f"{a}|{b}", set()).add(bool(m[a] > m[b]))
    flips = sorted(k for k, v in pair_order.items() if len(v) == 2)
    return {"n_settings": 2 ** len(names),
            "rank_range": {n: [min(v), max(v)] for n, v in ranks.items()},
            "kendall_tau_range": [min(taus), max(taus)], "top_detectors": sorted(firsts),
            "n_pairs_that_flip": len(flips), "n_pairs": len(pair_order), "flipping_pairs": flips}


def boundary_synthetic() -> dict[str, Any]:
    """Registered B1-B4 (artifacts/preregistration/boundary_hypothesis.md)."""
    path = ART / "synthetic" / "boundary.json"
    if not path.exists():
        return {}
    df = pd.DataFrame(json.loads(path.read_text()))
    cells = df.groupby(["kind", "mag", "ratio", "variant"]).mean(numeric_only=True)
    out: dict[str, Any] = {"cells": {"|".join(map(str, k)): {c: float(v) for c, v in row.items()
                                                             if c != "seed" and pd.notna(v)}
                                     for k, row in cells.iterrows()}}
    b12 = [(k, m, r) for k, m in (("level", 1.0), ("level", 3.0), ("scale", 2.0), ("scale", 3.0))
           for r in (4, 16)]
    pw = [cells.loc[(k, m, r, "per_window")] for k, m, r in b12]
    gl = [cells.loc[(k, m, r, "global")] for k, m, r in b12]
    b1_ok = sum(p["auc_interior"] <= 0.65 and g["auc_interior"] >= 0.90 for p, g in zip(pw, gl))
    b1_bad = sum(p["auc_interior"] >= 0.75 for p in pw)
    b2_ok = sum(p["auc_boundary"] >= 0.80 for p in pw)
    b2_bad = sum(p["auc_boundary"] < 0.65 for p in pw)
    out["B1"] = {"cells_meeting": int(b1_ok), "cells_refuting": int(b1_bad),
                 "decision": "supported" if b1_ok >= 7 else "refuted" if b1_bad >= 3 else "inconclusive"}
    out["B2"] = {"cells_meeting": int(b2_ok), "cells_refuting": int(b2_bad),
                 "decision": "supported" if b2_ok >= 7 else "refuted" if b2_bad >= 3 else "inconclusive"}
    rng = np.random.default_rng(0)
    b3: dict[str, Any] = {}
    for mag in (1.0, 3.0):
        def gaps(r: float) -> np.ndarray:
            sub = df[(df.kind == "level") & (df.mag == mag) & (df.ratio == r)].pivot(
                index="seed", columns="variant", values="VUS-PR")
            return (sub["global"] - sub["per_window"]).to_numpy()
        g16, g025 = gaps(16), gaps(0.25)
        boot = [rng.choice(g16, g16.size).mean() - rng.choice(g025, g025.size).mean() for _ in range(10000)]
        b3[str(mag)] = {"gap_r16": float(g16.mean()), "gap_r0.25": float(g025.mean()),
                        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
    b3_ok = all(v["ci95"][0] > 0 for v in b3.values())
    b3_bad = any(v["ci95"][1] < 0 for v in b3.values())
    out["B3"] = {**b3, "decision": "supported" if b3_ok else "refuted" if b3_bad else "inconclusive"}
    # F3 (boundary_followup.md): the same contrast paired by seed, which both cells share.
    rng3 = np.random.default_rng(0)
    f3: dict[str, Any] = {}
    for mag in (1.0, 3.0):
        piv = df[(df.kind == "level") & (df.mag == mag)].pivot_table(
            index="seed", columns=["ratio", "variant"], values="VUS-PR")
        dd = ((piv[(16.0, "global")] - piv[(16.0, "per_window")])
              - (piv[(0.25, "global")] - piv[(0.25, "per_window")])).to_numpy()
        boot = [rng3.choice(dd, dd.size).mean() for _ in range(10000)]
        f3[str(mag)] = {"mean": float(dd.mean()), "ci95": [float(np.percentile(boot, 2.5)),
                                                            float(np.percentile(boot, 97.5))]}
    out["F3_paired_B3"] = f3
    diffs = {str(r): float(cells.loc[("shape", 0.0, r, "per_window"), "VUS-PR"]
                           - cells.loc[("shape", 0.0, r, "global"), "VUS-PR"]) for r in (0.25, 1, 4, 16)}
    b4_ok = all(d >= -0.05 for d in diffs.values())
    b4_bad = sum(d < -0.10 for d in diffs.values()) >= 2
    out["B4"] = {"per_window_minus_global": diffs,
                 "decision": "supported" if b4_ok else "refuted" if b4_bad else "inconclusive"}
    return out


def seed_robustness() -> dict[str, Any]:
    """Sub-IForest and Sub-MCD with random_state 1 and 2 besides TSB-AD's fixed seeds: per-series
    Deltas averaged over the seeds first, then summarized over series."""
    out: dict[str, Any] = {}
    for label, base_pat, seed_pat, expected in (
            ("eva", "detnorm_shard*.json", "detnorm_seed{}_shard*.json", 350),
            ("held_out_472", "detnorm_rest_shard*.json", "detnorm_rest_seed{}_shard*.json", H9_SERIES)):
        base = keyed(f"TSB-AD-U/{base_pat}")
        seeds = {s: keyed(f"TSB-AD-U/{seed_pat.format(s)}") for s in (1, 2)}
        if any(v.empty for v in seeds.values()):
            continue
        ran = {s: int(len(v)) for s, v in seeds.items()}
        # A partial run would summarize whichever series finished first; report only its status.
        if min(ran.values()) < expected:
            out[label] = {"status": "incomplete", "series_run": ran, "expected": expected}
            continue
        res: dict[str, Any] = {"series_run": ran, "expected": expected}
        for d in ("Sub_IForest", "Sub_MCD"):
            deltas = [base[f"{d}|global"] - base[f"{d}|window"]]
            deltas += [v[f"{d}|global"] - v[f"{d}|window"] for v in seeds.values() if f"{d}|global" in v]
            frame = pd.concat(deltas, axis=1).dropna()
            if frame.empty:
                continue
            mean_d = frame.mean(axis=1)
            res[d] = {"n_series_all_seeds": int(len(frame)),
                      "n_failed_any_seed": int(expected - len(frame)),
                      "delta_per_seed": [float(frame[c].mean()) for c in frame.columns],
                      "delta_seed_averaged": float(mean_d.mean()), "ci95": _boot_ci(mean_d.to_numpy()),
                      "series_sign_agreement": float((np.sign(frame).nunique(axis=1) == 1).mean())}
        out[label] = res
    return out


def constant_score() -> dict[str, Any]:
    """Published univariate IForest and LOF against the VUS-PR of a constant score per series."""
    path = ART / "constant_score.json"
    if not path.exists():
        return {}
    c = json.loads(path.read_text())
    pub = published("TSB-AD-U")
    idx = [s for s in pub.index if s in c]
    v = pd.Series({s: c[s]["VUS-PR"] for s in idx})
    out: dict[str, Any] = {"n": len(idx), "constant_mean": float(v.mean()),
                           "anomaly_ratio_mean": float(np.mean([c[s]["anomaly_ratio"] for s in idx]))}
    for d in ("IForest", "LOF"):
        diff = pub.loc[idx, d] - v
        out[d] = {"published_mean": float(pub.loc[idx, d].mean()), "above_by_0.01": int((diff > 0.01).sum()),
                  "within_0.01": int((diff.abs() <= 0.01).sum()), "below_by_0.01": int((diff < -0.01).sum())}
    return out


def repeat_determinism() -> dict[str, Any]:
    """Series that two processes ran independently (a shard and its --part b continuation): the
    largest absolute VUS-PR difference per detector variant, a check that runs are deterministic."""
    by: dict[str, list[dict[str, Any]]] = {}
    for r in _records("TSB-AD-U/detnorm_rest_shard*.json"):
        by.setdefault(r["series"], []).append(r)
    dups = {k: v for k, v in by.items() if len(v) > 1}
    worst: dict[str, float] = {}
    for recs in dups.values():
        a, b = recs[0], recs[1]
        for k in a:
            if "|" in k and a[k] and b.get(k):
                worst[k] = max(worst.get(k, 0.0), abs(a[k]["VUS-PR"] - b[k]["VUS-PR"]))
    return {"n_series_run_twice": len(dups), "max_abs_diff": worst}


def boundary_followup() -> dict[str, Any]:
    """Registered F1-F2 (artifacts/preregistration/boundary_followup.md); F3 is in boundary_synthetic."""
    path = ART / "synthetic" / "boundary_followup.json"
    if not path.exists():
        return {}
    df = pd.DataFrame(json.loads(path.read_text()))
    out: dict[str, Any] = {}
    f1 = df[df.part == "F1"].groupby(["kind", "mag", "ratio", "variant"]).mean(numeric_only=True)
    cells = [(k, m, r) for k, m in (("level", 1.0), ("level", 3.0), ("scale", 2.0), ("scale", 3.0))
             for r in (4.0, 16.0)]
    clean = {f"{k}|{m}|{r}": {v: float(f1.loc[(k, m, r, v), "auc_interior_clean"]) for v in ("global", "per_window")}
             for k, m, r in cells}
    meets = sum(c["per_window"] <= 0.65 for c in clean.values())
    out["F1"] = {"clean_interior_auc": clean, "cells_meeting": int(meets),
                 "decision": "B1 reading stands" if meets >= 7 else "B1 depended on edge-contaminated negatives"}
    f2 = df[df.part == "F2"].groupby(["kind", "total", "d", "variant"]).mean(numeric_only=True)
    gaps: dict[str, Any] = {}
    lower, higher = 0, 0
    for kind in ("level", "scale"):
        for total in (400, 1600):
            ds = sorted(d for (k, t, d, v) in f2.index if k == kind and t == total and v == "global")
            g = {d: float(f2.loc[(kind, total, d, "per_window"), "test_auc_pr"]
                          - f2.loc[(kind, total, d, "global"), "test_auc_pr"]) for d in ds}
            gaps[f"{kind}|{total}"] = g
            lower += g[ds[-1]] <= g[ds[0]] - 0.05
            higher += g[ds[-1]] > g[ds[0]]
    out["F2"] = {"per_window_minus_global_test_auc_pr": gaps,
                 "pairs_lower_by_0.05": int(lower), "pairs_higher": int(higher),
                 "decision": "supported" if lower == 4 else "refuted" if higher >= 2 else "inconclusive",
                 "cells": {"|".join(map(str, k)): {c: float(v) for c, v in row.items() if c != "seed" and pd.notna(v)}
                           for k, row in f2.iterrows()}}
    return out


def validation_selection() -> dict[str, Any]:
    """Choose each detector's normalization on TSB-AD's tuning split (higher mean VUS-PR) and apply
    the choice to the evaluation split and to the 472 held-out series; per-series oracle (uses the
    test labels) as a diagnostic upper bound only."""
    tun = keyed("tuning/TSB-AD-U/detnorm_tuning_shard*.json")
    sets = {"eva": keyed("TSB-AD-U/detnorm_shard*.json"), "held_out_472": keyed("TSB-AD-U/detnorm_rest_shard*.json")}
    out: dict[str, Any] = {}
    for d in SUB_DETECTORS:
        pick = "global" if (tun[f"{d}|global"] - tun[f"{d}|window"]).mean() > 0 else "window"
        row: dict[str, Any] = {"chosen_on_tuning": pick}
        for name, df in sets.items():
            sub = df[[f"{d}|global", f"{d}|window"]].dropna()
            row[name] = {"window": float(sub[f"{d}|window"].mean()), "global": float(sub[f"{d}|global"].mean()),
                         "chosen": float(sub[f"{d}|{pick}"].mean()),
                         "per_series_oracle": float(sub.max(axis=1).mean())}
        out[d] = row
    return out


def _h9_no_partner(det: pd.DataFrame) -> dict[str, Any]:
    """Secondary H9 analysis: series without a |r| > 0.99 partner in eva or tuning (amendment)."""
    dup = json.loads((ART / "near_duplicates.json").read_text())["rest_with_partner"]["0.99"]
    sub = det.drop(index=[s for s in dup if s in det.index])
    fam = {d: pair(sub, f"{d}|global", f"{d}|window") for d in SUB_DETECTORS}
    fam = {k: v for k, v in fam.items() if v}
    _holm(fam)
    return {"n_series": int(len(sub)), "tests": fam}


def ocsvm_clean() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for subset, d in (("TSB-AD-U", "Sub_OCSVM"), ("TSB-AD-M", "OCSVM")):
        det = keyed(f"{subset}/detnorm_ocsvmclean_shard*.json")
        if not det.empty:
            out[subset] = {"means": {c: float(det[c].mean()) for c in det.columns},
                           "global_minus_window": pair(det, f"{d}|global", f"{d}|window")}
    return out


def threshold() -> dict[str, Any]:
    """Exchangeable-calibration threshold study (run_drift_threshold.py)."""
    recs = _records("TSB-AD-U/threshold_shard*.json")
    if not recs:
        return {}
    out: dict[str, Any] = {"n": len(recs)}
    keys = sorted({k for r in recs for k in r if "|" in k})
    for k in keys:
        vals = [r[k] for r in recs if r.get(k)]
        out[k] = {m: float(np.mean([v[m] for v in vals])) for m in ("far", "point_recall", "event_recall")}
    return out


def threshold2() -> dict[str, Any]:
    """Threshold calibrated on point scores with a reference/calibration gap, and equal-FAR
    operating points (run_drift_threshold2.py)."""
    recs = _records("TSB-AD-U/threshold2_shard*.json")
    if not recs:
        return {}
    out: dict[str, Any] = {"n": len(recs)}
    metrics = ("far", "point_recall", "event_recall", "false_episodes_per_1000")
    for k in sorted({k for r in recs for k in r if "|" in k}):
        vals = [r[k] for r in recs if r.get(k)]
        row: dict[str, Any] = {}
        for op in vals[0]:
            row[op] = {m: float(np.mean([v[op][m] for v in vals])) for m in metrics}
            delays = [v[op]["median_delay"] for v in vals if v[op]["median_delay"] is not None]
            row[op]["median_delay_median_over_series"] = float(np.median(delays))
        out[k] = row
    # Equal realized FAR: paired event recall, global minus per-window, per drift.
    paired: dict[str, Any] = {}
    for drift in ("none", "level1", "level3", "scale2", "trend3"):
        for op in ("equal_far_0.01", "equal_far_0.05"):
            g = [(r[f"{drift}|global"][op]["event_recall"], r[f"{drift}|per_window"][op]["event_recall"])
                 for r in recs if r.get(f"{drift}|global") and r.get(f"{drift}|per_window")]
            a, b = np.array(g).T
            paired[f"{drift}|{op}"] = _paired(a, b)
    out["event_recall_global_minus_per_window"] = paired
    # Ties and finite samples move the realized FAR below the target on some series; repeat the
    # comparison on series where both variants are within half the target of it.
    near: dict[str, Any] = {}
    real: dict[str, Any] = {}
    for drift in ("none", "level1", "level3", "scale2", "trend3"):
        for op, alpha in (("equal_far_0.01", 0.01), ("equal_far_0.05", 0.05)):
            rows = [(r[f"{drift}|global"][op], r[f"{drift}|per_window"][op]) for r in recs
                    if r.get(f"{drift}|global") and r.get(f"{drift}|per_window")]
            keep = [(g, w) for g, w in rows if abs(g["far"] - alpha) <= alpha / 2 and abs(w["far"] - alpha) <= alpha / 2]
            near[f"{drift}|{op}"] = _paired(np.array([g["event_recall"] for g, _ in keep]),
                                            np.array([w["event_recall"] for _, w in keep]))
            for m, i in (("global", 0), ("per_window", 1)):
                f = np.array([x[i]["far"] for x in rows])
                real[f"{drift}|{m}|{op}"] = {"mean": float(f.mean()), "max": float(f.max()),
                                            "n_below_half_target": int((f < alpha / 2).sum())}
    out["event_recall_near_target_far"] = near
    out["realized_far"] = real
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    zr = z_ratios()
    res: dict[str, Any] = {
        "toggle": {s: toggle(s) for s in TABLES},
        "pinned": pinned(),
        "baselines": {s: baselines(s) for s in TABLES},
        "factorial": factorial(),
        "regime": {s: regime(s, zr.get(s, {})) for s in TABLES},
        "render": render(),
        "drift": drift(),
        "held_out": held_out(),
        "held_out_472": held_out_472(),
        "pinned_head": {s: pinned(s, "head") for s in TABLES},
        "later_code_ranking": later_code_ranking(),
        "failure_policy": failure_policy(),
        "validation_selection": validation_selection(),
        "source_decomposition": source_decomposition(),
        "rank_sensitivity": rank_sensitivity(),
        "boundary_synthetic": boundary_synthetic(),
        "boundary_followup": boundary_followup(),
        "constant_score": constant_score(),
        "repeat_determinism": repeat_determinism(),
        "seed_robustness": seed_robustness(),
        "contamination": contamination(),
        "other_measures": other_measures(),
        "stride_both": stride_both(),
        "preregistered": preregistered(zr),
        "render_lift": render_lift(),
        "ocsvm_clean": ocsvm_clean(),
        "threshold": threshold(),
        "threshold2": threshold2(),
        "far": far(),
        "tercile_sub_detectors": tercile_all(),
        "robustness": robustness(zr),
        "held_out_472_regime": held_out_472_regime(zr),
    }
    res["key_conflicts"] = {"n": len(KEY_CONFLICTS), "first": KEY_CONFLICTS[:20]}
    for name in ("stride_check.json", "cost.json", "scaling.json"):
        if (ART / name).exists():
            res[name.split(".")[0]] = json.loads((ART / name).read_text())
    (ART / "paper_numbers.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    LOGGER.info("wrote %s", ART / "paper_numbers.json")


if __name__ == "__main__":
    main()
