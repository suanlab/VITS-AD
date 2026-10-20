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
MULTI_DETECTORS = ("KNN", "MCD", "OCSVM", "HBOS", "IForest", "LOF", "PCA")


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


def keyed(pattern: str, value: str = "VUS-PR") -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    for r in _records(pattern):
        for k, v in r.items():
            if k in ("series", "n", "n_train", "sliding_window", "period3", "n_channels"):
                continue
            if isinstance(v, dict):
                rows.setdefault(r["series"], {})[k] = v[value]
            elif isinstance(v, (int, float)) and v is not None:
                rows.setdefault(r["series"], {})[k] = float(v)
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
            family[d] = t
        out["detectors"][d] = row
    _holm(family)
    # Re-ranking: replace published values by the global variant (where it ran).
    new = pub.copy()
    for d in names:
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


def pinned() -> dict[str, Any]:
    df = keyed("TSB-AD-U/pinned_shard*.json")
    if df.empty:
        return {}
    pub = published("TSB-AD-U")
    out: dict[str, Any] = {}
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
                               dict.fromkeys(top + fms + list(later.columns)
                                             + ["raw_knn", "raw_knn_wz", "raw_flatten"])
                               if c != ref and c in df}
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
    }
    for name in ("stride_check.json", "cost.json"):
        if (ART / name).exists():
            res[name.split(".")[0]] = json.loads((ART / name).read_text())
    (ART / "paper_numbers.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    LOGGER.info("wrote %s", ART / "paper_numbers.json")


if __name__ == "__main__":
    main()
