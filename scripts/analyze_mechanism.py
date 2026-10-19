"""Mechanism analyses on TSB-AD's evaluation split (official protocol).

Three questions, each answered from per-series VUS-PR already computed by
run_official_protocol.py and run_normalization_ablation.py:

1. ablation  Which ingredient separates TSB-AD's Sub-MCD (0.237) from raw_flatten (0.505)?
             Full 3x2x2 factorial (normalisation x estimator x window) on TSB-AD-U-Eva; the
             anchor cell (window_z, mcd, period3) is checked against TSB-AD's published Sub-MCD.
2. encoder   What does the rendered pipeline's deficit come from? Tests, per subset, in one
             pre-specified Holm family:
               vision vs random_vision       (does pretraining matter?)
               vision_fixed vs vision        (does restoring level/amplitude in the image help?)
               raw_flatten vs vision_fixed   (what remains once it is restored?)
             and, on TSB-AD-U, the raw analogue of the renderer's normalisation
             (window_minmax|lw|w100) against vision.
3. regime    Where does rendering still win? Spearman correlation of the label-derived
             z_ratio = median|z| on anomalous points / median|z| on normal points (z from
             train-part statistics, averaged over channels) with the per-series difference
             vision - raw_flatten, plus per-source-dataset means.

Output: artifacts/official_protocol/mechanism.json. Inputs are read from artifacts/ only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from scripts.summarize_official import _holm, _paired

LOGGER = logging.getLogger(__name__)
RES = Path("artifacts/official_protocol")  # per-series inputs (see collect_official_artifacts)
DATA = Path("data/tsb_ad")
OFF = DATA / "official"
ART = Path("artifacts/official_protocol")
SUBSETS = ("TSB-AD-U", "TSB-AD-M")
TABLES = {"TSB-AD-U": "uni_mergedTable_VUS-PR.csv", "TSB-AD-M": "multi_mergedTable_VUS-PR.csv"}
ARMS = {"raw": ("raw_meanpool", "raw_flatten"), "vision": ("vision",),
        "randvision": ("random_vision",), "vfixed": ("vision_fixed",)}


def arms_table(subset: str) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    for tag, arms in ARMS.items():
        for f in sorted((RES / subset).glob(f"per_series_{tag}_shard*.json")):
            for r in json.loads(f.read_text()):
                for arm in arms:
                    if r.get(arm):
                        rows.setdefault(r["series"], {})[arm] = r[arm]["VUS-PR"]
    return pd.DataFrame.from_dict(rows, orient="index")


def ablation_table() -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    for f in sorted((RES / "TSB-AD-U").glob("ablation_shard*.json")):
        for r in json.loads(f.read_text()):
            for k, v in r.items():
                if "|" in k and v:
                    rows.setdefault(r["series"], {})[k] = v["VUS-PR"]
    return pd.DataFrame.from_dict(rows, orient="index")


def _pair(df: pd.DataFrame, a: str, b: str) -> dict[str, Any] | None:
    if a not in df or b not in df:
        return None
    sub = df[[a, b]].dropna()
    return _paired(sub[a].to_numpy(), sub[b].to_numpy()) if len(sub) else None


def analyse_ablation(arms_u: pd.DataFrame) -> dict[str, Any]:
    abl = ablation_table()
    if abl.empty:
        return {}
    complete = abl.dropna()
    out: dict[str, Any] = {"n_series": int(len(abl)), "n_complete": int(len(complete)),
                           "cell_means": complete.mean().round(4).to_dict()}
    pub = pd.read_csv(OFF / TABLES["TSB-AD-U"]).set_index("file")["Sub-MCD"]
    anchor = complete["window_z|mcd|period3"]
    common = anchor.index.intersection(pub.index)
    diff = anchor[common] - pub[common]
    out["anchor_vs_published_sub_mcd"] = {
        "n": int(len(common)), "mean_ours": float(anchor[common].mean()),
        "mean_published": float(pub[common].mean()),
        "max_abs_diff": float(diff.abs().max()), "median_abs_diff": float(diff.abs().median()),
        "spearman": float(stats.spearmanr(anchor[common], pub[common]).statistic)}
    # Main effects: average paired difference across the other factors' levels.
    effects: dict[str, Any] = {}
    for name, lo, hi, pos in (("norm: global vs window_z", "window_z", "global", 0),
                              ("norm: global vs window_minmax", "window_minmax", "global", 0),
                              ("estimator: lw vs mcd", "mcd", "lw", 1),
                              ("window: w100 vs period3", "period3", "w100", 2)):
        pairs = []
        for col in complete.columns:
            parts = col.split("|")
            if parts[pos] != hi:
                continue
            parts[pos] = lo
            other = "|".join(parts)
            pairs.append(complete[col] - complete[other])
        d = pd.concat(pairs, axis=1).mean(axis=1)
        effects[name] = {"mean_paired_diff": float(d.mean()), "cells_averaged": len(pairs)}
    out["main_effects"] = effects
    # One-factor-at-a-time path from Sub-MCD to raw_flatten, each step a paired test.
    path = ["window_z|mcd|period3", "global|mcd|period3", "global|lw|period3", "global|lw|w100"]
    out["path_from_sub_mcd"] = {f"{a} -> {b}": _pair(complete, b, a) for a, b in zip(path, path[1:])}
    both = complete.join(arms_u, how="inner")
    out["consistency_global_lw_w100_vs_raw_flatten"] = _pair(both, "global|lw|w100", "raw_flatten")
    out["renderer_normalisation_raw_analogue_vs_vision"] = _pair(both, "vision",
                                                                 "window_minmax|lw|w100")
    return out


def z_ratio(path: Path) -> float | None:
    df = pd.read_csv(path).dropna()
    x = df.iloc[:, 0:-1].to_numpy(dtype=float)
    lab = df["Label"].astype(int).to_numpy().astype(bool)
    n_train = int(path.stem.split("_")[-3])
    mu, sd = x[:n_train].mean(axis=0), x[:n_train].std(axis=0) + 1e-8
    z = np.abs((x - mu) / sd).mean(axis=1)
    if lab.all() or not lab.any():
        return None
    return float(np.median(z[lab]) / (np.median(z[~lab]) + 1e-9))


def analyse_regime(subset: str, df: pd.DataFrame) -> dict[str, Any]:
    if "vision" not in df or "raw_flatten" not in df:
        return {}
    df = df.dropna(subset=["vision", "raw_flatten"]).copy()
    df["zr"] = [z_ratio(DATA / subset / s) for s in df.index]
    df["delta"] = df["vision"] - df["raw_flatten"]
    df["source"] = [s.split("_")[1] for s in df.index]
    ok = df.dropna(subset=["zr"])
    rho = stats.spearmanr(ok["zr"], ok["delta"])
    out: dict[str, Any] = {"n": int(len(ok)), "spearman_zr_delta": float(rho.statistic),
                           "p": float(rho.pvalue)}
    for arm in ("raw_flatten", "vision", "vision_fixed"):
        if arm in ok:
            r = stats.spearmanr(ok["zr"], ok[arm], nan_policy="omit")
            out[f"spearman_zr_{arm}"] = float(r.statistic)
    q = pd.qcut(ok["zr"], 3, labels=["low", "mid", "high"])
    out["delta_by_zr_tercile"] = {str(k): {"n": int(len(g)), "mean_delta": float(g["delta"].mean()),
                                           "vision_wins": int((g["delta"] > 0).sum()),
                                           "zr_max": float(g["zr"].max())}
                                  for k, g in ok.groupby(q, observed=True)}
    cols = [c for c in ("raw_flatten", "vision", "vision_fixed", "random_vision") if c in ok]
    out["by_source"] = {src: {"n": int(len(g)), **{c: round(float(g[c].mean()), 4) for c in cols},
                              "vision_wins": int((g["delta"] > 0).sum())}
                        for src, g in ok.groupby("source")}
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    result: dict[str, Any] = {}
    tables = {s: arms_table(s) for s in SUBSETS}
    for subset, df in tables.items():
        fam = {name: t for name, t in (
            ("vision vs random_vision", _pair(df, "vision", "random_vision")),
            ("vision_fixed vs vision", _pair(df, "vision_fixed", "vision")),
            ("raw_flatten vs vision_fixed", _pair(df, "raw_flatten", "vision_fixed")),
        ) if t is not None}
        _holm(fam)
        means = {c: float(df[c].mean()) for c in df.columns}
        result[subset] = {"n_by_arm": df.notna().sum().to_dict(), "means": means,
                          "encoder_family": fam, "regime": analyse_regime(subset, df)}
    result["ablation_TSB-AD-U"] = analyse_ablation(tables["TSB-AD-U"])
    (ART / "mechanism.json").write_text(json.dumps(result, indent=2) + "\n")
    LOGGER.info("wrote %s", ART / "mechanism.json")


if __name__ == "__main__":
    main()
