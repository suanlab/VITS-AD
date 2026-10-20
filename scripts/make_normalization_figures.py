"""Figures for the normalization study (factorial path and amplitude regime).

Reads only released artifacts (artifacts/official_protocol/*, including the cached amplitude
ratios in z_ratio.json), so it runs without a GPU or the raw series.

  fig_toggle.pdf     per-window vs global normalization of TSB-AD's Sub- detectors (ranks)
  fig_factorial.pdf  mean VUS-PR along the one-factor-at-a-time path from Sub-MCD to the
                     baseline, with the other normalization levels for reference
  fig_regime.pdf     paired difference to raw-flatten by amplitude-ratio tercile

Usage: python -m scripts.make_normalization_figures --out paper/icde2027/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.analyze_mechanism import DATA, z_ratio

ART = Path("artifacts/official_protocol")
ARMS = {"raw": ("raw_flatten",), "vision": ("vision",), "vfixed": ("vision_fixed",)}


def _per_series(subset: str) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    for tag, arms in ARMS.items():
        for f in sorted((ART / subset).glob(f"per_series_{tag}_shard*.json")):
            for r in json.loads(f.read_text()):
                for a in arms:
                    if r.get(a):
                        rows.setdefault(r["series"], {})[a] = r[a]["VUS-PR"]
    return pd.DataFrame.from_dict(rows, orient="index")


def _ablation() -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    for f in sorted((ART / "TSB-AD-U").glob("ablation_shard*.json")):
        for r in json.loads(f.read_text()):
            for k, v in r.items():
                if "|" in k and v:
                    rows.setdefault(r["series"], {})[k] = v["VUS-PR"]
    return pd.DataFrame.from_dict(rows, orient="index").dropna()


def _z_ratios(subset: str) -> pd.Series:
    """Cached amplitude ratios (written by paper_numbers.py); computed from raw data if absent."""
    cache = ART / "z_ratio.json"
    if cache.exists():
        return pd.Series(json.loads(cache.read_text())[subset])
    files = sorted({r["series"] for f in (ART / subset).glob("per_series_raw_shard*.json")
                    for r in json.loads(f.read_text())})
    return pd.Series({s: z_ratio(DATA / subset / s) for s in files})


def fig_factorial(out: Path) -> None:
    abl = _ablation()
    path = [("window_z|mcd|period3", "Sub-MCD\n(re-impl.)"),
            ("global|mcd|period3", "+ global\nnorm."),
            ("global|lw|period3", "+ Ledoit-\nWolf"),
            ("global|lw|w100", "+ $W$=100\n(baseline, stride 1)")]
    means = [abl[c].mean() for c, _ in path]
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    ax.bar(range(4), means, color=["#999999", "#1f77b4", "#1f77b4", "#1f77b4"], width=0.6)
    for i, m in enumerate(means):
        ax.text(i, m + 0.01, f"{m:.3f}", ha="center", fontsize=7)
    for c, ls, lab in (("window_minmax|lw|w100", ":", "per-window min-max, LW"),
                       ("window_z|lw|w100", "--", "per-window z, LW")):
        ax.axhline(abl[c].mean(), color="#d62728", ls=ls, lw=0.8, label=lab)
    ax.set_xticks(range(4), [lab for _, lab in path], fontsize=7)
    ax.set_ylabel("mean VUS-PR", fontsize=8)
    ax.set_ylim(0, 0.6)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=6, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(out / "fig_factorial.pdf")
    plt.close(fig)


def fig_regime(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9), sharey=True)
    for ax, subset, title in zip(axes, ("TSB-AD-U", "TSB-AD-M"), ("univariate", "multivariate")):
        df = _per_series(subset).dropna()
        df["zr"] = _z_ratios(subset).reindex(df.index)
        df = df.dropna(subset=["zr"])
        q = pd.qcut(df["zr"], 3, labels=["low", "mid", "high"])
        x = np.arange(3)
        for k, (arm, col, lab) in enumerate((("vision", "#ff7f0e", "vision"),
                                             ("vision_fixed", "#2ca02c", "vision-fixed"))):
            d = (df[arm] - df["raw_flatten"]).groupby(q, observed=True).mean()
            ax.bar(x + (k - 0.5) * 0.36, d.values, width=0.36, color=col, label=lab)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(x, ["low", "mid", "high"], fontsize=7)
        ax.set_xlabel("amplitude ratio $z_r$ tercile", fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("$\\Delta$ VUS-PR vs. raw-flatten", fontsize=7)
    axes[0].legend(fontsize=6, frameon=False, loc="lower left")
    fig.tight_layout(pad=0.2)
    fig.savefig(out / "fig_regime.pdf")
    plt.close(fig)


def fig_toggle(out: Path) -> None:
    """Published (per-window) vs. global normalization for TSB-AD's Sub- detectors, with ranks."""
    nums = json.loads((ART / "paper_numbers.json").read_text())["toggle"]["TSB-AD-U"]
    dets = ["Sub_KNN", "Sub_MCD", "Sub_OCSVM", "Sub_HBOS", "Sub_IForest", "Sub_LOF"]
    pca = nums["detectors"]["Sub_PCA"]["published_mean_all"]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for i, d in enumerate(dets):
        row = nums["detectors"][d]
        w, g = row["window_mean"], row["global_mean"]
        r0, r1 = nums["rank_change"][d.replace("_", "-")]
        ax.plot([w, g], [i, i], color="#999999", lw=1, zorder=1)
        ax.scatter([w], [i], color="#d62728", s=18, zorder=2)
        ax.scatter([g], [i], color="#1f77b4", s=18, zorder=2)
        ax.text(g + 0.008, i, f"#{r0}$\\to$#{r1}", va="center", fontsize=6)
    ax.axvline(pca, color="black", ls=":", lw=0.8)
    ax.text(pca + 0.005, -0.75, "Sub-PCA, best published", fontsize=6, ha="left", va="center")
    ax.set_yticks(range(len(dets)), [d.replace("_", "-") for d in dets], fontsize=7)
    ax.set_xlabel("mean VUS-PR (TSB-AD-U-Eva, 350 series)", fontsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.scatter([], [], color="#d62728", s=18, label="per-window z (as published)")
    ax.scatter([], [], color="#1f77b4", s=18, label="global (train statistics)")
    ax.legend(fontsize=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.set_xlim(0.18, 0.62)
    ax.set_ylim(len(dets) - 0.5, -1.1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(out / "fig_toggle.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("paper/icde2027/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_toggle(args.out)
    fig_factorial(args.out)
    fig_regime(args.out)


if __name__ == "__main__":
    main()
