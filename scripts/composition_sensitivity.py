"""How far a benchmark conclusion depends on which sources its split contains.

A benchmark reports a difference between two systems as a mean over items (series), and the items
come from sources (datasets). With per-source mean differences d_s and the split's source weights
w_s, the reported value is sum_s w_s d_s: a linear function of the weights. This module computes,
from per-item results and each item's source alone, so for any harness:

* the composition range [min_s d_s, max_s d_s]: every reweighting of the sources lies in it, so a
  sign holds under every split composition exactly when no source reverses it;
* the flip radius: the smallest total-variation distance from the split's weights to weights under
  which the mean difference reaches zero (a linear program); optionally with a cap per source, e.g.
  the share of a split of n items that the benchmark's pool of that source can fill, which makes
  the reweighting realizable as a re-drawn split of the same benchmark;
* an error budget for one entry: the interval from resampling items, the interval from resampling
  whole sources, and systematic terms given by the caller (another preprocessing setting, another
  split, another code version), side by side.

    python -m scripts.composition_sensitivity results.csv --a global --b window [--n-split 350]

``results.csv`` is long format with columns item, source, system, value. An optional column
in_split (1/0) marks the split's items among a larger pool; the realizable radius then uses the
pool's per-source means and caps. Output is JSON on stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog

LOGGER = logging.getLogger(__name__)


def source_means(diff: pd.Series, source: pd.Series) -> pd.DataFrame:
    """Per-source item count and mean difference, sorted by mean.

    Args:
        diff: per-item difference (system a minus system b), indexed by item.
        source: per-item source label, indexed by item (a superset of diff's index is fine).

    Returns:
        DataFrame indexed by source with columns ``n`` and ``mean``.
    """
    d = diff.dropna()
    g = d.groupby(source.reindex(d.index))
    return pd.DataFrame({"n": g.size(), "mean": g.mean()}).sort_values("mean")


def flip_radius(means: pd.Series, weights: pd.Series, cap: pd.Series | None = None) -> float | None:
    """Smallest total-variation distance from ``weights`` to source weights whose weighted mean
    of ``means`` is zero or has the opposite sign.

    Solved as a linear program: minimise sum_s u_s / 2 subject to u_s >= |w_s - w0_s|,
    sum_s w_s = 1, 0 <= w_s <= cap_s and sign * sum_s w_s m_s <= 0.

    Args:
        means: per-source mean difference.
        weights: the split's source weights (shares of its items), same index; normalized here.
        cap: optional per-source upper bound on a weight (e.g. pool size / split size).

    Returns:
        The radius in [0, 1], 0.0 when the weighted mean is already zero, or None when no
        admissible weights reverse the sign.
    """
    m = means.astype(float)
    w0 = weights.reindex(m.index).fillna(0.0).astype(float)
    w0 = w0 / w0.sum()
    current = float((w0 * m).sum())
    if current == 0.0:
        return 0.0
    sign = 1.0 if current > 0 else -1.0
    k = len(m)
    ub = np.ones(k) if cap is None else np.minimum(1.0, cap.reindex(m.index).fillna(0.0).to_numpy(float))
    # variables: w (k), u (k)
    c = np.concatenate([np.zeros(k), 0.5 * np.ones(k)])
    eye = np.eye(k)
    a_ub = np.vstack([np.hstack([eye, -eye]),                  # w - u <= w0
                      np.hstack([-eye, -eye]),                 # -w - u <= -w0
                      np.concatenate([sign * m.to_numpy(), np.zeros(k)])[None, :]])
    b_ub = np.concatenate([w0.to_numpy(), -w0.to_numpy(), [0.0]])
    a_eq = np.concatenate([np.ones(k), np.zeros(k)])[None, :]
    bounds = [(0.0, float(b)) for b in ub] + [(0.0, None)] * k
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=[1.0], bounds=bounds, method="highs")
    if res.status == 2:
        return None
    if res.status != 0:
        raise RuntimeError(f"linear program failed: {res.message}")
    return float(res.fun)


def _boot_half(values: np.ndarray, groups: np.ndarray | None, seed: int = 0, n_boot: int = 10000) -> float:
    """Half-width of the 95% percentile interval of the mean, resampling items or whole groups."""
    rng = np.random.default_rng(seed)
    if groups is None:
        boot = [values[rng.integers(0, values.size, values.size)].mean() for _ in range(n_boot)]
    else:
        labels = np.unique(groups)
        sums = np.array([values[groups == g].sum() for g in labels])
        sizes = np.array([(groups == g).sum() for g in labels])
        boot = []
        for _ in range(n_boot):
            ix = rng.integers(0, labels.size, labels.size)
            boot.append(sums[ix].sum() / sizes[ix].sum())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float((hi - lo) / 2)


def analyse(diff: pd.Series, source: pd.Series) -> dict[str, Any]:
    """Split mean, per-source means, composition range, reversing sources and flip radius.

    Args:
        diff: per-item difference (a minus b) on the split, indexed by item.
        source: per-item source label.

    Returns:
        Dict with ``mean``, ``range``, ``reversing_sources``, ``flip_radius`` (total variation,
        None if no reweighting reverses the sign), ``flip_radius_items`` (the radius times the
        number of items) and ``by_source``.
    """
    sm = source_means(diff, source)
    w = sm["n"] / sm["n"].sum()
    mean = float((w * sm["mean"]).sum())
    sign = np.sign(mean)
    r = flip_radius(sm["mean"], w)
    return {
        "n_items": int(sm["n"].sum()), "n_sources": int(len(sm)), "mean": mean,
        "range": [float(sm["mean"].min()), float(sm["mean"].max())],
        "reversing_sources": [str(x) for x, v in sm["mean"].items() if np.sign(v) == -sign and v != 0],
        "flip_radius": r, "flip_radius_items": None if r is None else float(r * sm["n"].sum()),
        "by_source": {str(x): {"n": int(row["n"]), "mean": float(row["mean"])} for x, row in sm.iterrows()},
    }


def analyse_with_pool(pool_diff: pd.Series, pool_source: pd.Series, split_items: pd.Index,
                      n_split: int) -> dict[str, Any]:
    """Realizable flip radius: per-source means over the whole pool, starting weights those of the
    split, each source capped at the share of an ``n_split``-item split its pool items can fill.
    The radius is then the fewest items of the split (as a share) that must be re-drawn from the
    same pool to bring the mean difference to zero, assuming each source's items are exchangeable.
    """
    pm = source_means(pool_diff, pool_source)
    split_src = pool_source.reindex(split_items).dropna()
    w0 = split_src.value_counts().reindex(pm.index).fillna(0.0)
    w0 = w0 / w0.sum()
    cap = (pm["n"] / n_split).clip(upper=1.0)
    mean = float((w0 * pm["mean"]).sum())
    r = flip_radius(pm["mean"], w0, cap)
    return {"n_pool": int(pm["n"].sum()), "n_split": int(n_split), "mean_at_split_weights": mean,
            "flip_radius": r, "flip_radius_items": None if r is None else float(r * n_split),
            "range_pool": [float(pm["mean"].min()), float(pm["mean"].max())]}


def error_budget(values: pd.Series, source: pd.Series, systematic: dict[str, float]) -> dict[str, Any]:
    """Statistical and systematic terms for one entry's mean, in the entry's units.

    Args:
        values: per-item values of the entry (e.g. one detector's VUS-PR) on the split.
        source: per-item source.
        systematic: named systematic shifts of the mean (absolute differences supplied by the
            caller, e.g. {"normalization": |global - window|, "split": |eva - rest|}).

    Returns:
        Dict with the mean, ``items_half`` and ``sources_half`` (95% interval half-widths), the
        systematic terms, and the largest systematic term relative to ``items_half``.
    """
    v = values.dropna()
    s = source.reindex(v.index).to_numpy()
    items_half = _boot_half(v.to_numpy(), None)
    sources_half = _boot_half(v.to_numpy(), s)
    sysd = {k: float(abs(x)) for k, x in systematic.items() if x is not None and np.isfinite(x)}
    biggest = max(sysd, key=sysd.get) if sysd else None
    return {"n": int(v.size), "mean": float(v.mean()), "items_half": items_half, "sources_half": sources_half,
            "systematic": sysd, "largest_systematic": biggest,
            "largest_over_items_half": (sysd[biggest] / items_half) if biggest and items_half > 0 else None}


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", help="long CSV: item, source, system, value[, in_split]")
    ap.add_argument("--a", required=True, help="system whose advantage is tested")
    ap.add_argument("--b", required=True, help="reference system")
    ap.add_argument("--n-split", type=int, default=None,
                    help="with an in_split column: split size for the realizable radius over the pool")
    args = ap.parse_args(argv)
    df = pd.read_csv(args.results)
    wide = df.pivot_table(index="item", columns="system", values="value", aggfunc="first")
    meta = df.drop_duplicates("item").set_index("item")
    diff = wide[args.a] - wide[args.b]
    if "in_split" in meta:
        split = meta.index[meta["in_split"].astype(bool)]
        out = analyse(diff.reindex(split), meta["source"])
        out["realizable"] = analyse_with_pool(diff, meta["source"], split, args.n_split or len(split))
    else:
        out = analyse(diff, meta["source"])
    json.dump(out, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
