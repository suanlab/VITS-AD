"""Decompose the gap between TSB-AD's Sub-MCD and our raw_flatten control (TSB-AD-U-Eva).

Both are Mahalanobis distances of sliding windows fitted on the train part, yet TSB-AD reports
Sub-MCD at 0.237 VUS-PR while raw_flatten scores 0.505 under the same protocol. They differ in
three ingredients, which this script crosses in a full factorial design:

  norm       global   z-score with train-part statistics (raw_flatten; keeps amplitude)
             window_z each window z-scored by its own mean/std, ddof=1 (TSB-AD's MCD/Sub-MCD)
             window_minmax each window min-max scaled to [0, 1] (what our line-plot renderer
                      does before drawing; constant windows -> 0.5)
  estimator  lw       Ledoit-Wolf covariance (raw_flatten)
             mcd      Minimum Covariance Determinant, sklearn MinCovDet, random_state=2024
                      (TSB-AD's Sub-MCD with its tuned support_fraction=None)
  window     w100     W = 100 (raw_flatten)
             period3  find_length_rank(series, rank=3) (TSB-AD's tuned Sub-MCD, periodicity=3)

Everything else is held fixed at TSB-AD's Sub-MCD convention: windows at stride 1, fitted on the
train part, every window of the whole series scored, the score assigned to the window centre and
edges padded, metrics from TSB-AD's get_metrics with slidingWindow = find_length_rank(rank=1).
Cell (window_z, mcd, period3) re-implements Sub-MCD and is checked against TSB-AD's published
per-series values.

Output: results/official_protocol/TSB-AD-U/ablation_shard<k>.json
"""

from __future__ import annotations

import argparse
import itertools
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.covariance import LedoitWolf, MinCovDet

from scripts.run_contract import open_run, require_inputs, save_records
from scripts.run_official_protocol import DATA, OFFICIAL, OUT, SPLITS, official_metrics

LOGGER = logging.getLogger(__name__)
NORMS = ("global", "window_z", "window_minmax")
ESTIMATORS = ("lw", "mcd")
WINDOWS = ("w100", "period3")
CELLS = list(itertools.product(NORMS, ESTIMATORS, WINDOWS))


def normalise(win: np.ndarray, norm: str) -> np.ndarray:
    """Apply the per-window normalisation of a cell (global z-scoring happens before windowing)."""
    if norm == "global":
        return win
    if norm == "window_z":
        mu = win.mean(axis=1, keepdims=True)
        sd = win.std(axis=1, ddof=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (win - mu) / sd
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    lo = win.min(axis=1, keepdims=True)
    span = win.max(axis=1, keepdims=True) - lo
    safe = np.where(np.isclose(span, 0.0), 1.0, span)
    return np.where(np.isclose(span, 0.0), 0.5, (win - lo) / safe)


def centre_pad(win_scores: np.ndarray, n: int, w: int) -> np.ndarray:
    """TSB-AD's convention: window score at its centre, both ends padded with edge values."""
    head = math.ceil((w - 1) / 2)
    tail = n - head - win_scores.size
    return np.concatenate([np.full(head, win_scores[0]), win_scores, np.full(tail, win_scores[-1])])


def fit_score(train: np.ndarray, allw: np.ndarray, estimator: str) -> np.ndarray:
    if estimator == "lw":
        model = LedoitWolf().fit(train)
    else:
        model = MinCovDet(support_fraction=None, random_state=2024).fit(train)
    return model.mahalanobis(allw)


def run_series(path: Path) -> dict[str, Any] | None:
    from TSB_AD.utils.slidingWindows import find_length_rank

    df = pd.read_csv(path).dropna()
    x = df.iloc[:, 0].to_numpy(dtype=float)
    labels = df["Label"].astype(int).to_numpy()
    n_train = int(path.stem.split("_")[-3])
    n = x.size
    if labels.max() == 0:
        return None
    sliding = int(find_length_rank(x.reshape(-1, 1), rank=1))
    widths = {"w100": 100, "period3": int(find_length_rank(x.reshape(-1, 1), rank=3))}
    mu, sd = x[:n_train].mean(), x[:n_train].std() + 1e-8
    rec: dict[str, Any] = {"series": path.name, "n": n, "n_train": n_train,
                           "sliding_window": sliding, "period3": widths["period3"]}
    for norm, est, wname in CELLS:
        key = f"{norm}|{est}|{wname}"
        w = widths[wname]
        try:
            if n_train < w + 2:
                raise ValueError(f"train part {n_train} too short for window {w}")
            src = (x - mu) / sd if norm == "global" else x
            allw = normalise(sliding_window_view(src, w), norm)
            train = allw[: n_train - w + 1]
            pts = centre_pad(fit_score(train, allw, est), n, w)
            if not np.all(np.isfinite(pts)):
                raise ValueError("non-finite point scores")
            rec[key] = official_metrics(pts, labels, sliding)
        except Exception as exc:
            LOGGER.warning("%s [%s] failed: %s: %s", path.name, key, type(exc).__name__,
                           str(exc)[:160])
            rec[key] = None
            rec.setdefault("errors", {})[key] = f"{type(exc).__name__}: {str(exc)[:160]}"
    return rec


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--allow-missing", action="store_true",
                    help="run although some listed series are missing (logged in the fingerprint)")
    args = ap.parse_args()

    subset = "TSB-AD-U"
    files = pd.read_csv(OFFICIAL / SPLITS[subset])["file_name"].tolist()
    files, missing = require_inputs(files, DATA / subset, args.allow_missing)
    if args.limit:
        files = files[: args.limit]
    mine = [f for i, f in enumerate(files) if i % args.num_shards == args.shard]
    out_path = OUT / subset / f"ablation_shard{args.shard}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = open_run(out_path, {"limit": args.limit, "shard": [args.shard, args.num_shards]},
                       mine, __file__, OFFICIAL / SPLITS[subset], missing)
    done = {r["series"] for r in records}
    for i, f in enumerate(mine, start=1):
        if f in done:
            continue
        rec = run_series(DATA / subset / f)
        if rec is not None:
            records.append(rec)
        save_records(out_path, records)
        LOGGER.info("%d/%d %s", i, len(mine), f)


if __name__ == "__main__":
    main()
