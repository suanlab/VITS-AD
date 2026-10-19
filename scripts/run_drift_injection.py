"""When is global normalization wrong? Benign drift injected into TSB-AD-U-Eva test parts.

Global normalization assumes the training part's level and scale stay valid. TSB-AD's training
parts are anomaly-free prefixes and its labels count level shifts as anomalies, so the benchmark
never tests that assumption. Here every univariate evaluation series gets a *benign* change in
its test part (after the training part; labels unchanged), and Raw-1NN is scored under three
normalizations:

  global      z-score with training-part statistics (as in the paper)
  per_window  each window z-scored by its own mean and std
  rolling     each point z-scored by the mean and std of the preceding ROLL points
              (a trailing reference; training points use the training statistics)

Drifts (sigma = training-part standard deviation):
  none        unchanged series (reference)
  level1      +1 sigma level shift for the whole test part
  level3      +3 sigma level shift
  scale2      test-part deviations from the training mean doubled
  trend3      linear trend from 0 at the end of the training part to +3 sigma at the end

Everything else follows run_official_protocol.py (W=100, strides, centre mapping, TSB-AD metric).

Output: results/official_protocol/TSB-AD-U/drift_shard<k>.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.run_official_protocol import (
    DATA,
    OFFICIAL,
    OUT,
    SPLITS,
    W,
    _nn_distance,
    _window_z,
    official_metrics,
    strides,
    to_points,
    window_starts,
    windows_at,
)

LOGGER = logging.getLogger(__name__)
ROLL = 1000
DRIFTS = ("none", "level1", "level3", "scale2", "trend3")
NORMS = ("global", "per_window", "rolling")


def inject(x: np.ndarray, n_train: int, drift: str) -> np.ndarray:
    y = x.copy()
    mu, sd = x[:n_train].mean(), x[:n_train].std() + 1e-8
    m = x.size - n_train
    if drift == "level1":
        y[n_train:] += sd
    elif drift == "level3":
        y[n_train:] += 3 * sd
    elif drift == "scale2":
        y[n_train:] = mu + 2 * (y[n_train:] - mu)
    elif drift == "trend3":
        y[n_train:] += 3 * sd * np.arange(1, m + 1) / m
    return y


def rolling_z(x: np.ndarray, n_train: int) -> np.ndarray:
    mu, sd = x[:n_train].mean(), x[:n_train].std() + 1e-8
    s = pd.Series(x)
    rm = s.rolling(ROLL, min_periods=W).mean().shift(1).to_numpy()
    rs = s.rolling(ROLL, min_periods=W).std().shift(1).to_numpy()
    out = (x - mu) / sd
    t = np.arange(x.size) >= n_train
    ok = t & np.isfinite(rm) & (rs > 1e-8)
    out[ok] = (x[ok] - rm[ok]) / rs[ok]
    return out


def run_series(path: Path) -> dict[str, Any] | None:
    from TSB_AD.utils.slidingWindows import find_length_rank

    df = pd.read_csv(path).dropna()
    x0 = df.iloc[:, 0].to_numpy(dtype=float)
    labels = df["Label"].astype(int).to_numpy()
    n_train = int(path.stem.split("_")[-3])
    n = x0.size
    if n_train < W + 2 or labels.max() == 0:
        return None
    sliding = int(find_length_rank(x0.reshape(-1, 1), rank=1))
    stride, train_stride = strides(n, n_train)
    all_starts, train_starts = window_starts(n, stride), window_starts(n_train, train_stride)
    rec: dict[str, Any] = {"series": path.name, "n": n, "n_train": n_train}
    for drift in DRIFTS:
        x = inject(x0, n_train, drift)
        mu, sd = x[:n_train].mean(), x[:n_train].std() + 1e-8
        for norm in NORMS:
            key = f"{drift}|{norm}"
            try:
                z = rolling_z(x, n_train) if norm == "rolling" else (x - mu) / sd
                aw = windows_at(z[:, None], all_starts).reshape(len(all_starts), -1)
                tw = windows_at(z[:, None], train_starts).reshape(len(train_starts), -1)
                if norm == "per_window":
                    aw, tw = _window_z(aw), _window_z(tw)
                pts = to_points(_nn_distance(tw, aw), all_starts, n)
                rec[key] = official_metrics(pts, labels, sliding)["VUS-PR"]
            except Exception as exc:
                LOGGER.warning("%s [%s] failed: %s: %s", path.name, key, type(exc).__name__,
                               str(exc)[:160])
                rec[key] = None
    return rec


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    subset = "TSB-AD-U"
    files = pd.read_csv(OFFICIAL / SPLITS[subset])["file_name"].tolist()
    files = [f for f in files if (DATA / subset / f).exists()][: args.limit]
    mine = [f for i, f in enumerate(files) if i % args.num_shards == args.shard]
    out_path = OUT / subset / f"drift_shard{args.shard}.json"
    records = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {r["series"] for r in records}
    for i, f in enumerate(mine, start=1):
        if f in done:
            continue
        rec = run_series(DATA / subset / f)
        if rec is not None:
            records.append(rec)
        out_path.write_text(json.dumps(records, indent=2) + "\n")
        LOGGER.info("%d/%d %s", i, len(mine), f)


if __name__ == "__main__":
    main()
