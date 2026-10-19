"""Pre-registered H5: false-alarm rate at a calibrated threshold under benign drift.

Ranking measures such as VUS-PR compare scores within a series, so a drift that shifts the whole
test part can leave them unchanged. A deployed detector uses a fixed threshold instead. Here
Raw-1NN (global or per-window normalization) is calibrated on its own training part, scored out
of sample (2-fold, contiguous halves), with the threshold at the 99th percentile of those
training-window scores. The false-alarm rate (FAR) is the fraction of normal test points (label 0,
after the training part) whose point score exceeds the threshold. Drifts and windowing follow
run_drift_injection.py (none, +1 and +3 sigma level shifts, doubled scale, 3-sigma trend).

Pre-registration: artifacts/preregistration/normalization_hypotheses.md (H5).
Output: results/official_protocol/TSB-AD-U/far_shard<k>.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.run_drift_injection import DRIFTS, inject
from scripts.run_official_protocol import (
    DATA,
    OFFICIAL,
    OUT,
    SPLITS,
    W,
    _nn_distance,
    _window_z,
    strides,
    to_points,
    window_starts,
    windows_at,
)

LOGGER = logging.getLogger(__name__)
NORMS = ("global", "per_window")
QUANTILE = 0.99


def _oos_train_scores(tw: np.ndarray) -> np.ndarray:
    half = len(tw) // 2
    a, b = tw[:half], tw[half:]
    return np.concatenate([_nn_distance(b, a), _nn_distance(a, b)])


def run_series(path: Path) -> dict[str, Any] | None:
    df = pd.read_csv(path).dropna()
    x0 = df.iloc[:, 0].to_numpy(dtype=float)
    labels = df["Label"].astype(int).to_numpy()
    n_train = int(path.stem.split("_")[-3])
    n = x0.size
    if n_train < 2 * W + 4 or labels.max() == 0:
        return None
    stride, train_stride = strides(n, n_train)
    all_starts, train_starts = window_starts(n, stride), window_starts(n_train, train_stride)
    if len(train_starts) < 4:
        return None
    normal_test = (np.arange(n) >= n_train) & (labels == 0)
    rec: dict[str, Any] = {"series": path.name, "n": n, "n_train": n_train,
                           "n_normal_test": int(normal_test.sum())}
    for drift in DRIFTS:
        x = inject(x0, n_train, drift)
        mu, sd = x[:n_train].mean(), x[:n_train].std() + 1e-8
        z = (x - mu) / sd
        aw = windows_at(z[:, None], all_starts).reshape(len(all_starts), -1)
        tw = windows_at(z[:, None], train_starts).reshape(len(train_starts), -1)
        for norm in NORMS:
            key = f"{drift}|{norm}"
            try:
                a, t = (_window_z(aw), _window_z(tw)) if norm == "per_window" else (aw, tw)
                thr = float(np.quantile(_oos_train_scores(t), QUANTILE))
                pts = to_points(_nn_distance(t, a), all_starts, n)
                rec[key] = float((pts[normal_test] > thr).mean()) if normal_test.any() else None
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
    out_path = OUT / subset / f"far_shard{args.shard}.json"
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
