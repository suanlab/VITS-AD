"""Calibrated-threshold behaviour under benign drift, with an exchangeable calibration and recall.

Follow-up to the pre-registered H5 (run_drift_far.py), added after review: H5 reported only the
false-alarm rate (FAR), its calibration scored the training part with a reference half the size of
the one used for the test part, and it did not include the trailing-reference normalization.

Here, for Raw-1NN on every TSB-AD-U-Eva series:
  * reference = first half of the training windows; calibration = second half of the training
    windows, scored by their distance to the reference; test windows are scored against the same
    reference, so without drift calibration and test scores are exchangeable;
  * threshold = 99th percentile of the calibration scores (nominal FAR 1%);
  * FAR = fraction of normal test points (label 0, after the training part) above the threshold;
    point recall = fraction of anomalous test points above it; event recall = fraction of anomalous
    test segments with at least one point above it;
  * normalizations: global, per_window, rolling (as in run_drift_injection.py); drifts likewise.

Output: results/official_protocol/TSB-AD-U/threshold_shard<k>.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.run_drift_injection import DRIFTS, NORMS, inject, rolling_z
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
QUANTILE = 0.99


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return list(zip(idx[::2], idx[1::2]))


def run_series(path: Path) -> dict[str, Any] | None:
    df = pd.read_csv(path).dropna()
    x0 = df.iloc[:, 0].to_numpy(dtype=float)
    labels = df["Label"].astype(int).to_numpy()
    n_train = int(path.stem.split("_")[-3])
    n = x0.size
    test = np.arange(n) >= n_train
    normal_test, anom_test = test & (labels == 0), test & (labels == 1)
    if n_train < 2 * W + 4 or not anom_test.any() or not normal_test.any():
        return None
    stride, train_stride = strides(n, n_train)
    all_starts, train_starts = window_starts(n, stride), window_starts(n_train, train_stride)
    if len(train_starts) < 4:
        return None
    half = len(train_starts) // 2
    events = _segments(anom_test)
    rec: dict[str, Any] = {"series": path.name, "n": n, "n_train": n_train,
                           "n_events": len(events)}
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
                ref, cal = tw[:half], tw[half:]
                thr = float(np.quantile(_nn_distance(ref, cal), QUANTILE))
                pts = to_points(_nn_distance(ref, aw), all_starts, n)
                alarm = pts > thr
                rec[key] = {"far": float(alarm[normal_test].mean()),
                            "point_recall": float(alarm[anom_test].mean()),
                            "event_recall": float(np.mean([alarm[a:b].any() for a, b in events]))}
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
    out_path = OUT / subset / f"threshold_shard{args.shard}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
