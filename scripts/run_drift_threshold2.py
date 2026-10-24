"""Calibrated thresholds under benign drift, calibrated in the unit in which they are evaluated.

Revision of run_drift_threshold.py after a third review. There, the threshold was the 99th
percentile of calibration *window* scores but was applied to interpolated *point* scores, and the
calibration windows directly followed the reference windows. Here, for Raw-1NN on every
TSB-AD-U-Eva series:

  * reference = training windows ending in the first half of the training part; calibration =
    training windows starting at least W points after the half (a gap of W, so no calibration
    window overlaps or touches a reference window); test windows are scored against the same
    reference;
  * calibration window scores are mapped to points over the calibration region with the same
    mapping as the test scores (to_points), and the threshold is the 99th percentile of those
    point scores (nominal FAR 1% on points); no exchangeability of calibration and test is assumed;
  * reported at that threshold: FAR (normal test points above it), point recall, event recall
    (anomalous segments with at least one alarm), median detection delay (points from segment
    start to its first alarm, detected segments only) and false alarm episodes (maximal runs of
    alarmed normal points; a persistent alarm counts once per normal stretch) per 1,000 normal
    test points;
  * operating-point analysis (uses test labels, not deployable): the threshold that gives a
    realized FAR of exactly alpha on the normal test points, alpha in {0.01, 0.05}, and the event
    recall, point recall and delay there, so the normalizations are compared at equal false alarms.

Drifts and normalizations as in run_drift_injection.py.

Output: results/official_protocol/TSB-AD-U/threshold2_shard<k>.json
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
ALPHAS = (0.01, 0.05)


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return list(zip(idx[::2], idx[1::2]))


def _at_threshold(pts: np.ndarray, thr: float, test: np.ndarray, labels: np.ndarray,
                  events: list[tuple[int, int]]) -> dict[str, float | None]:
    alarm = (pts > thr) & test
    normal, anom = test & (labels == 0), test & (labels == 1)
    delays = []
    for a, b in events:
        hits = np.flatnonzero(alarm[a:b])
        if hits.size:
            delays.append(int(hits[0]))
    false_runs = len(_segments(alarm & normal))
    return {"far": float(alarm[normal].mean()), "point_recall": float(alarm[anom].mean()),
            "event_recall": len(delays) / len(events),
            "median_delay": float(np.median(delays)) if delays else None,
            "false_episodes_per_1000": 1000.0 * false_runs / int(normal.sum())}


def run_series(path: Path) -> dict[str, Any] | None:
    df = pd.read_csv(path).dropna()
    x0 = df.iloc[:, 0].to_numpy(dtype=float)
    labels = df["Label"].astype(int).to_numpy()
    n_train = int(path.stem.split("_")[-3])
    n = x0.size
    test = np.arange(n) >= n_train
    normal_test, anom_test = test & (labels == 0), test & (labels == 1)
    if not anom_test.any() or not normal_test.any():
        return None
    stride, train_stride = strides(n, n_train)
    all_starts, train_starts = window_starts(n, stride), window_starts(n_train, train_stride)
    half = n_train // 2
    ref_starts = train_starts[train_starts + W <= half]
    cal_starts = train_starts[train_starts >= half + W]
    if len(ref_starts) < 4 or len(cal_starts) < 4:
        return None
    c0 = int(cal_starts[0])
    events = _segments(anom_test)
    rec: dict[str, Any] = {"series": path.name, "n": n, "n_train": n_train,
                           "n_events": len(events), "n_ref": len(ref_starts),
                           "n_cal": len(cal_starts)}
    for drift in DRIFTS:
        x = inject(x0, n_train, drift)
        mu, sd = x[:n_train].mean(), x[:n_train].std() + 1e-8
        for norm in NORMS:
            key = f"{drift}|{norm}"
            try:
                z = rolling_z(x, n_train) if norm == "rolling" else (x - mu) / sd
                aw = windows_at(z[:, None], all_starts).reshape(len(all_starts), -1)
                rw = windows_at(z[:, None], ref_starts).reshape(len(ref_starts), -1)
                cw = windows_at(z[:, None], cal_starts).reshape(len(cal_starts), -1)
                if norm == "per_window":
                    aw, rw, cw = _window_z(aw), _window_z(rw), _window_z(cw)
                cal_pts = to_points(_nn_distance(rw, cw), cal_starts - c0, n_train - c0)
                thr = float(np.quantile(cal_pts, QUANTILE))
                pts = to_points(_nn_distance(rw, aw), all_starts, n)
                out = {"calibrated": _at_threshold(pts, thr, test, labels, events)}
                for alpha in ALPHAS:
                    t_a = float(np.quantile(pts[normal_test], 1 - alpha))
                    out[f"equal_far_{alpha}"] = _at_threshold(pts, t_a, test, labels, events)
                rec[key] = out
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
    out_path = OUT / subset / f"threshold2_shard{args.shard}.json"
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
