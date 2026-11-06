"""Calibrated thresholds under benign drift for TSB-AD's own Sub- detectors (registered T1-T2,
artifacts/preregistration/threshold_sub_hypothesis.md).

The Raw-1NN threshold study (run_drift_threshold2.py) is repeated with three of TSB-AD's per-window
Sub- detectors run through TSB-AD 1.5's wrappers with tuned hyperparameters: Sub-KNN, Sub-IForest
and Sub-HBOS, each with per-window and global normalization (the replacement of
run_detector_normalization.py). Benign drifts are injected into the test part as in
run_drift_injection.py (none, level +3 training standard deviations, linear trend to +3).

These detectors are fitted on, and score, the whole series, including the drifted test part: this
is an offline, transductive threshold diagnostic, not a test of a frozen detector and threshold
applied to future data. The threshold is the 99th percentile of
the point scores on the second half of the training part (calibration region), a nominal 1%
false-alarm rate; FAR, point and event recall, and false-alarm episodes are measured on the test part
as in run_drift_threshold2._at_threshold. As there, an equal-FAR operating point (1%, 5%) set with the
test labels is reported as a diagnostic.

Output: results/official_protocol/TSB-AD-U/threshold_sub_shard<k>.json, and with --list-eligible
threshold_sub_expected.json (the series that must have a record, and why the others are skipped)

    python -m scripts.run_threshold_sub --shard k --num-shards K
    python -m scripts.run_threshold_sub --list-eligible
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.run_contract import open_run, require_inputs, save_records
from scripts.run_detector_normalization import DATA, OFFICIAL, OUT, SPLITS, TUNED, _patch, _runner
from scripts.run_drift_injection import inject
from scripts.run_drift_threshold2 import _at_threshold, _segments

LOGGER = logging.getLogger(__name__)
DETECTORS = ("Sub_KNN", "Sub_IForest", "Sub_HBOS")
DRIFTS = ("none", "level3", "trend3")
QUANTILE = 0.99
ALPHAS = (0.01, 0.05)


def _skip_reason(labels: np.ndarray, n_train: int) -> str | None:
    """Why a series is not eligible (no anomalous or normal test point, calibration region < 20)."""
    n = labels.size
    test = np.arange(n) >= n_train
    if not (test & (labels == 1)).any():
        return "no anomalous test point"
    if not (test & (labels == 0)).any():
        return "no normal test point"
    if min(n_train, n) - n_train // 2 < 20:
        return "calibration region shorter than 20 points"
    return None


def run_series(path: Path) -> dict[str, Any] | None:
    df = pd.read_csv(path).dropna()
    x0 = df.iloc[:, 0].to_numpy(dtype=float)
    labels = df["Label"].astype(int).to_numpy()
    n_train = int(path.stem.split("_")[-3])
    n = x0.size
    test = np.arange(n) >= n_train
    normal_test, anom_test = test & (labels == 0), test & (labels == 1)
    cal = (np.arange(n) >= n_train // 2) & (np.arange(n) < n_train)
    if _skip_reason(labels, n_train) is not None:
        return None
    events = _segments(anom_test)
    rec: dict[str, Any] = {"series": path.name, "n": n, "n_train": n_train, "n_events": len(events)}
    for drift in DRIFTS:
        x = inject(x0, n_train, drift)
        mu, sd = x[:n_train].mean(), x[:n_train].std() + 1e-8
        for variant in ("window", "global"):
            _patch(variant)
            data = ((x - mu) / sd if variant == "global" else x).reshape(-1, 1)
            for det in DETECTORS:
                key = f"{drift}|{det}|{variant}"
                try:
                    pts = np.asarray(_runner(det)(data, **TUNED[det]), dtype=float).ravel()
                    if pts.size != n:
                        raise ValueError(f"score vector has size {pts.size}, series has {n}")
                    if not np.all(np.isfinite(pts)):
                        raise ValueError(f"{int((~np.isfinite(pts)).sum())} non-finite scores of {pts.size}")
                    thr = float(np.quantile(pts[cal], QUANTILE))
                    out = {"calibrated": _at_threshold(pts, thr, test, labels, events)}
                    for alpha in ALPHAS:
                        t_a = float(np.quantile(pts[normal_test], 1 - alpha))
                        out[f"equal_far_{alpha}"] = _at_threshold(pts, t_a, test, labels, events)
                    rec[key] = out
                except Exception as exc:
                    LOGGER.warning("%s [%s] failed: %s: %s", path.name, key, type(exc).__name__, str(exc)[:160])
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
    ap.add_argument("--only", default=None, help="comma-separated series of this shard to run")
    ap.add_argument("--part", default="", help="output-file suffix used with --only")
    ap.add_argument("--allow-missing", action="store_true",
                    help="run although some listed series are missing (logged in the fingerprint)")
    ap.add_argument("--list-eligible", action="store_true",
                    help="only write the expected series and the skip reasons (threshold_sub_expected.json)")
    args = ap.parse_args()
    subset = "TSB-AD-U"
    files = pd.read_csv(OFFICIAL / SPLITS[subset])["file_name"].tolist()
    files, missing = require_inputs(files, DATA / subset, args.allow_missing)
    if args.list_eligible:
        skipped = {}
        for f in files:
            df = pd.read_csv(DATA / subset / f, usecols=["Label"]).dropna()
            reason = _skip_reason(df["Label"].astype(int).to_numpy(), int(Path(f).stem.split("_")[-3]))
            if reason:
                skipped[f] = reason
        path = OUT / subset / "threshold_sub_expected.json"
        save_records(path, {"eligible": [f for f in files if f not in skipped], "skipped": skipped})
        LOGGER.info("wrote %s: %d eligible, %d skipped", path, len(files) - len(skipped), len(skipped))
        return
    files = files[: args.limit]
    mine = [f for i, f in enumerate(files) if i % args.num_shards == args.shard]
    if args.only:
        mine = [f for f in mine if f in set(args.only.split(","))]
    out_path = OUT / subset / f"threshold_sub_shard{args.shard}{args.part}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = open_run(out_path, {"detectors": DETECTORS, "drifts": DRIFTS, "quantile": QUANTILE,
                                  "alphas": ALPHAS, "hyperparameters": {d: TUNED[d] for d in DETECTORS},
                                  "limit": args.limit, "shard": [args.shard, args.num_shards]},
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
