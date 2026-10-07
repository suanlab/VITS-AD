"""Score our arms on TSB-AD's evaluation split under TSB-AD's own protocol.

This replaces the window-level protocol of earlier runs. It follows TSB-AD's benchmark scripts
(benchmark_exp/Run_Detector_{U,M}.py) exactly where they define the protocol:

  * the whole series is scored, train part included; models are fitted on the train part only;
  * labels are the original point labels; nothing is aggregated into windows for evaluation;
  * metrics come from TSB_AD.evaluation.metrics.get_metrics with
    slidingWindow = find_length_rank(data[:, 0], rank=1), and scores are not rescaled.

Our arms are window-based, so each window of W=100 steps is scored and window scores are mapped
to points exactly as TSB-AD's own window-based detectors do (score at the window centre, edges
padded), interpolating between centres when the stride exceeds 1. The stride is chosen so that a series yields about
3,000 windows, and is never larger than W, so every point is covered. The train part uses the
same stride, reduced if needed so that at least 100 training windows exist.

Arms:
  raw_meanpool  Ledoit-Wolf Mahalanobis on the window mean (D dims)
  raw_flatten   Ledoit-Wolf Mahalanobis on the flattened window (W*D dims), via the Woodbury
                form, so no dimension cap is needed
  vision        line-plot rendering -> frozen DINOv2-base -> mean-pooled patch tokens ->
                Ledoit-Wolf Mahalanobis (Woodbury)
  random_vision the same with randomly initialised encoder weights (optional)

Because TSB-AD's own per-series results for 40 detectors are published for this split
(benchmark_exp/benchmark_eval_results), every arm here is directly comparable with them.

Outputs: results/official_protocol/<subset>/per_series_shard<k>.json, and per-window scores in
results/official_protocol/<subset>/scores/<series>.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.scoring.woodbury_mahalanobis import WoodburyLedoitWolfScorer

LOGGER = logging.getLogger(__name__)
W = 100
TARGET_WINDOWS = 3000
MIN_TRAIN_WINDOWS = 100
DATA = Path("data/tsb_ad")
OFFICIAL = DATA / "official"
OUT = Path("results/official_protocol")
SPLITS = {"TSB-AD-U": "TSB-AD-U-Eva.csv", "TSB-AD-M": "TSB-AD-M-Eva.csv"}


def window_starts(n: int, stride: int) -> np.ndarray:
    """Start indices of W-long windows covering [0, n), the last one flush with the end."""
    if n < W:
        return np.array([], dtype=np.int64)
    starts = np.arange(0, n - W + 1, stride, dtype=np.int64)
    if starts[-1] != n - W:
        starts = np.append(starts, n - W)
    return starts


def windows_at(x: np.ndarray, starts: np.ndarray) -> np.ndarray:
    idx = starts[:, None] + np.arange(W)[None, :]
    return x[idx]                                           # (n_windows, W, D)


def to_points(win_scores: np.ndarray, starts: np.ndarray, n: int) -> np.ndarray:
    """Map window scores to points the way TSB-AD's window-based detectors do.

    TSB-AD's subsequence detectors (Sub-PCA, Sub-MCD, ...) score windows at stride 1, assign
    each window's score to its centre point (offset ceil((W-1)/2) from the start) and pad both
    ends with the edge values. At stride > 1 we interpolate linearly between window centres,
    which reduces to exactly that convention at stride 1.
    """
    centres = starts + math.ceil((W - 1) / 2)
    return np.interp(np.arange(n), centres, win_scores)


def to_points_max(win_scores: np.ndarray, starts: np.ndarray, n: int) -> np.ndarray:
    """Alternative mapping (sensitivity analysis only): max over the windows covering a point."""
    out = np.full(n, -np.inf)
    for s, v in zip(starts, win_scores):
        np.maximum(out[s:s + W], v, out=out[s:s + W])
    return out


def strides(n_total: int, n_train: int) -> tuple[int, int]:
    stride = int(min(W, max(1, math.ceil(n_total / TARGET_WINDOWS))))
    train_room = max(1, (n_train - W) // MIN_TRAIN_WINDOWS)
    return stride, int(max(1, min(stride, train_room)))


def official_metrics(scores: np.ndarray, labels: np.ndarray, sliding: int) -> dict[str, float]:
    from TSB_AD.evaluation.metrics import get_metrics

    return {k: float(v) for k, v in get_metrics(scores, labels, slidingWindow=sliding).items()}


def _vision_encoder(random_init: bool) -> Callable[[np.ndarray], np.ndarray]:
    from scripts.run_tsb_ad_vision_paired import _vision_features
    from src.models.backbone import VisionBackbone

    backbone = VisionBackbone("facebook/dinov2-base")
    if random_init:
        from scripts.run_random_backbone_control import build_random_backbone
        backbone = build_random_backbone()
    return lambda w: _vision_features(backbone, w)


def run_series(path: Path, arms: list[str], encoders: dict[str, Callable],
               tag: str = "", score_stride: int | None = None,
               crossfit: bool = False) -> dict[str, Any] | None:
    from TSB_AD.utils.slidingWindows import find_length_rank

    df = pd.read_csv(path).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    labels = df["Label"].astype(int).to_numpy()
    n_train = int(path.stem.split("_")[-3])
    n, d = data.shape
    if n_train < W + 2 or labels.max() == 0:
        return None
    sliding = int(find_length_rank(data[:, 0].reshape(-1, 1), rank=1))
    mu = data[:n_train].mean(axis=0)
    sd = data[:n_train].std(axis=0) + 1e-8
    z = (data - mu) / sd
    stride, train_stride = strides(n, n_train)
    if score_stride is not None:  # sensitivity runs: score every window at a fixed stride
        stride = int(score_stride)
    all_starts = window_starts(n, stride)
    train_starts = window_starts(n_train, train_stride)
    all_w = windows_at(z, all_starts)
    train_w = windows_at(z, train_starts)
    rec: dict[str, Any] = {"series": path.name, "subset": path.parent.name, "n": n,
                           "n_channels": d, "n_train": n_train, "stride": stride,
                           "train_stride": train_stride, "n_windows": int(all_starts.size),
                           "n_train_windows": int(train_starts.size), "sliding_window": sliding}
    feats: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "raw_meanpool": lambda w: w.mean(axis=1),
        "raw_flatten": lambda w: w.reshape(w.shape[0], -1),
        **encoders,
    }
    saved: dict[str, np.ndarray] = {"starts": all_starts}
    for arm in arms:
        try:
            ftr, fall = feats[arm](train_w), feats[arm](all_w)
            win = WoodburyLedoitWolfScorer().fit(ftr).score(fall)
            if crossfit:
                # Sensitivity: windows lying inside the train part are scored out of sample, by
                # a model fitted on the other half of the training windows (2-fold, contiguous),
                # so the train part gets no in-sample advantage over the test part.
                inside = all_starts + W <= n_train
                half = len(ftr) // 2
                folds = [(np.arange(half), np.arange(half, len(ftr))),
                         (np.arange(half, len(ftr)), np.arange(half))]
                cut = train_starts[half] if half < len(train_starts) else n_train
                for fit_idx, _ in folds:
                    model = WoodburyLedoitWolfScorer().fit(ftr[fit_idx])
                    fit_lo = train_starts[fit_idx[0]]
                    target = inside & ((all_starts < cut) if fit_lo >= cut else (all_starts >= cut))
                    if target.any():
                        win[target] = model.score(fall[target])
            pts = to_points(win, all_starts, n)
            if not np.all(np.isfinite(pts)):
                raise ValueError("non-finite point scores")
            rec[arm] = official_metrics(pts, labels, sliding)
            saved[arm] = win.astype(np.float32)
        except Exception as exc:
            LOGGER.warning("%s [%s] failed: %s: %s", path.name, arm, type(exc).__name__, str(exc)[:160])
            rec[arm] = None
    score_dir = OUT / path.parent.name / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".{tag}" if tag else ""
    np.savez_compressed(score_dir / f"{path.stem}{suffix}.npz", **saved)
    return rec


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=sorted(SPLITS), required=True)
    ap.add_argument("--arms", default="raw_meanpool,raw_flatten")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="", help="suffix for the output file, e.g. 'vision'")
    ap.add_argument("--score-stride", type=int, default=None,
                    help="override the scoring stride (training windows keep the default stride)")
    ap.add_argument("--crossfit-train", action="store_true",
                    help="score windows inside the train part out of sample (2-fold)")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    encoders: dict[str, Callable] = {}
    if "vision" in arms:
        encoders["vision"] = _vision_encoder(random_init=False)
    if "random_vision" in arms:
        encoders["random_vision"] = _vision_encoder(random_init=True)

    files = pd.read_csv(OFFICIAL / SPLITS[args.subset])["file_name"].tolist()
    files = [f for f in files if (DATA / args.subset / f).exists()]
    if args.limit:
        files = files[: args.limit]
    mine = [f for i, f in enumerate(files) if i % args.num_shards == args.shard]
    out_dir = OUT / args.subset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"per_series{('_' + args.tag) if args.tag else ''}_shard{args.shard}.json"
    records = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {r["series"] for r in records}
    LOGGER.info("%s shard %d/%d: %d series (%d done), arms=%s", args.subset, args.shard,
                args.num_shards, len(mine), len(done), arms)
    for i, f in enumerate(mine, start=1):
        if f in done:
            continue
        rec = run_series(DATA / args.subset / f, arms, encoders, args.tag, args.score_stride,
                         args.crossfit_train)
        if rec is not None:
            records.append(rec)
        out_path.write_text(json.dumps(records, indent=2) + "\n")
        LOGGER.info("%d/%d %s", i, len(mine), f)


if __name__ == "__main__":
    main()
