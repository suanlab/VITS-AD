"""Wall-clock cost of the arms in run_official_protocol.py, broken down by stage.

For a fixed sample of TSB-AD evaluation series (the first N of each split, in TSB-AD's listed
order), times each stage separately on the same machine: building windows, rendering line plots
(CPU, matplotlib), encoding them with frozen DINOv2-base (GPU), fitting the Woodbury Ledoit-Wolf
model, and scoring. Each stage is timed once per series after one warm-up series.

Output: artifacts/official_protocol/cost.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from scripts.run_official_protocol import (
    DATA, OFFICIAL, SPLITS, W, strides, window_starts, windows_at,
)
from src.models.backbone import VisionBackbone
from src.rendering.line_plot import render_line_plot
from src.scoring.woodbury_mahalanobis import WoodburyLedoitWolfScorer

LOGGER = logging.getLogger(__name__)
ART = Path("artifacts/official_protocol")
BATCH = 64


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_series(path: Path, backbone: VisionBackbone) -> dict[str, Any]:
    df = pd.read_csv(path).dropna()
    data = df.iloc[:, 0:-1].to_numpy(dtype=float)
    n_train = int(path.stem.split("_")[-3])
    n, d = data.shape
    t0 = time.perf_counter()
    mu, sd = data[:n_train].mean(axis=0), data[:n_train].std(axis=0) + 1e-8
    z = (data - mu) / sd
    stride, train_stride = strides(n, n_train)
    all_starts, train_starts = window_starts(n, stride), window_starts(n_train, train_stride)
    all_w, train_w = windows_at(z, all_starts), windows_at(z, train_starts)
    t_window = time.perf_counter() - t0

    t0 = time.perf_counter()
    model = WoodburyLedoitWolfScorer().fit(train_w.reshape(len(train_w), -1))
    t_raw_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    model.score(all_w.reshape(len(all_w), -1))
    t_raw_score = time.perf_counter() - t0

    t_render = t_encode = 0.0
    feats = []
    for i in range(0, len(all_w), BATCH):
        t0 = time.perf_counter()
        imgs = np.stack([render_line_plot(w.astype(np.float32)) for w in all_w[i:i + BATCH]])
        t_render += time.perf_counter() - t0
        _sync()
        t0 = time.perf_counter()
        feats.append(backbone.extract_patch_tokens_from_numpy(imgs).mean(axis=1))
        _sync()
        t_encode += time.perf_counter() - t0
    feats_all = np.concatenate(feats).astype(np.float64)
    # The training windows are a subset of the scored windows at the same stride in most series;
    # timing a fit on an equally sized slice of the features measures the fitting cost.
    t0 = time.perf_counter()
    vm = WoodburyLedoitWolfScorer().fit(feats_all[: len(train_w)])
    t_vis_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    vm.score(feats_all)
    t_vis_score = time.perf_counter() - t0
    return {"series": path.name, "n": n, "channels": d, "n_windows": int(len(all_w)),
            "n_train_windows": int(len(train_w)), "windowing_s": t_window,
            "raw_flatten_fit_s": t_raw_fit, "raw_flatten_score_s": t_raw_score,
            "render_s": t_render, "encode_s": t_encode,
            "vision_fit_s": t_vis_fit, "vision_score_s": t_vis_score}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-split", type=int, default=10)
    args = ap.parse_args()
    backbone = VisionBackbone("facebook/dinov2-base")
    records: list[dict[str, Any]] = []
    for subset, listing in SPLITS.items():
        files = pd.read_csv(OFFICIAL / listing)["file_name"].tolist()[: args.per_split + 1]
        for k, f in enumerate(files):
            rec = time_series(DATA / subset / f, backbone)
            if k == 0:
                continue  # warm-up
            rec["subset"] = subset
            records.append(rec)
            LOGGER.info("%s %s", subset, json.dumps({k: (round(v, 3) if isinstance(v, float)
                                                          else v) for k, v in rec.items()}))
    df = pd.DataFrame(records)
    summary: dict[str, Any] = {"device": torch.cuda.get_device_name(0)
                               if torch.cuda.is_available() else "cpu",
                               "per_series": records}
    for subset, g in df.groupby("subset"):
        per_kwin = {c: float((g[c] / g["n_windows"]).median() * 1000)
                    for c in ("raw_flatten_fit_s", "raw_flatten_score_s", "render_s",
                              "encode_s", "vision_fit_s", "vision_score_s")}
        raw_total = g["windowing_s"] + g["raw_flatten_fit_s"] + g["raw_flatten_score_s"]
        vis_total = (g["windowing_s"] + g["render_s"] + g["encode_s"] + g["vision_fit_s"]
                     + g["vision_score_s"])
        summary[subset] = {"n_series": int(len(g)),
                           "median_windows": float(g["n_windows"].median()),
                           "median_seconds_per_1000_windows": per_kwin,
                           "median_total_s_raw_flatten": float(raw_total.median()),
                           "median_total_s_vision": float(vis_total.median()),
                           "median_speedup": float((vis_total / raw_total).median())}
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "cost.json").write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.info("wrote %s", ART / "cost.json")


if __name__ == "__main__":
    main()
