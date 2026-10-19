"""Wall-clock cost of the arms in run_official_protocol.py, by stage, plus a scaling study.

Timing protocol (chosen because the machine is a shared server):
  * BLAS/OpenMP threads are pinned (default 4) before numpy is imported;
  * each stage is run --repeats times (default 5) and the minimum is reported, the usual
    estimator of a computation's cost when other jobs add only positive noise; the median is
    stored as well;
  * series are drawn across sources: per subset, up to --per-source series from every source
    dataset in TSB-AD's evaluation list, chosen with a fixed seed;
  * fit cost is normalized by the number of *training* windows, scoring cost by the number of
    scored windows.

Stages: raw-flatten (Woodbury Ledoit-Wolf) fit and score; dense Ledoit-Wolf fit; global-z
1-NN index build and query; line-plot rendering (CPU); DINOv2-base encoding (GPU).

Scaling study (--scaling): synthetic Gaussian windows, fit time of the Woodbury and dense
Ledoit-Wolf forms as the number of training windows n and the dimension d vary.

Output: artifacts/official_protocol/cost.json
"""

from __future__ import annotations

import os

THREADS = os.environ.get("BENCH_THREADS", "4")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = THREADS

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402

from scripts.run_official_protocol import (  # noqa: E402
    DATA,
    OFFICIAL,
    SPLITS,
    _nn_distance,
    strides,
    window_starts,
    windows_at,
)
from src.scoring.woodbury_mahalanobis import WoodburyLedoitWolfScorer  # noqa: E402

LOGGER = logging.getLogger(__name__)
ART = Path("artifacts/official_protocol")
BATCH = 64


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time(fn: Callable[[], Any], repeats: int) -> tuple[float, float]:
    ts = []
    for _ in range(repeats):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append(time.perf_counter() - t0)
    return float(min(ts)), float(np.median(ts))


def sample_series(subset: str, per_source: int, seed: int = 0) -> list[str]:
    files = pd.read_csv(OFFICIAL / SPLITS[subset])["file_name"].tolist()
    files = [f for f in files if (DATA / subset / f).exists()]
    by_src: dict[str, list[str]] = {}
    for f in files:
        by_src.setdefault(f.split("_")[1], []).append(f)
    rng = np.random.default_rng(seed)
    out: list[str] = []
    for src in sorted(by_src):
        pool = by_src[src]
        out += [pool[i] for i in sorted(rng.choice(len(pool), min(per_source, len(pool)),
                                                   replace=False))]
    return out


def time_series(path: Path, backbone: Any, repeats: int) -> dict[str, Any]:
    from src.rendering.line_plot import render_line_plot

    df = pd.read_csv(path).dropna()
    data = df.iloc[:, 0:-1].to_numpy(dtype=float)
    n_train = int(path.stem.split("_")[-3])
    n, d = data.shape
    mu, sd = data[:n_train].mean(axis=0), data[:n_train].std(axis=0) + 1e-8
    z = (data - mu) / sd
    stride, train_stride = strides(n, n_train)
    all_starts, train_starts = window_starts(n, stride), window_starts(n_train, train_stride)
    xa = windows_at(z, all_starts).reshape(len(all_starts), -1)
    xt = windows_at(z, train_starts).reshape(len(train_starts), -1)
    rec: dict[str, Any] = {"series": path.name, "n": n, "channels": d, "dim": int(xa.shape[1]),
                           "n_windows": int(len(xa)), "n_train_windows": int(len(xt))}
    model = WoodburyLedoitWolfScorer().fit(xt)
    rec["woodbury_fit_s"] = _time(lambda: WoodburyLedoitWolfScorer().fit(xt), repeats)
    rec["woodbury_score_s"] = _time(lambda: model.score(xa), repeats)
    if xa.shape[1] <= 2000:
        rec["dense_lw_fit_s"] = _time(lambda: LedoitWolf().fit(xt), repeats)
    rec["knn_s"] = _time(lambda: _nn_distance(xt, xa), repeats)
    if backbone is not None:
        wins = windows_at(z, all_starts).astype(np.float32)
        sub = wins[: min(len(wins), 512)]  # rendering/encoding cost is per window
        imgs = np.stack([render_line_plot(w) for w in sub[:BATCH]])
        rec["render_per_window_s"] = [
            v / len(sub) for v in _time(lambda: [render_line_plot(w) for w in sub], repeats)]
        rec["encode_per_window_s"] = [
            v / BATCH for v in _time(lambda: backbone.extract_patch_tokens_from_numpy(imgs),
                                     repeats)]
        rec["vision_window_subset"] = int(len(sub))
    return rec


def scaling(repeats: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(0)
    out = []
    for d in (100, 500, 2000, 5500):
        for n in (250, 500, 1000, 2000, 4000):
            x = rng.standard_normal((n, d))
            row: dict[str, Any] = {"n": n, "d": d,
                                   "woodbury_fit_s": _time(
                                       lambda: WoodburyLedoitWolfScorer().fit(x), repeats)[0]}
            if d <= 2000:
                row["dense_lw_fit_s"] = _time(lambda: LedoitWolf().fit(x), repeats)[0]
            out.append(row)
            LOGGER.info("scaling %s", row)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--scaling", action="store_true")
    args = ap.parse_args()
    backbone = None
    if not args.no_vision:
        import torch

        from src.models.backbone import VisionBackbone

        backbone = VisionBackbone("facebook/dinov2-base")
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    else:
        device = "none"
    summary: dict[str, Any] = {"threads": int(THREADS), "repeats": args.repeats,
                               "statistic": "minimum over repeats (median also stored)",
                               "gpu": device, "per_series": []}
    for subset in SPLITS:
        files = sample_series(subset, args.per_source)
        for f in files:
            rec = time_series(DATA / subset / f, backbone, args.repeats)
            rec["subset"] = subset
            summary["per_series"].append(rec)
            LOGGER.info("%s %s", subset, f)
    df = pd.DataFrame(summary["per_series"])
    for subset, g in df.groupby("subset"):
        best = lambda c: g[c].map(lambda v: v[0])  # noqa: E731  (minimum over repeats)
        s: dict[str, Any] = {
            "n_series": int(len(g)), "sources": sorted({x.split("_")[1] for x in g["series"]}),
            "median_ms_per_train_window_woodbury_fit": float(
                (best("woodbury_fit_s") / g["n_train_windows"]).median() * 1e3),
            "median_ms_per_window_woodbury_score": float(
                (best("woodbury_score_s") / g["n_windows"]).median() * 1e3),
            "median_ms_per_window_knn": float((best("knn_s") / g["n_windows"]).median() * 1e3),
            "median_s_per_series_raw_flatten": float(
                (best("woodbury_fit_s") + best("woodbury_score_s")).median()),
            "median_s_per_series_knn": float(best("knn_s").median()),
        }
        if "render_per_window_s" in g:
            render = g["render_per_window_s"].map(lambda v: v[0])
            encode = g["encode_per_window_s"].map(lambda v: v[0])
            s["median_ms_per_window_render"] = float(render.median() * 1e3)
            s["median_ms_per_window_encode"] = float(encode.median() * 1e3)
            vision_total = (render + encode) * g["n_windows"] + best("woodbury_fit_s") \
                + best("woodbury_score_s")
            raw_total = best("woodbury_fit_s") + best("woodbury_score_s")
            s["median_s_per_series_vision"] = float(vision_total.median())
            s["median_ratio_vision_over_raw_flatten"] = float((vision_total / raw_total).median())
            s["median_ratio_encode_only_over_raw_flatten"] = float(
                (encode * g["n_windows"] / raw_total).median())
        summary[subset] = s
    if args.scaling:
        summary["scaling"] = scaling(args.repeats)
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "cost.json").write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.info("wrote %s", ART / "cost.json")


if __name__ == "__main__":
    main()
