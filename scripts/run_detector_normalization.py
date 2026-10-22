"""Does window normalization matter beyond the covariance family? (TSB-AD Eva splits)

TSB-AD's subsequence detectors share one preprocessing switch. In the code that produced
TSB-AD's published tables (uploaded 2024-11-29), Sub-KNN, Sub-LOF, Sub-IForest, Sub-OCSVM,
Sub-HBOS and Sub-MCD z-normalize each window by its own mean and standard deviation, while
Sub-PCA instead standardizes each window position across windows. This script runs each of
these seven detectors, with the tuned hyperparameters of TSB-AD's pinned HP_list
(Optimal_Uni_algo_HP_dict), under two normalizations that differ in nothing else:

  window  every window z-scored by its own mean and std (ddof=1, constant windows -> 0)
  global  the series z-scored once with train-part statistics, windows left as they are

Both variants run TSB-AD's own detector classes; only the normalization is replaced. For that,
each detector module's `zscore` is made the identity (disabling its built-in normalization,
whatever axis the installed release uses) and its `Window` is replaced by one that applies the
variant's normalization after windowing. Sub-PCA's internal column standardizer is part of
the model and is kept in both variants. Fitting follows TSB-AD: Sub-OCSVM and Sub-MCD are fitted
on the train part, the others on the whole series; the metric is TSB-AD's get_metrics.

The `window` variant re-implements the published Sub-KNN/LOF/IForest/OCSVM/HBOS/MCD and the
`global` variant the published Sub-PCA, which the summary checks against TSB-AD's tables.

With --subset TSB-AD-M the same is done for TSB-AD's multivariate detectors (KNN, LOF, IForest,
HBOS, OCSVM, MCD, PCA) with Optimal_Multi_algo_HP_dict. Most of them score single time steps
(slidingWindow=1), so the per-row normalization there z-scores each time step across its
channels; IForest and PCA use 100-step windows.

Output: results/official_protocol/<subset>/detnorm_shard<k>.json. Files named
detnorm*_shard<k>b.json hold series of shard k run by a second process (--only, --part b) to
finish a shard sooner; readers merge them by series name.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from scripts.run_official_protocol import (
    DATA,
    OFFICIAL,
    OUT,
    SPLITS,
    TUNING_SPLITS,
    official_metrics,
)

LOGGER = logging.getLogger(__name__)
VARIANTS = ("window", "global")
# Optimal_Uni_algo_HP_dict at TSB-AD commit 6beac72e11d1155ade40870492c00d0d1cfdcaaf
TUNED: dict[str, dict[str, Any]] = {
    "Sub_IForest": {"periodicity": 1, "n_estimators": 150},
    "Sub_LOF": {"periodicity": 2, "n_neighbors": 30},
    "Sub_PCA": {"periodicity": 1, "n_components": None},
    "Sub_HBOS": {"periodicity": 1, "n_bins": 10},
    "Sub_OCSVM": {"periodicity": 2, "kernel": "rbf"},
    "Sub_MCD": {"periodicity": 3, "support_fraction": None},
    "Sub_KNN": {"periodicity": 2, "n_neighbors": 50},
}
# Optimal_Multi_algo_HP_dict at the same commit
TUNED_MULTI: dict[str, dict[str, Any]] = {
    "IForest": {"n_estimators": 25, "max_features": 0.8},
    "LOF": {"n_neighbors": 50, "metric": "euclidean"},
    "PCA": {"n_components": 0.25},
    "HBOS": {"n_bins": 30, "tol": 0.5},
    "OCSVM": {"kernel": "rbf", "nu": 0.1},
    "MCD": {"support_fraction": 0.8},
    "KNN": {"n_neighbors": 50, "method": "mean"},
}
SEMI_SUPERVISED = {"Sub_OCSVM", "Sub_MCD", "OCSVM", "MCD"}
MODULES = ("KNN", "LOF", "IForest", "OCSVM", "HBOS", "MCD", "PCA")


def _window_z(w: np.ndarray) -> np.ndarray:
    mu = w.mean(axis=1, keepdims=True)
    sd = w.std(axis=1, ddof=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (w - mu) / sd
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _make_window_class(variant: str) -> type:
    class PatchedWindow:
        def __init__(self, window: int = 100, stride: int = 1) -> None:
            self.window, self.stride = window, stride

        def convert(self, x: np.ndarray) -> np.ndarray:
            shape = (x.shape[0] - (self.window - 1), -1)
            w = sliding_window_view(x, window_shape=self.window, axis=0).reshape(shape)
            w = np.ascontiguousarray(w[:: self.stride, :], dtype=float)
            return _window_z(w) if variant == "window" else w

    return PatchedWindow


def _check_is_fitted(estimator: Any, attributes: list[str] | None = None, **_: Any) -> None:
    """The attribute check TSB-AD's guard performs; scikit-learn >= 1.6's version requires
    ``__sklearn_tags__``, which TSB-AD's BaseDetector lacks (see run_tsb_ad_pool.py)."""
    missing = [a for a in (attributes or []) if not hasattr(estimator, a)]
    if missing:
        raise ValueError(f"{type(estimator).__name__} is not fitted: missing {missing}")


def _patch(variant: str) -> None:
    window_cls = _make_window_class(variant)
    for name in MODULES:
        mod = importlib.import_module(f"TSB_AD.models.{name}")
        mod.Window = window_cls
        if hasattr(mod, "zscore"):
            mod.zscore = lambda a, axis=0, ddof=0: np.asarray(a, dtype=float)
        if hasattr(mod, "check_is_fitted"):
            mod.check_is_fitted = _check_is_fitted


class _IdentityScaler:
    """Stands in for the MinMaxScaler that TSB-AD's OCSVM.fit applies to each window (rows)."""

    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float)


def _patch_ocsvm_minmax() -> None:
    """Remove OCSVM's fit-time per-window MinMax scaling (TSB-AD 1.5 applies it only in fit, not
    in decision_function; TSB-AD later removed it, commit 9179dee)."""
    mod = importlib.import_module("TSB_AD.models.OCSVM")
    mod.MinMaxScaler = _IdentityScaler


def _runner(name: str) -> Callable[..., np.ndarray]:
    wrapper = importlib.import_module("TSB_AD.model_wrapper")
    return getattr(wrapper, f"run_{name}")


def run_series(path: Path, detectors: list[str],
               tuned: dict[str, dict[str, Any]] = TUNED) -> dict[str, Any] | None:
    from TSB_AD.utils.slidingWindows import find_length_rank

    df = pd.read_csv(path).dropna()
    x = df.iloc[:, 0:-1].to_numpy(dtype=float)
    labels = df["Label"].astype(int).to_numpy()
    n_train = int(path.stem.split("_")[-3])
    if labels.max() == 0:
        return None
    sliding = int(find_length_rank(x[:, 0].reshape(-1, 1), rank=1))
    mu, sd = x[:n_train].mean(axis=0), x[:n_train].std(axis=0) + 1e-8
    rec: dict[str, Any] = {"series": path.name, "n": int(x.shape[0]), "n_train": n_train}
    for variant in VARIANTS:
        _patch(variant)
        data = (x - mu) / sd if variant == "global" else x
        for det in detectors:
            key = f"{det}|{variant}"
            try:
                run = _runner(det)
                if det in SEMI_SUPERVISED:
                    score = run(data[:n_train], data, **tuned[det])
                else:
                    score = run(data, **tuned[det])
                score = np.asarray(score, dtype=float).ravel()
                if score.size != labels.size or not np.all(np.isfinite(score)):
                    raise ValueError(f"bad score vector (size {score.size}, n {labels.size})")
                rec[key] = official_metrics(score, labels, sliding)
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
    ap.add_argument("--subset", choices=("TSB-AD-U", "TSB-AD-M"), default="TSB-AD-U")
    ap.add_argument("--detectors", default=None)
    ap.add_argument("--split", choices=("eva", "tuning", "rest"), default="eva",
                    help="TSB-AD's evaluation split (default), its tuning split (H6), or the "
                         "univariate series in neither split (H9)")
    ap.add_argument("--ocsvm-clean", action="store_true",
                    help="remove OCSVM's fit-time per-window MinMax scaling in both variants")
    ap.add_argument("--only", default=None,
                    help="comma-separated series of this shard to run, written to a separate file "
                         "(<prefix>_shard<k><part>.json) so a second process can share a shard")
    ap.add_argument("--part", default="", help="output-file suffix used with --only")
    args = ap.parse_args()
    subset = args.subset
    tuned = TUNED if subset == "TSB-AD-U" else TUNED_MULTI
    detectors = [d for d in (args.detectors or ",".join(tuned)).split(",") if d]

    if args.ocsvm_clean:
        _patch_ocsvm_minmax()
    if args.split == "rest":
        used = set(pd.read_csv(OFFICIAL / SPLITS[subset])["file_name"])
        used |= set(pd.read_csv(OFFICIAL / TUNING_SPLITS[subset])["file_name"])
        files = sorted(f.name for f in (DATA / subset).glob("*.csv") if f.name not in used)
    else:
        listing = (SPLITS if args.split == "eva" else TUNING_SPLITS)[subset]
        files = pd.read_csv(OFFICIAL / listing)["file_name"].tolist()
    files = [f for f in files if (DATA / subset / f).exists()]
    if args.limit:
        files = files[: args.limit]
    mine = [f for i, f in enumerate(files) if i % args.num_shards == args.shard]
    if args.only:
        only = set(args.only.split(","))
        mine = [f for f in mine if f in only]
    prefix = {"eva": "detnorm", "tuning": "detnorm_tuning", "rest": "detnorm_rest"}[args.split]
    if args.ocsvm_clean:
        prefix += "_ocsvmclean"
    out_path = OUT / subset / f"{prefix}_shard{args.shard}{args.part}.json"
    records = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {r["series"] for r in records}
    for i, f in enumerate(mine, start=1):
        if f in done:
            continue
        rec = run_series(DATA / subset / f, detectors, tuned)
        if rec is not None:
            records.append(rec)
        out_path.write_text(json.dumps(records, indent=2) + "\n")
        LOGGER.info("%d/%d %s", i, len(mine), f)


if __name__ == "__main__":
    main()
