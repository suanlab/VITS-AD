"""Run TSB-AD's subsequence detectors from the pinned TSB-AD commit, unmodified (TSB-AD-U-Eva).

TSB-AD's published per-series tables were produced by an older revision of its code. Commit
a79f3154 (2025-10-01, "update norm") later changed how some detectors normalize univariate
windows, but the tables were not regenerated. This script measures what TSB-AD's *current*
code (the commit we pin, 6beac72e) scores, so the paper can say how stale the published
leaderboard is relative to the benchmark's own code.

The detectors run exactly as TSB-AD's wrappers call them, with the tuned hyperparameters of the
same commit's HP_list (Optimal_Uni_algo_HP_dict). The only change is a compatibility shim: on
scikit-learn >= 1.6, ``check_is_fitted`` requires ``__sklearn_tags__``, which TSB-AD's
BaseDetector lacks; it is replaced, per module, by the attribute check it is meant to perform.

Run in a separate process with the pinned source first on the path:
    PYTHONPATH=<tsbad_src> python -m scripts.run_tsbad_pinned_detectors --tsbad-src <tsbad_src>

Output: results/official_protocol/TSB-AD-U/pinned_shard<k>.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
DETECTORS = ("Sub_KNN", "Sub_LOF", "Sub_IForest", "Sub_HBOS", "Sub_OCSVM", "Sub_MCD", "Sub_PCA")
SEMI_SUPERVISED = {"Sub_OCSVM", "Sub_MCD"}
MODULES = ("KNN", "LOF", "IForest", "OCSVM", "HBOS", "MCD", "PCA")


def _check_is_fitted(estimator: Any, attributes: list[str] | None = None, **_: Any) -> None:
    missing = [a for a in (attributes or []) if not hasattr(estimator, a)]
    if missing:
        raise ValueError(f"{type(estimator).__name__} is not fitted: missing {missing}")


def _load(tsbad_src: Path) -> tuple[Any, dict[str, dict[str, Any]]]:
    import TSB_AD

    origin = Path(TSB_AD.__file__).resolve()
    if tsbad_src.resolve() not in origin.parents:
        raise RuntimeError(f"TSB_AD imported from {origin}, not from {tsbad_src}; set PYTHONPATH")
    for name in MODULES:
        mod = importlib.import_module(f"TSB_AD.models.{name}")
        if hasattr(mod, "check_is_fitted"):
            mod.check_is_fitted = _check_is_fitted
    wrapper = importlib.import_module("TSB_AD.model_wrapper")
    hp = importlib.import_module("TSB_AD.HP_list").Optimal_Uni_algo_HP_dict
    return wrapper, {d: dict(hp[d]) for d in DETECTORS}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsbad-src", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    wrapper, tuned = _load(args.tsbad_src)
    from TSB_AD.evaluation.metrics import get_metrics
    from TSB_AD.utils.slidingWindows import find_length_rank

    data_dir, out_dir = Path("data/tsb_ad/TSB-AD-U"), Path("results/official_protocol/TSB-AD-U")
    files = pd.read_csv("data/tsb_ad/official/TSB-AD-U-Eva.csv")["file_name"].tolist()
    files = [f for f in files if (data_dir / f).exists()][: args.limit]
    mine = [f for i, f in enumerate(files) if i % args.num_shards == args.shard]
    out_path = out_dir / f"pinned_shard{args.shard}.json"
    records = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {r["series"] for r in records}
    LOGGER.info("tuned: %s", tuned)
    for i, f in enumerate(mine, start=1):
        if f in done:
            continue
        df = pd.read_csv(data_dir / f).dropna()
        data = df.iloc[:, 0:-1].to_numpy(dtype=float)
        labels = df["Label"].astype(int).to_numpy()
        n_train = int(Path(f).stem.split("_")[-3])
        sliding = int(find_length_rank(data[:, 0].reshape(-1, 1), rank=1))
        rec: dict[str, Any] = {"series": f}
        for det in DETECTORS:
            try:
                run = getattr(wrapper, f"run_{det}")
                score = (run(data[:n_train], data, **tuned[det]) if det in SEMI_SUPERVISED
                         else run(data, **tuned[det]))
                score = np.asarray(score, dtype=float).ravel()
                if score.size != labels.size or not np.all(np.isfinite(score)):
                    raise ValueError(f"bad score vector (size {score.size})")
                rec[det] = {k: float(v) for k, v in
                            get_metrics(score, labels, slidingWindow=sliding).items()}
            except Exception as exc:
                LOGGER.warning("%s [%s] failed: %s: %s", f, det, type(exc).__name__,
                               str(exc)[:160])
                rec[det] = None
        records.append(rec)
        out_path.write_text(json.dumps(records, indent=2) + "\n")
        LOGGER.info("%d/%d %s", i, len(mine), f)


if __name__ == "__main__":
    main()
