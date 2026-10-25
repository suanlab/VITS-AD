"""Per-window vs global normalization for two more per-window detector families in TSB-AD:
Matrix Profile (stumpy) and KMeansAD. Registered as H10 (artifacts/preregistration/family_hypothesis.md).

Both run through TSB-AD 1.5's own wrappers (run_MatrixProfile, run_KMeansAD_U) with the tuned
hyperparameters of Optimal_Uni_algo_HP_dict; only the normalization differs:

  window  as published: MatrixProfile computes z-normalized distances (stumpy.stump, normalize=True);
          KMeansAD z-normalizes each window (normalize=True); raw input, as TSB-AD passes it
  global  the series is z-scored once with training-part statistics; MatrixProfile uses the
          non-normalized Euclidean distance (stumpy.stump(normalize=False)); KMeansAD clusters the
          windows as they are (normalize=False)

KMeansAD's KMeans has no fixed seed in TSB-AD; both variants here use random_state=0, so the
per-window variant matches the published values only approximately.

Output: results/official_protocol/TSB-AD-U/family[_rest]_shard<k>[<part>].json (a second process
can take part of a shard with --only/--part; readers merge by series)

    python -m scripts.run_family_toggle --split eva --shard k --num-shards K
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from scripts.run_detector_normalization import DATA, OFFICIAL, OUT, SPLITS, TUNING_SPLITS
from scripts.run_official_protocol import official_metrics

LOGGER = logging.getLogger(__name__)
DETECTORS = ("MatrixProfile", "KMeansAD_U")
PUBLISHED_NAME = {"MatrixProfile": "MatrixProfile", "KMeansAD_U": "KMeansAD"}


def _setup(variant: str) -> None:
    import stumpy
    from sklearn.cluster import KMeans

    mp = importlib.import_module("TSB_AD.models.MatrixProfile")
    normalize = variant == "window"
    mp.stumpy = SimpleNamespace(stump=lambda T, m: stumpy.stump(T, m=m, normalize=normalize))
    km = importlib.import_module("TSB_AD.models.KMeansAD")
    km.KMeans = lambda n_clusters: KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    original = getattr(km.KMeansAD, "_orig_init", km.KMeansAD.__init__)
    km.KMeansAD._orig_init = original

    def init(self: Any, *a: Any, **kw: Any) -> None:
        kw["normalize"] = normalize
        original(self, *a, **kw)

    km.KMeansAD.__init__ = init


def run_series(path: Path) -> dict[str, Any] | None:
    from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict
    from TSB_AD.utils.slidingWindows import find_length_rank

    wrapper = importlib.import_module("TSB_AD.model_wrapper")
    df = pd.read_csv(path).dropna()
    x = df.iloc[:, 0:-1].to_numpy(dtype=float)
    labels = df["Label"].astype(int).to_numpy()
    n_train = int(path.stem.split("_")[-3])
    if labels.max() == 0:
        return None
    sliding = int(find_length_rank(x[:, 0].reshape(-1, 1), rank=1))
    mu, sd = x[:n_train].mean(axis=0), x[:n_train].std(axis=0) + 1e-8
    rec: dict[str, Any] = {"series": path.name, "n": int(x.shape[0]), "n_train": n_train}
    for variant in ("window", "global"):
        _setup(variant)
        data = (x - mu) / sd if variant == "global" else x
        for det in DETECTORS:
            key = f"{det}|{variant}"
            try:
                score = np.asarray(getattr(wrapper, f"run_{det}")(data, **Optimal_Uni_algo_HP_dict[det]),
                                   dtype=float).ravel()
                if score.size != labels.size or not np.all(np.isfinite(score)):
                    raise ValueError(f"bad score vector (size {score.size}, n {labels.size})")
                rec[key] = official_metrics(score, labels, sliding)
            except Exception as exc:
                LOGGER.warning("%s [%s] failed: %s: %s", path.name, key, type(exc).__name__, str(exc)[:160])
                rec[key] = None
    return rec


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("eva", "rest"), default="eva")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="comma-separated series of this shard to run")
    ap.add_argument("--part", default="", help="output-file suffix used with --only")
    args = ap.parse_args()
    subset = "TSB-AD-U"
    if args.split == "rest":
        used = set(pd.read_csv(OFFICIAL / SPLITS[subset])["file_name"])
        used |= set(pd.read_csv(OFFICIAL / TUNING_SPLITS[subset])["file_name"])
        files = sorted(f.name for f in (DATA / subset).glob("*.csv") if f.name not in used)
    else:
        files = pd.read_csv(OFFICIAL / SPLITS[subset])["file_name"].tolist()
    files = [f for f in files if (DATA / subset / f).exists()][: args.limit]
    mine = [f for i, f in enumerate(files) if i % args.num_shards == args.shard]
    if args.only:
        mine = [f for f in mine if f in set(args.only.split(","))]
    prefix = "family" if args.split == "eva" else "family_rest"
    out_path = OUT / subset / f"{prefix}_shard{args.shard}{args.part}.json"
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
