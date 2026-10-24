"""Two per-series features used to explain where global normalization helps (exploratory).

For every TSB-AD-U series in the evaluation split and in neither split:
  len_ratio  median length of the labelled anomalous segments divided by the series' period-based
             window, TSB-AD's find_length_rank(rank=1); uses labels
  out_range  fraction of normal test points (after the training part) outside the [min, max] of
             the training part; a label-free measure of drift between training and test parts,
             except that it excludes anomalous points

Both are computed once from the raw series and cached, so the regeneration does not need the data.

Output: artifacts/official_protocol/series_features.json

    python -m scripts.series_features
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_detector_normalization import DATA, OFFICIAL, SPLITS, TUNING_SPLITS

LOGGER = logging.getLogger(__name__)
OUT = Path("artifacts/official_protocol/series_features.json")


def features(path: Path) -> dict[str, float] | None:
    from TSB_AD.utils.slidingWindows import find_length_rank

    df = pd.read_csv(path).dropna()
    x = df.iloc[:, 0].to_numpy(dtype=float)
    lab = df["Label"].astype(int).to_numpy()
    n_train = int(path.stem.split("_")[-3])
    if lab.max() == 0:
        return None
    idx = np.flatnonzero(np.diff(np.concatenate([[0], lab, [0]])))
    lengths = idx[1::2] - idx[::2]
    window = int(find_length_rank(x.reshape(-1, 1), rank=1))
    test_normal = (np.arange(x.size) >= n_train) & (lab == 0)
    lo, hi = x[:n_train].min(), x[:n_train].max()
    out = float(np.mean((x[test_normal] < lo) | (x[test_normal] > hi))) if test_normal.any() else float("nan")
    return {"len_ratio": float(np.median(lengths) / max(window, 1)), "window": window,
            "median_anomaly_length": float(np.median(lengths)), "out_range": out}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    subset = "TSB-AD-U"
    eva = list(pd.read_csv(OFFICIAL / SPLITS[subset])["file_name"])
    used = set(eva) | set(pd.read_csv(OFFICIAL / TUNING_SPLITS[subset])["file_name"])
    rest = sorted(f.name for f in (DATA / subset).glob("*.csv") if f.name not in used)
    res: dict[str, dict[str, dict[str, float]]] = {"eva": {}, "rest": {}}
    for name, files in (("eva", eva), ("rest", rest)):
        for f in files:
            if (DATA / subset / f).exists() and (v := features(DATA / subset / f)) is not None:
                res[name][f] = v
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    LOGGER.info("wrote %s (%d eva, %d rest)", OUT, len(res["eva"]), len(res["rest"]))


if __name__ == "__main__":
    main()
