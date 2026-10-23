"""VUS-PR of a constant score on every TSB-AD-U-Eva series, with TSB-AD's own get_metrics.

At the commit of the uploaded results, TSB-AD's univariate IForest and LOF z-score each time step
across its single channel, which maps every value to 0 and yields a constant score. This script
computes what such a score attains under the official metric and sliding-window parameter, to
compare with the published values of those two detectors.

Output: artifacts/official_protocol/constant_score.json

    python -m scripts.constant_score_vus
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_official_protocol import DATA, OFFICIAL, SPLITS, official_metrics

LOGGER = logging.getLogger(__name__)
OUT = Path("artifacts/official_protocol/constant_score.json")


def main() -> None:
    from TSB_AD.utils.slidingWindows import find_length_rank

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    files = pd.read_csv(OFFICIAL / SPLITS["TSB-AD-U"])["file_name"].tolist()
    out: dict[str, dict[str, float]] = {}
    for f in files:
        df = pd.read_csv(DATA / "TSB-AD-U" / f).dropna()
        x = df.iloc[:, 0].to_numpy(dtype=float)
        labels = df["Label"].astype(int).to_numpy()
        sliding = int(find_length_rank(x.reshape(-1, 1), rank=1))
        m = official_metrics(np.zeros_like(x), labels, sliding)
        out[f] = {"VUS-PR": m["VUS-PR"], "AUC-PR": m["AUC-PR"], "anomaly_ratio": float(labels.mean())}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    LOGGER.info("wrote %s (%d series, mean VUS-PR %.3f)", OUT, len(out),
                np.mean([v["VUS-PR"] for v in out.values()]))


if __name__ == "__main__":
    main()
