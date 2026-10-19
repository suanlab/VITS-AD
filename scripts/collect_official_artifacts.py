"""Copy per-series results of the official-protocol runs into artifacts/ (the only GPU-free input).

The run scripts write to results/official_protocol/<subset>/ (not tracked). Every analysis
script reads only artifacts/official_protocol/, so the paper's numbers can be regenerated from a
clean clone. This script is the single place where results are copied over.

    python -m scripts.collect_official_artifacts
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

LOGGER = logging.getLogger(__name__)
RES = Path("results/official_protocol")
ART = Path("artifacts/official_protocol")
PATTERNS = ("per_series_*_shard*.json", "ablation_shard*.json", "detnorm_shard*.json",
            "detnorm_tuning_shard*.json", "pinned_shard*.json", "drift_shard*.json",
            "far_shard*.json")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    n = 0
    for subset in ("TSB-AD-U", "TSB-AD-M"):
        for pattern in PATTERNS:
            for f in sorted((RES / subset).glob(pattern)):
                # Tuning-split runs (tag tune_m<margin>) are kept apart from evaluation-split ones.
                tuning = "_tune_" in f.name or f.name.startswith("detnorm_tuning")
                dest = ART / "tuning" / subset if tuning else ART / subset
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, dest / f.name)
                n += 1
    LOGGER.info("copied %d files into %s", n, ART)


if __name__ == "__main__":
    main()
