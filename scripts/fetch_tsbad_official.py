"""Download TSB-AD's published per-series results and evaluation file lists, pinned to a commit.

The comparisons with TSB-AD's detectors use the per-series VUS-PR that TSB-AD itself publishes
(benchmark_exp/benchmark_eval_results for the 40 detectors of the TSB-AD paper,
benchmark_exp/leaderboard_results for later leaderboard submissions). They were produced with the
same TSB_AD.evaluation.metrics.get_metrics we use. TSB-AD is Apache-2.0 licensed; we fetch the
files rather than vendor them, pin the commit, and verify each file's SHA-256.

Output: data/tsb_ad/official/
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.request
from pathlib import Path

LOGGER = logging.getLogger(__name__)
COMMIT = "6beac72e11d1155ade40870492c00d0d1cfdcaaf"
BASE = f"https://raw.githubusercontent.com/TheDatumOrg/TSB-AD/{COMMIT}/"
OUT = Path("data/tsb_ad/official")
SOURCES = {
    "uni_mergedTable_VUS-PR.csv": "benchmark_exp/benchmark_eval_results/uni_mergedTable_VUS-PR.csv",
    "multi_mergedTable_VUS-PR.csv": "benchmark_exp/benchmark_eval_results/multi_mergedTable_VUS-PR.csv",
    "TSB-AD-U-Eva.csv": "Datasets/File_List/TSB-AD-U-Eva.csv",
    "TSB-AD-M-Eva.csv": "Datasets/File_List/TSB-AD-M-Eva.csv",
    "TSB-AD-U-Tuning.csv": "Datasets/File_List/TSB-AD-U-Tuning.csv",
    "TSB-AD-M-Tuning.csv": "Datasets/File_List/TSB-AD-M-Tuning.csv",
    **{f"leaderboard/{n}.csv": f"benchmark_exp/leaderboard_results/{n}.csv" for n in (
        "Multi_AxonAD", "Multi_CHARM", "Multi_MMPAD", "Multi_StreamVAE", "Multi_TSPulse",
        "Multi_Time_RCD", "Multi_xLSTMAD", "Uni_AxonAD", "Uni_CHARM", "Uni_MMPAD",
        "Uni_StreamVAE", "Uni_TSPulse", "Uni_TimeRCD_MAFT", "Uni_Time_RCD", "Uni_xLSTMAD")},
}
HASHES = Path(__file__).with_name("tsbad_official_sha256.json")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    expected = json.loads(HASHES.read_text())
    for local, remote in SOURCES.items():
        target = OUT / local
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            urllib.request.urlretrieve(BASE + remote, target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != expected[local]:
            raise SystemExit(f"{local}: SHA-256 mismatch ({digest})")
        LOGGER.info("ok  %s", local)


if __name__ == "__main__":
    main()
