"""Copy per-series results of the official-protocol runs into artifacts/ (the only GPU-free input).

The run scripts write to results/official_protocol/<subset>/ (not tracked). Every analysis
script reads only artifacts/official_protocol/, so the paper's numbers can be regenerated from a
clean clone. This script is the single place where results are copied over.

    python -m scripts.collect_official_artifacts             # add or overwrite files
    python -m scripts.collect_official_artifacts --replace   # experiment modes: drop the released
                                                             # files of every family being copied

With --replace (used by reproduce_paper.sh in its experiment modes core, legacy and full), the released per-series files of each result family
present in results/ are removed before copying, so that a series missing from a re-run cannot be
filled silently by its released value. Released families with no file in results/ are kept, and
listed as such in artifacts/official_protocol/collect_report.json; --strict makes them an error.
Run fingerprints (<file>.fingerprint, see run_contract.py) are copied with their files.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

LOGGER = logging.getLogger(__name__)
RES = Path("results/official_protocol")
ART = Path("artifacts/official_protocol")
PATTERNS = ("per_series_*_shard*.json", "ablation_shard*.json", "detnorm_shard*.json",
            "detnorm_tuning_shard*.json", "detnorm_rest_shard*.json",
            "detnorm_rest_ocsvmclean_shard*.json", "detnorm_ocsvmclean_shard*.json",
            "pinned_shard*.json", "head_shard*.json", "drift_shard*.json", "far_shard*.json",
            "threshold_shard*.json", "threshold2_shard*.json", "detnorm_seed*_shard*.json",
            "detnorm_rest_seed*_shard*.json", "family_shard*.json", "family_rest_shard*.json",
            "family_seed*_shard*.json", "family_rest_seed*_shard*.json",
            "detnorm_uad_primary_shard*.json", "detnorm_uad_secondary_shard*.json",
            "threshold_sub_shard*.json")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="with --replace: fail if a released family has no file in results/")
    args = ap.parse_args()
    n = 0
    removed: set[tuple[Path, str]] = set()
    released = {(old.parent, old.name.split("_shard")[0]) for subset in ("TSB-AD-U", "TSB-AD-M")
                for d in (ART / subset, ART / "tuning" / subset) for pattern in PATTERNS
                for old in d.glob(pattern)}
    for subset in ("TSB-AD-U", "TSB-AD-M"):
        for pattern in PATTERNS:
            for f in sorted((RES / subset).glob(pattern)):
                # Tuning-split runs (tag tune_m<margin>) are kept apart from evaluation-split ones.
                tuning = "_tune_" in f.name or f.name.startswith("detnorm_tuning")
                dest = ART / "tuning" / subset if tuning else ART / subset
                family = f.name.split("_shard")[0]
                if args.replace and (dest, family) not in removed:
                    for old in dest.glob(f"{family}_shard*.json"):
                        if old.name.split("_shard")[0] == family:
                            old.unlink()
                            old.with_name(old.name + ".fingerprint").unlink(missing_ok=True)
                    removed.add((dest, family))
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, dest / f.name)
                fp = f.with_name(f.name + ".fingerprint")
                if fp.exists():
                    shutil.copy(fp, dest / fp.name)
                n += 1
        expected = RES / subset / "threshold_sub_expected.json"
        if expected.exists():
            shutil.copy(expected, ART / subset / expected.name)
    uad = Path("data/tsb_uad/converted/listing.csv")      # H11's series list (convert_tsbuad.py)
    if uad.exists():
        shutil.copy(uad, ART / "tsbuad_listing.csv")
    for syn in (RES / "synthetic" / "boundary.json", RES / "synthetic" / "boundary_followup.json"):
        if syn.exists():
            (ART / "synthetic").mkdir(parents=True, exist_ok=True)
            shutil.copy(syn, ART / "synthetic" / syn.name)
            n += 1
    LOGGER.info("copied %d files into %s", n, ART)
    if args.replace:
        kept = sorted(f"{d.relative_to(ART)}/{fam}" for d, fam in released - removed)
        (ART / "collect_report.json").write_text(json.dumps(
            {"re_run": sorted(f"{d.relative_to(ART)}/{fam}" for d, fam in removed),
             "kept_from_release": kept}, indent=1) + "\n")
        if kept:
            LOGGER.warning("%d released families were not re-run and are kept: %s", len(kept), kept)
            if args.strict:
                raise SystemExit("--strict: released families without a re-run")


if __name__ == "__main__":
    main()
