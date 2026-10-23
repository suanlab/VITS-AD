"""Write artifacts/RELEASE_MANIFEST.json: what a release of the paper's artifacts contains.

For one commit of this repository it records
  * the commit, and whether the working tree had uncommitted changes to tracked files;
  * the pinned inputs: TSB-AD commits used, hashes of TSB-AD's published tables;
  * for each experiment: series expected, series with results, failed runs, and the files read;
  * for each table and figure of the paper: the paper_numbers.json keys and generator it comes from;
  * the SHA-256 of every file under artifacts/.

Run after collect_official_artifacts and paper_numbers; reproduce_paper.sh calls it last.

    python -m scripts.make_release_manifest
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
ART = Path("artifacts")
OP = ART / "official_protocol"

# (experiment, expected series, glob of per-series files, result keys that must be present)
EXPERIMENTS = (
    ("toggle_univariate_eva", 350, "TSB-AD-U/detnorm_shard*.json", "Sub_KNN|global"),
    ("toggle_multivariate_eva", 180, "TSB-AD-M/detnorm_shard*.json", "KNN|global"),
    ("toggle_multivariate_E5_MCD", 180, "TSB-AD-M/detnorm_shard*.json", "MCD|global"),
    ("toggle_multivariate_E5_IForest", 180, "TSB-AD-M/detnorm_shard*.json", "IForest|global"),
    ("toggle_multivariate_E5_PCA", 180, "TSB-AD-M/detnorm_shard*.json", "PCA|global"),
    ("ocsvm_clean_univariate", 350, "TSB-AD-U/detnorm_ocsvmclean_shard*.json", "Sub_OCSVM|global"),
    ("ocsvm_clean_multivariate", 180, "TSB-AD-M/detnorm_ocsvmclean_shard*.json", "OCSVM|global"),
    ("H6_tuning", 48, "tuning/TSB-AD-U/detnorm_tuning_shard*.json", "Sub_KNN|global"),
    ("H9_rest", 472, "TSB-AD-U/detnorm_rest_shard*.json", "Sub_KNN|global"),
    ("H9_ocsvm_clean", 472, "TSB-AD-U/detnorm_rest_ocsvmclean_shard*.json", "Sub_OCSVM|global"),
    ("seed1_eva", 350, "TSB-AD-U/detnorm_seed1_shard*.json", "Sub_IForest|global"),
    ("seed2_eva", 350, "TSB-AD-U/detnorm_seed2_shard*.json", "Sub_IForest|global"),
    ("seed1_rest", 472, "TSB-AD-U/detnorm_rest_seed1_shard*.json", "Sub_IForest|global"),
    ("seed2_rest", 472, "TSB-AD-U/detnorm_rest_seed2_shard*.json", "Sub_IForest|global"),
    ("tsbad_6beac72e_univariate", 350, "TSB-AD-U/pinned_shard*.json", "Sub_KNN"),
    ("tsbad_303dffbf_univariate", 350, "TSB-AD-U/head_shard*.json", "Sub_OCSVM"),
    ("tsbad_303dffbf_multivariate", 180, "TSB-AD-M/head_shard*.json", "KNN"),
    # one evaluation series has too short a training part to split into reference and calibration
    ("threshold2", 349, "TSB-AD-U/threshold2_shard*.json", "none|global"),
)

ELEMENTS = {
    "Table I (registered hypotheses)": (["preregistered", "held_out", "held_out_472", "boundary_synthetic"],
                                        "artifacts/preregistration/results.md"),
    "Table II (audit)": ([], "artifacts/normalization_audit/norm_audit.csv"),
    "Table III, Fig. 1 (toggle)": (["toggle", "failure_policy", "robustness"], "make_paper_tables.table_toggle"),
    "Table IV (H9)": (["held_out_472"], "make_paper_tables.table_h9"),
    "Fig. 2 (splits)": (["source_decomposition"], "make_normalization_figures.fig_splits"),
    "Table V (baselines)": (["baselines"], "make_paper_tables.table_bench"),
    "Table VI (scaling)": (["scaling"], "make_paper_tables.table_scaling"),
    "Table VII (boundary test)": (["boundary_synthetic"], "make_paper_tables.table_boundary"),
    "Table VIII, Fig. 3 (factorial)": (["factorial"], "make_paper_tables.table_factorial"),
    "Table IX (amplitude terciles)": (["tercile_sub_detectors", "regime"], "make_paper_tables.table_regime"),
    "Table X (thresholds)": (["threshold2"], "make_paper_tables.table_threshold"),
    "Sec. XI checks": ([], "artifacts/normalization_audit/checks_summary.md"),
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout.strip()


def experiment_counts() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, expected, pattern, key in EXPERIMENTS:
        files = sorted(glob.glob(str(OP / pattern)))
        ok, failed = set(), set()
        for f in files:
            for r in json.loads(Path(f).read_text()):
                if key not in r:
                    continue
                (ok if r[key] else failed).add(r["series"])
        failed -= ok
        out[name] = {"expected": expected, "with_result": len(ok), "failed": len(failed),
                     "missing": expected - len(ok) - len(failed),
                     "status": "complete" if len(ok) + len(failed) >= expected else "incomplete",
                     "files": [str(Path(f).relative_to(ART)) for f in files]}
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    numbers = json.loads((OP / "paper_numbers.json").read_text())
    manifest = {
        "repository_commit": _git("rev-parse", "HEAD"),
        "uncommitted_changes_to_tracked_files": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "tsbad_commits": {"upload_of_published_results": "35ba9c3b", "pinned": "6beac72e",
                          "current": "303dffbf"},
        "experiments": experiment_counts(),
        "paper_elements": {k: {"paper_numbers_keys": v[0], "produced_by": v[1],
                               "keys_present": all(x in numbers and numbers[x] for x in v[0])}
                           for k, v in ELEMENTS.items()},
        "paper_numbers_key_conflicts": numbers.get("key_conflicts", {}).get("n"),
        "files": {str(p.relative_to(ART)): _sha(p) for p in sorted(ART.rglob("*"))
                  if p.is_file() and p.name != "RELEASE_MANIFEST.json"},
    }
    (ART / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    inc = [k for k, v in manifest["experiments"].items() if v["status"] != "complete"]
    LOGGER.info("wrote %s; %d files; incomplete experiments: %s", ART / "RELEASE_MANIFEST.json",
                len(manifest["files"]), inc or "none")


if __name__ == "__main__":
    main()
