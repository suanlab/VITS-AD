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

# Expected series per family of per-series result files. Families not listed take the size of the
# split their directory holds; NOTES explain families that legitimately have fewer series.
EXPECTED = {"TSB-AD-U": 350, "TSB-AD-M": 180, "tuning/TSB-AD-U": 48, "tuning/TSB-AD-M": 20, "rest": 472}
NOTES = {
    "threshold2": "349: one series has too short a training part to split into reference and calibration",
    "threshold": "349: as threshold2",
}

ELEMENTS = {
    "Table I (registered hypotheses)": (["preregistered", "held_out", "held_out_472", "boundary_synthetic",
                                         "boundary_followup"],
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
    """Per family of per-series files: series present, and per detector or arm the series with a
    result and, for global/per-window pairs, the series where both variants succeeded."""
    out: dict[str, Any] = {}
    files = sorted(glob.glob(str(OP / "**" / "*_shard*.json"), recursive=True))
    families: dict[str, list[str]] = {}
    for f in files:
        rel = Path(f).relative_to(OP)
        fam = str(rel.parent / rel.name.split("_shard")[0])
        families.setdefault(fam, []).append(f)
    for fam, fs in sorted(families.items()):
        recs: dict[str, dict[str, Any]] = {}
        for f in fs:
            for r in json.loads(Path(f).read_text()):
                recs.setdefault(r["series"], {}).update(r)
        keys = sorted({k for r in recs.values() for k, v in r.items() if isinstance(v, dict) or
                       (v is None and k not in ("series",))})
        ok = {k: sum(1 for r in recs.values() if r.get(k)) for k in keys}
        bases = sorted({k.split("|")[0] for k in keys if k.endswith("|global") and
                        k.replace("|global", "|window") in keys})
        pairs = {b: sum(1 for r in recs.values() if r.get(f"{b}|global") and r.get(f"{b}|window"))
                 for b in bases}
        name = Path(fam).name
        parent = str(Path(fam).parent)
        expected = EXPECTED["rest"] if "rest" in name else EXPECTED.get(parent)
        out[fam] = {"series": len(recs), "expected": expected, "note": NOTES.get(name),
                    "with_result": ok, "both_variants": pairs or None,
                    "files": [str(Path(f).relative_to(ART)) for f in fs]}
    for syn in sorted((OP / "synthetic").glob("*.json")):
        out[f"synthetic/{syn.stem}"] = {"records": len(json.loads(syn.read_text())),
                                        "files": [str(syn.relative_to(ART))]}
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    numbers = json.loads((OP / "paper_numbers.json").read_text())
    manifest = {
        # The commit the manifest was written at; a manifest committed afterwards names its parent.
        "repository_commit": _git("rev-parse", "HEAD"),
        "run_at_repository_root": _git("rev-parse", "--show-toplevel") == str(Path.cwd().resolve()),
        "uncommitted_changes_to_tracked_files": bool(_git("status", "--porcelain", "--untracked-files=no")),
        # Tracked sources only, so that anyone with the repository can check them.
        "source_hashes": {f: _sha(Path(f)) for f in _git("ls-files", "scripts", "paper").splitlines()
                          if Path(f).is_file()},
        "tsbad_commits": {"upload_of_published_results": "35ba9c3b", "pinned": "6beac72e",
                          "current": "303dffbf"},
        "seeds": {"IForest random_state": "0 (TSB-AD default); 1 and 2 in detnorm*_seed<k> files",
                  "MCD random_state": "2024 (TSB-AD default); 1 and 2 in detnorm*_seed<k> files",
                  "bootstrap": "numpy default_rng(0), 10,000 resamples",
                  "synthetic boundary tests": "series seeds 0-19 (B1-B4), 0-19 and 0-9 (F1, F2)"},
        "experiments": experiment_counts(),
        "paper_elements": {k: {"paper_numbers_keys": v[0], "produced_by": v[1],
                               "keys_present": all(x in numbers and numbers[x] for x in v[0])}
                           for k, v in ELEMENTS.items()},
        "paper_numbers_key_conflicts": numbers.get("key_conflicts", {}).get("n"),
        "files": {str(p.relative_to(ART)): _sha(p) for p in sorted(ART.rglob("*"))
                  if p.is_file() and p.name != "RELEASE_MANIFEST.json"},
    }
    (ART / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    short = [k for k, v in manifest["experiments"].items()
             if v.get("expected") and v["series"] < v["expected"] and not v.get("note")]
    LOGGER.info("wrote %s; %d files; families short of their split without a note: %s",
                ART / "RELEASE_MANIFEST.json", len(manifest["files"]), short or "none")


if __name__ == "__main__":
    main()
