"""Write artifacts/RELEASE_MANIFEST.json: what a release of the paper's artifacts contains.

For one commit of this repository it records
  * the commit, and whether the working tree had uncommitted changes to tracked files;
  * the pinned inputs: TSB-AD commits used, hashes of TSB-AD's published tables;
  * for each experiment: series expected, series with results, failed runs, and the files read;
  * for each table and figure of the paper: the paper_numbers.json keys and generator it comes from;
  * the SHA-256 of every file under artifacts/.

Run after collect_official_artifacts and paper_numbers; reproduce_paper.sh calls it last. It exits
non-zero if a paper element's keys are missing, if paper_numbers recorded conflicting duplicate
values, or if a family is short of its split without a note. With --verify it instead checks the
current files against an existing manifest and exits non-zero on any difference.

    python -m scripts.make_release_manifest
    python -m scripts.make_release_manifest --verify
"""

from __future__ import annotations

import argparse
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
UAD_EXPECTED = {"primary": 34, "secondary": 132,   # H11, TSB-UAD sources absent from TSB-AD
                "shared": 102}                       # TR, TSB-UAD YAHOO series TSB-AD left out
NOTES = {
    "threshold2": "349: one series (277_NEK_id_1) has no anomalous point in its test part",
    "threshold": "349: as threshold2",
    "threshold_sub": "349: one series has no anomalous test point (threshold_sub_expected.json)",
    "detnorm_uad_shared": "99: three TSB-UAD YAHOO files carry no anomaly label and are skipped, as in TSB-AD's protocol",
}

ELEMENTS = {
    "Table 1 (registered hypotheses)": (["preregistered", "held_out", "held_out_472", "boundary_synthetic",
                                         "boundary_followup", "family_toggle", "threshold_sub"],
                                        "artifacts/preregistration/results.md"),
    "Table 2 (audit)": ([], "artifacts/normalization_audit/norm_audit.csv"),
    "Table 3, Fig. 1 (toggle)": (["toggle", "failure_policy", "robustness"], "make_paper_tables.table_toggle"),
    "Table 4 (H9)": (["held_out_472"], "make_paper_tables.table_h9"),
    "Fig. 2 (splits)": (["source_decomposition"], "make_normalization_figures.fig_splits"),
    "Table 5, Sec. 6 text (baselines, held-out)": (["baselines", "baselines_held_out"], "make_paper_tables.table_bench"),
    "Sec. 6.1 text (scaling grid; full table in the artifact)": (["scaling"], "make_paper_tables.table_scaling"),
    "Table 6 (boundary test)": (["boundary_synthetic"], "make_paper_tables.table_boundary"),
    "Table 7 (factorial)": (["factorial"], "make_paper_tables.table_factorial"),
    "Table 8 (amplitude terciles)": (["tercile_sub_detectors", "regime"], "make_paper_tables.table_regime"),
    "Table 9 (thresholds)": (["threshold2"], "make_paper_tables.table_threshold"),
    "Sec. 5 text (rank intervals, H10, configurations, H9 decomposition)":
        (["rank_bootstrap", "family_toggle", "rank_sensitivity", "source_decomposition", "h9_explanation"],
         "paper_numbers.py"),
    "Sec. 8 text (T1-T2)": (["threshold_sub"], "paper_numbers.py"),
    "Table 1, Sec. 5 text (H11, TSB-UAD sources)": (["tsbuad"], "paper_numbers.py"),
    "Sec. 9 text (seeds, contamination)": (["seed_robustness", "contamination"], "paper_numbers.py"),
    "Sec. 10 checks": ([], "artifacts/normalization_audit/checks_summary.md"),
    "Sec. 4 text (constant score)": (["constant_score"], "constant_score_vus.py, paper_numbers.py"),
    "Sec. 5 text (held-out measures, chance floor)": (["other_measures_held_out", "chance_floor"], "paper_numbers.py"),
    "Sec. 5 text, abstract (split composition, source random-effects model, random re-splits)": (["composition"], "paper_numbers.py"),
    "Sec. 5 text, Table 1 (TR, SP: inside YAHOO)": (["transport", "spikein"], "paper_numbers.py"),
    "Sec. 10 text (flip radius, systematic error terms)": (["composition_sensitivity"],
                                                           "paper_numbers.py, composition_sensitivity.py"),
    "Sec. 5 text (code versions, OCSVM bug, other measures, repeat runs)":
        (["pinned", "pinned_head", "later_code_ranking", "ocsvm_clean", "other_measures", "repeat_determinism"],
         "paper_numbers.py"),
    "Sec. 6 text (stride, cost)": (["stride_both", "stride_check", "cost", "cost_median_repeat"],
                                    "paper_numbers.py, bench_official_cost.py"),
    "Sec. 7.4 text (held-out terciles)": (["held_out_472_regime"], "paper_numbers.py"),
    "Sec. 8 text (drift, H5)": (["drift", "far"], "paper_numbers.py"),
    "Sec. 10 text (choosing on the tuning split)": (["validation_selection"], "paper_numbers.py"),
}
# paper_numbers.json keys that no paper element uses: an earlier threshold protocol superseded by
# threshold2, the rendered pipeline of earlier work, and the conflict log itself.
UNUSED_KEYS = ("threshold", "render", "render_lift", "key_conflicts")
# Tracked files whose hashes the manifest records: code, tests, configuration, environment and docs.
SOURCE_PATHS = ("scripts", "src", "tests", "configs", "paper", "README.md", "LICENSE", "pyproject.toml",
                "requirements.txt", "requirements-lock.txt", "requirements-regen.txt",
                "requirements-paper-lock.txt")


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
        # Arms are dict-valued (metrics) or, in the drift/FAR families, "drift|variant" scalars.
        keys = sorted({k for r in recs.values() for k, v in r.items() if k != "errors" and
                       (isinstance(v, dict) or (v is None and k != "series") or
                        ("|" in k and isinstance(v, (int, float))))})
        ok = {k: sum(1 for r in recs.values() if r.get(k) is not None) for k in keys}
        bases = sorted({k.split("|")[0] for k in keys if k.endswith("|global") and
                        k.replace("|global", "|window") in keys})
        pairs = {b: sum(1 for r in recs.values() if r.get(f"{b}|global") and r.get(f"{b}|window"))
                 for b in bases}
        name = Path(fam).name
        parent = str(Path(fam).parent)
        expected = (UAD_EXPECTED["primary"] if "uad_primary" in name else UAD_EXPECTED["secondary"]
                    if "uad_secondary" in name else UAD_EXPECTED["shared"] if "uad_shared" in name
                    else EXPECTED["rest"] if "rest" in name else EXPECTED.get(parent))
        out[fam] = {"series": len(recs), "expected": expected, "note": NOTES.get(name),
                    "with_result": ok, "both_variants": pairs or None,
                    "files": [str(Path(f).relative_to(ART)) for f in fs]}
    for syn in sorted((OP / "synthetic").glob("*.json")):
        out[f"synthetic/{syn.stem}"] = {"records": len(json.loads(syn.read_text())),
                                        "files": [str(syn.relative_to(ART))]}
    return out


def verify() -> int:
    """Compare the current tracked sources and artifacts with RELEASE_MANIFEST.json."""
    manifest = json.loads((ART / "RELEASE_MANIFEST.json").read_text())
    bad: list[str] = []
    for group, base in (("source_hashes", Path(".")), ("files", ART)):
        for name, digest in manifest[group].items():
            path = base / name
            if not path.is_file():
                bad.append(f"missing: {path}")
            elif _sha(path) != digest:
                bad.append(f"changed: {path}")
    listed = set(manifest["files"])
    bad += [f"not in manifest: {ART / p.relative_to(ART)}" for p in sorted(ART.rglob("*"))
            if p.is_file() and p.name != "RELEASE_MANIFEST.json" and str(p.relative_to(ART)) not in listed]
    for line in bad:
        LOGGER.error(line)
    LOGGER.info("%d differences from the manifest of commit %s", len(bad), manifest["repository_commit"])
    return 1 if bad else 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="check files against the existing manifest")
    if ap.parse_args().verify:
        raise SystemExit(verify())
    numbers = json.loads((OP / "paper_numbers.json").read_text())
    manifest = {
        # The commit the manifest was written at; a manifest committed afterwards names its parent.
        "repository_commit": _git("rev-parse", "HEAD"),
        "run_at_repository_root": _git("rev-parse", "--show-toplevel") == str(Path.cwd().resolve()),
        "uncommitted_changes_to_tracked_files": bool(_git("status", "--porcelain", "--untracked-files=no")),
        # Tracked files only, so that anyone with the repository can check them.
        "source_hashes": {f: _sha(Path(f)) for f in _git("ls-files", *SOURCE_PATHS).splitlines()
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
        "paper_numbers_keys_unused": list(UNUSED_KEYS),
        "paper_numbers_keys_unmapped": sorted(set(numbers) - set(UNUSED_KEYS)
                                              - {k for v in ELEMENTS.values() for k in v[0]}),
        "files": {str(p.relative_to(ART)): _sha(p) for p in sorted(ART.rglob("*"))
                  if p.is_file() and p.name != "RELEASE_MANIFEST.json"},
    }
    (ART / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    short = [k for k, v in manifest["experiments"].items()
             if v.get("expected") and v["series"] < v["expected"] and not v.get("note")]
    LOGGER.info("wrote %s; %d files; families short of their split without a note: %s",
                ART / "RELEASE_MANIFEST.json", len(manifest["files"]), short or "none")
    missing_keys = [k for k, v in manifest["paper_elements"].items() if not v["keys_present"]]
    problems = ([f"paper elements without their keys: {missing_keys}"] if missing_keys else []) + \
        ([f"paper_numbers keys mapped to no element: {manifest['paper_numbers_keys_unmapped']}"]
         if manifest["paper_numbers_keys_unmapped"] else []) + \
        ([f"{manifest['paper_numbers_key_conflicts']} key conflicts"] if manifest["paper_numbers_key_conflicts"]
         else []) + ([f"families short of their split: {short}"] if short else [])
    if problems:
        raise SystemExit("; ".join(problems))


if __name__ == "__main__":
    main()
