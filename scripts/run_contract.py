"""Shared run contract for the per-series experiment scripts.

Three checks that make an incomplete or stale run fail instead of being aggregated as a new one:

* Inputs: every series listed by the split must exist; a missing file stops the run unless
  --allow-missing is given (then the missing names are logged and stored in the fingerprint).
* Fingerprint: next to each output file ``X.json`` a file ``X.json.fingerprint`` records the run's
  configuration (detectors, hyperparameters, options), the hash of the split listing and of the
  selected series names, and the hash of the script's source. Resuming with a different
  configuration or series list raises; a different source hash or environment (library versions,
  thread settings) is logged and appended to the fingerprint's history, since a comment change or a
  patch release should not force a re-run, but a reader can see it. An
  output written before fingerprints existed gets one on its first resume, marked as such.
* Atomic writes: records are written to a temporary file and renamed.

Failed arms are kept in each record as ``None``, with the exception text under ``errors``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def require_inputs(files: list[str], data_dir: Path, allow_missing: bool = False) -> tuple[list[str], list[str]]:
    """Return (present, missing); raise FileNotFoundError if any is missing and not allowed."""
    missing = [f for f in files if not (Path(data_dir) / f).is_file()]
    if missing and not allow_missing:
        raise FileNotFoundError(f"{len(missing)} of {len(files)} required series missing in {data_dir}; "
                                f"first: {missing[:3]} (pass --allow-missing for a partial run)")
    if missing:
        LOGGER.warning("partial run: %d series missing, e.g. %s", len(missing), missing[:3])
    return [f for f in files if f not in set(missing)], missing


def environment() -> dict[str, Any]:
    """Python and library versions and the thread settings, recorded with each run."""
    import platform
    from importlib import metadata

    versions = {}
    for pkg in ("numpy", "scipy", "pandas", "scikit-learn", "TSB-AD", "stumpy", "torch"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    threads = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                              "NUMBA_NUM_THREADS")}
    return {"python": platform.python_version(), "packages": versions, "threads": threads}


def fingerprint_path(out_path: Path) -> Path:
    return out_path.with_name(out_path.name + ".fingerprint")


def open_run(out_path: Path, config: dict[str, Any], series: list[str], script: str | Path,
             split_file: Path | None = None, missing: list[str] | None = None) -> list[dict[str, Any]]:
    """Check or create the fingerprint of ``out_path`` and return the records already written.

    Raises RuntimeError when an existing fingerprint's configuration or series list differs.
    """
    fp = {"config": json.loads(json.dumps(config, sort_keys=True, default=str)),
          "series_sha256": sha256_text("\n".join(series)), "n_series": len(series),
          "split_sha256": sha256_file(split_file) if split_file else None,
          "missing_inputs": sorted(missing or [])}
    code = sha256_file(Path(script))
    path = fingerprint_path(out_path)
    records = json.loads(out_path.read_text()) if out_path.exists() else []
    if path.exists():
        old = json.loads(path.read_text())
        for k in ("config", "series_sha256", "split_sha256"):
            if old.get(k) != fp[k]:
                raise RuntimeError(f"{out_path}: existing run has a different {k}; use a new output "
                                   f"(e.g. --part) or remove it. old={old.get(k)!r} new={fp[k]!r}")
        env = environment()
        changed = False
        if code not in old.get("code_sha256_seen", []):
            LOGGER.warning("%s: script source changed since this output was started", out_path.name)
            old.setdefault("code_sha256_seen", []).append(code)
            changed = True
        if env not in old.get("environments_seen", []):
            LOGGER.warning("%s: environment differs from the one this output was started in", out_path.name)
            old.setdefault("environments_seen", []).append(env)
            changed = True
        if changed:
            _write(path, old)
        return records
    fp["code_sha256_seen"] = [code]
    fp["environments_seen"] = [environment()]
    if records:
        fp["note"] = f"created on resume; {len(records)} earlier records predate the fingerprint"
        LOGGER.warning("%s: no fingerprint; creating one for %d existing records", out_path.name, len(records))
    _write(path, fp)
    return records


def save_records(out_path: Path, records: list[dict[str, Any]]) -> None:
    _write(out_path, records, indent=2)


def _write(path: Path, obj: Any, indent: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=indent) + "\n")
    os.replace(tmp, path)
