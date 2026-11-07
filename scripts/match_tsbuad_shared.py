"""Which TSB-UAD series of the sources TSB-AD shares did TSB-AD leave out? (registered TR,
artifacts/preregistration/transport_hypothesis.md)

TSB-AD curated its univariate series from TSB-UAD and other archives. For the sources both hold
(YAHOO, and KDD21, which TSB-AD names UCR), this matches every TSB-AD-U series to a TSB-UAD file by
raw values only: the same length and a Pearson correlation of the values above 0.999, or, where
TSB-AD kept a contiguous piece, an exact match of that piece. No labels, scores or detectors are
read (TSB-UAD's .out files hold a label column; only the value column is loaded).

Output artifacts/official_protocol/tsbuad_shared_match.json:
  <source>: {"tsbad_to_tsbuad": {tsbad file: tsbuad file or null},
             "unmatched_tsbuad": [tsbuad files no TSB-AD-U series matches, one per distinct series],
             "unmatched_duplicates": {tsbuad file: the unmatched file with the same values},
             "ambiguous": {tsbad file: every tsbuad file it matches}}
  Every file an ambiguous series matches counts as used.

    python -m scripts.match_tsbuad_shared
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
UAD = Path("data/tsb_uad/public/TSB-UAD-Public")
TSBAD = Path("data/tsb_ad/TSB-AD-U")
OUT = Path("artifacts/official_protocol/tsbuad_shared_match.json")
SOURCES = {"YAHOO": "YAHOO", "KDD21": "UCR"}  # TSB-UAD folder -> TSB-AD source name
R_MIN = 0.999


def _values_uad(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None, usecols=[0]).iloc[:, 0].to_numpy(dtype=float)


def _values_tsbad(path: Path) -> np.ndarray:
    return pd.read_csv(path, usecols=["Data"])["Data"].to_numpy(dtype=float)


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    if a.size != b.size or a.size < 3:
        return False
    if np.allclose(a, b):
        return True
    sa, sb = a.std(), b.std()
    return bool(sa > 0 and sb > 0 and abs(np.corrcoef(a, b)[0, 1]) > R_MIN)


def _piece_of(short: np.ndarray, long: np.ndarray) -> bool:
    """True if ``short`` equals a contiguous piece of ``long`` (first 32 values located exactly)."""
    if short.size >= long.size:
        return False
    head = short[:32]
    for start in np.flatnonzero(np.isclose(long[: long.size - short.size + 1], head[0])):
        if np.allclose(long[start:start + 32], head) and np.allclose(long[start:start + short.size], short):
            return True
    return False


def match(uad_folder: str, tsbad_source: str) -> dict[str, object]:
    uad_files = sorted((UAD / uad_folder).glob("*.out"))
    uad = {f.name: _values_uad(f) for f in uad_files}
    by_len: dict[int, list[str]] = {}
    for k, v in uad.items():
        by_len.setdefault(v.size, []).append(k)
    out: dict[str, str | None] = {}
    ambiguous: dict[str, list[str]] = {}
    used: set[str] = set()
    for f in sorted(TSBAD.glob(f"*_{tsbad_source}_id_*.csv")):
        x = _values_tsbad(f)
        hits = [k for k in by_len.get(x.size, []) if _same(x, uad[k])]
        if not hits:
            hits = [k for k, v in uad.items() if _piece_of(x, v)]
        if len(hits) > 1:
            ambiguous[f.name] = hits
        used.update(hits)  # a file whose values equal a matched one is not new either
        out[f.name] = hits[0] if hits else None
    # Unmatched files with the same values as an earlier unmatched file are one new series, not two.
    new, dup_of = [], {}
    for k in sorted(set(uad) - used):
        twin = next((j for j in new if _same(uad[k], uad[j])), None)
        if twin is None:
            new.append(k)
        else:
            dup_of[k] = twin
    return {"tsbad_to_tsbuad": out, "unmatched_tsbuad": new, "unmatched_duplicates": dup_of,
            "ambiguous": ambiguous, "n_tsbad": len(out), "n_tsbad_matched": sum(v is not None for v in out.values()),
            "n_tsbuad": len(uad), "n_unmatched_tsbuad": len(new)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = {}
    for folder, src in SOURCES.items():
        res[folder] = match(folder, src)
        r = res[folder]
        LOGGER.info("%s: %d of %d TSB-AD-U %s series matched; %d of %d TSB-UAD files unmatched; %d ambiguous",
                    folder, r["n_tsbad_matched"], r["n_tsbad"], src, r["n_unmatched_tsbuad"], r["n_tsbuad"],
                    len(r["ambiguous"]))
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    LOGGER.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
