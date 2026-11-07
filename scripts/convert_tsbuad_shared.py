"""Convert the YAHOO series of TSB-UAD that TSB-AD left out into TSB-AD's file format (registered TR,
artifacts/preregistration/transport_hypothesis.md).

`scripts/match_tsbuad_shared.py` matched all 259 TSB-AD-U YAHOO series to TSB-UAD files by raw values
and lists the TSB-UAD YAHOO files no TSB-AD series matches, one per distinct series. These are
written with TSB-AD's YAHOO training part (the first 500 points, as in every TSB-AD-U YAHOO file)
and the Yahoo S5 subset of each file (A1 real; A2 synthetic; A3 and A4, which share series, merged).

Input: data/tsb_uad/public/TSB-UAD-Public/YAHOO/*.out (value,label) from TSB-UAD-Public.zip
(SHA-256 checked by scripts/convert_tsbuad.py) and artifacts/official_protocol/tsbuad_shared_match.json.
Output: data/tsb_uad/converted_shared/yahoo/<idx>_YAHOO_id_<k>_<stratum>_tr_500_1st_<first>.csv
(Data, Label) and data/tsb_uad/converted_shared/listing.csv.

    python -m scripts.convert_tsbuad_shared
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
SRC = Path("data/tsb_uad/public/TSB-UAD-Public/YAHOO")
MATCH = Path("artifacts/official_protocol/tsbuad_shared_match.json")
OUT = Path("data/tsb_uad/converted_shared")
N_TRAIN = 500


def stratum(name: str) -> str:
    """Yahoo S5 subset from a TSB-UAD file name; A3 and A4 share series and are merged."""
    if "A1real" in name:
        return "A1"
    if "A2synthetic" in name:
        return "A2"
    return "A3A4"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    files = json.loads(MATCH.read_text())["YAHOO"]["unmatched_tsbuad"]
    (OUT / "yahoo").mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, name in enumerate(sorted(files), start=1):
        df = pd.read_csv(SRC / name, header=None, names=["Data", "Label"])
        df["Label"] = (df["Label"] > 0).astype(int)
        lab = df["Label"].to_numpy()
        first = int(np.argmax(lab)) if lab.any() else -1
        st = stratum(name)
        out = f"{idx:03d}_YAHOO_id_{idx}_{st}_tr_{N_TRAIN}_1st_{first}.csv"
        df.to_csv(OUT / "yahoo" / out, index=False)
        rows.append({"original": name, "file_name": out, "stratum": st, "n": len(df),
                     "has_label": bool(lab.any())})
    pd.DataFrame(rows).to_csv(OUT / "listing.csv", index=False)
    LOGGER.info("wrote %d series to %s", len(rows), OUT)


if __name__ == "__main__":
    main()
