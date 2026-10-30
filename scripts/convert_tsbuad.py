"""Convert TSB-UAD Public (v1) series from sources absent from TSB-AD into TSB-AD's file format
(registered H11, artifacts/preregistration/tsbuad_hypothesis.md).

TSB-UAD (Paparrizos et al., PVLDB 2022) is TSB-AD's predecessor; most of its sources were carried
into TSB-AD. This script takes only the sources that TSB-AD does not contain:

  primary    SensorScope, Occupancy, Dodgers   (in no TSB-AD file, univariate or multivariate)
  secondary  GHL, Genesis                      (absent from TSB-AD-U; TSB-AD-M has them as
                                                multivariate series)

ECG is excluded: its series are cut from record 14046 of the MIT-BIH Long-Term ECG Database, which
TSB-AD contains as LTDB. TSB-UAD has no training split; the training part is the first 30% of each
series (fixed before any run), whatever its labels, so some training parts contain anomalies.

Input: data/tsb_uad/public/TSB-UAD-Public/<source>/*.out (value,label; no header), from
https://www.thedatum.org/datasets/TSB-UAD-Public.zip (SHA-256 below).
Output: data/tsb_uad/converted/{primary,secondary}/<idx>_<source>_id_<k>_<domain>_tr_<n_train>_1st_<first>.csv
with columns Data, Label, and data/tsb_uad/converted/listing.csv (original file, new file, set).

    python -m scripts.convert_tsbuad
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
ROOT = Path("data/tsb_uad")
ZIP_SHA256 = "ff4aa83a5a111835d410d962152e8dbebcda1039b778bae45b6b9c3f46dd49a1"
SRC = ROOT / "public" / "TSB-UAD-Public"
OUT = ROOT / "converted"
TRAIN_FRACTION = 0.3
SETS = {"primary": {"SensorScope": "Environment", "Occupancy": "Facility", "Dodgers": "Traffic"},
        "secondary": {"GHL": "Facility", "Genesis": "Facility"}}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    digest = hashlib.sha256((ROOT / "TSB-UAD-Public.zip").read_bytes()).hexdigest()
    if digest != ZIP_SHA256:
        raise SystemExit(f"TSB-UAD-Public.zip has SHA-256 {digest}, expected {ZIP_SHA256}")
    rows = []
    idx = 0
    for set_name, sources in SETS.items():
        (OUT / set_name).mkdir(parents=True, exist_ok=True)
        for source, domain in sources.items():
            for k, f in enumerate(sorted((SRC / source).glob("*.out")), start=1):
                df = pd.read_csv(f, header=None, names=["Data", "Label"])
                df["Label"] = (df["Label"] > 0).astype(int)
                n = len(df)
                n_train = int(np.floor(TRAIN_FRACTION * n))
                lab = df["Label"].to_numpy()
                first = int(np.argmax(lab)) if lab.any() else -1
                idx += 1
                name = f"{idx:03d}_{source}_id_{k}_{domain}_tr_{n_train}_1st_{first}.csv"
                df.to_csv(OUT / set_name / name, index=False)
                rows.append({"set": set_name, "source": source, "original": f.name, "file_name": name, "n": n,
                             "n_train": n_train, "anomalies_in_training_part": int(lab[:n_train].sum()),
                             "anomalies_in_test_part": int(lab[n_train:].sum())})
    pd.DataFrame(rows).to_csv(OUT / "listing.csv", index=False)
    LOGGER.info("wrote %d series to %s", len(rows), OUT)


if __name__ == "__main__":
    main()
