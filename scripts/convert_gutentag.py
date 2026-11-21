"""Convert TimeEval's published GutenTAG univariate collection into TSB-AD's file format
(registered P, artifacts/preregistration/gutentag_hypothesis.md).

GutenTAG (Wenig et al., TimeEval) is a synthetic collection that no TSB-AD or TSB-UAD file contains.
Each univariate dataset has an anomaly-free training series and a test series of 10,000 points; we
take the semi-supervised variant and concatenate them, so that the training part (labels 0) is
followed by the test part, as in TSB-AD's files. The anomaly kinds and lengths come from the
collection's generation config (overview.yaml).

Input: data/timeeval_gutentag/raw/gutentag.zip, downloaded 2026-11-21 from the TimeEval authors' share
link https://my.hidrive.com/api/sharelink/download?id=j2imAkqU (MIT licence for the preprocessed
datasets, per https://timeeval.github.io/evaluation-paper/notebooks/Datasets.html; SHA-256 below).
Output: data/timeeval_gutentag/converted/<idx>_GutenTAG_id_<k>_<family>_tr_<n_train>_1st_<first>.csv
with columns Data, Label, and data/timeeval_gutentag/converted/listing.csv.

    python -m scripts.convert_gutentag   (reads the archive in memory; never extracts it to disk)
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)
ROOT = Path("data/timeeval_gutentag")
ZIP = ROOT / "raw" / "gutentag.zip"
ZIP_SHA256 = "3f3c65772adc20b06eb3abb78a8346757bc3031cba2186188dd4e51197c94d51"
OUT = ROOT / "converted"
# Kind classes, fixed in the registration from the paper's mechanism (Sec. 6): level and scale
# change a window's mean or spread; shape changes its form at a similar level and spread;
# extremum is a single-point spike.
KIND_CLASS = {"mean": "level", "platform": "level", "trend": "level", "variance": "scale", "amplitude": "scale",
              "pattern": "shape", "pattern-shift": "shape", "frequency": "shape", "extremum": "point"}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    digest = hashlib.sha256(ZIP.read_bytes()).hexdigest()
    if digest != ZIP_SHA256:
        raise SystemExit(f"{ZIP} has SHA-256 {digest}, expected {ZIP_SHA256}")
    z = zipfile.ZipFile(ZIP)
    index = list(csv.DictReader(io.StringIO(z.read("datasets.csv").decode())))
    cfg = {t["name"]: t for t in yaml.safe_load(z.read("overview.yaml"))["generated-timeseries"]}
    sel = sorted((r for r in index if r["input_type"] == "univariate" and r["train_type"] == "semi-supervised"),
                 key=lambda r: r["dataset_name"])
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    family_count: dict[str, int] = {}
    for idx, r in enumerate(sel, start=1):
        name = r["dataset_name"].rsplit(".", 1)[0]
        train = pd.read_csv(io.BytesIO(z.read(r["train_path"])))
        test = pd.read_csv(io.BytesIO(z.read(r["test_path"])))
        if int(train["is_anomaly"].sum()) != 0:
            raise SystemExit(f"{name}: training series has labelled anomalies")
        x = np.concatenate([train["value-0"].to_numpy(float), test["value-0"].to_numpy(float)])
        lab = np.concatenate([np.zeros(len(train), dtype=int), (test["is_anomaly"].to_numpy() > 0).astype(int)])
        family = name.split("-")[0]
        family_count[family] = family_count.get(family, 0) + 1
        first = int(np.argmax(lab)) if lab.any() else -1
        file_name = f"{idx:03d}_GutenTAG_id_{family_count[family]}_{family}_tr_{len(train)}_1st_{first}.csv"
        pd.DataFrame({"Data": x, "Label": lab}).to_csv(OUT / file_name, index=False)
        anomalies = cfg[name].get("anomalies", [])
        kinds = sorted({k["kind"] for a in anomalies for k in a["kinds"]})
        rows.append({"file_name": file_name, "dataset": name, "family": family, "n": int(x.size),
                     "n_train": int(len(train)), "num_anomalies": int(r["num_anomalies"]),
                     "median_anomaly_length": float(r["median_anomaly_length"]),
                     "kinds": json.dumps(kinds), "kind_classes": json.dumps(sorted({KIND_CLASS[k] for k in kinds})),
                     "anomalies_in_test_part": int(lab.sum())})
    pd.DataFrame(rows).to_csv(OUT / "listing.csv", index=False)
    LOGGER.info("wrote %d series to %s", len(rows), OUT)


if __name__ == "__main__":
    main()
