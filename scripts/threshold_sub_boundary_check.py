"""Exploratory check designed after the T1 result (artifacts/preregistration/threshold_sub_hypothesis.md):
does the injected step at the end of the training part raise the calibrated threshold?

For 40 TSB-AD-U-Eva series (drawn with seed 0 among series whose training part has at least 500
points and whose first anomaly lies before point 20,000), Sub-KNN and Sub-HBOS are run as in
run_threshold_sub.py without drift and with the +3 level shift, and the FAR at the 99th-percentile
threshold is computed twice: with the registered calibration region (second half of the training
part) and with that region ending GAP points before the end of the training part.

Output: artifacts/official_protocol/threshold_sub_boundary.json

    python -m scripts.threshold_sub_boundary_check
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_detector_normalization import DATA, OFFICIAL, SPLITS, TUNED, _patch, _runner
from scripts.run_drift_injection import inject

LOGGER = logging.getLogger(__name__)
GAP = 200
OUT = Path("artifacts/official_protocol/threshold_sub_boundary.json")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    files = pd.read_csv(OFFICIAL / SPLITS["TSB-AD-U"])["file_name"].tolist()
    pool = [f for f in files if int(f.split("_")[-1].split(".")[0]) < 20000 and int(f.split("_")[-3]) >= 500]
    sample = list(np.random.default_rng(0).choice(pool, 40, replace=False))
    out = []
    for f in sample:
        df = pd.read_csv(DATA / "TSB-AD-U" / f).dropna()
        x0 = df.iloc[:, 0].to_numpy(float)
        lab = df["Label"].astype(int).to_numpy()
        nt, n = int(f.split("_")[-3]), x0.size
        idx = np.arange(n)
        normal = (idx >= nt) & (lab == 0)
        if not ((idx >= nt) & (lab == 1)).any() or not normal.any():
            continue
        cal = (idx >= nt // 2) & (idx < nt)
        cal_gap = (idx >= nt // 2) & (idx < nt - GAP)
        if cal_gap.sum() < 20:
            continue
        for det in ("Sub_KNN", "Sub_HBOS"):
            for drift in ("none", "level3"):
                x = inject(x0, nt, drift)
                mu, sd = x[:nt].mean(), x[:nt].std() + 1e-8
                for v in ("window", "global"):
                    _patch(v)
                    s = np.asarray(_runner(det)(((x - mu) / sd if v == "global" else x).reshape(-1, 1),
                                                **TUNED[det]), float).ravel()
                    r = {"series": f, "det": det, "drift": drift, "v": v}
                    for name, c in (("cal", cal), ("cal_gap", cal_gap)):
                        r[name] = float((s[normal] > np.quantile(s[c], 0.99)).mean())
                    r["last_cal_max_ratio"] = float(s[(idx >= nt - GAP) & (idx < nt)].max()
                                                    / (np.quantile(s[cal_gap], 0.99) + 1e-12))
                    out.append(r)
        LOGGER.info("%s", f)
    OUT.write_text(json.dumps(out) + "\n")
    LOGGER.info("wrote %s (%d series)", OUT, len({r["series"] for r in out}))


if __name__ == "__main__":
    main()
