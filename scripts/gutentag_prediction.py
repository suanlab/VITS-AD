"""Registered predictions for P, the normalization toggle on TimeEval's GutenTAG collection
(artifacts/preregistration/gutentag_hypothesis.md). Run once, before any detector has been run on
the GutenTAG series; its output is committed with the registration and is not rewritten afterwards
(`--check` recomputes it and compares).

Uses only TSB-AD results already released (the toggle on the evaluation split and on the 472
series in neither split, and their cached series features) and label-derived or label-free
covariates of the GutenTAG series (anomaly type and length, TSB-AD's window estimate). No detector is
run here.

  P1  per detector, the mean gain on GutenTAG predicted from TSB-AD's gains by stratum (anomaly type x
      r-bin, r = median anomaly length / the detector's window), weighted by GutenTAG's strata;
      baseline: the pooled TSB-AD mean (822 series).
  P2  groups of GutenTAG series by kind class (level, scale, shape, point), from the generation config.
  P3  replication predictions (p_new, p_iid, p_sig) for the 66 pairwise orders of the twelve
      variants (six detectors x two normalizations, TSB-AD's tuned hyperparameters), discovered on the
      evaluation split (23 sources), for each of GutenTAG's five base-oscillation families as a new
      source (O1) and for the pooled collection (O2).

Output: artifacts/preregistration/gutentag_predictions.json

    python -m scripts.gutentag_prediction            # write
    python -m scripts.gutentag_prediction --check    # recompute and compare with the committed file
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.paper_numbers import SUB_DETECTORS, keyed
from scripts.replication_prediction import fit, predict

LOGGER = logging.getLogger(__name__)
ART = Path("artifacts/official_protocol")
GT = Path("data/timeeval_gutentag/converted")
OUT = Path("artifacts/preregistration/gutentag_predictions.json")
R_BINS = [0, 0.25, 1, 4, np.inf]
R_LABELS = ["<=0.25", "0.25-1", "1-4", ">4"]
MIN_STRATUM = 5


def _segments(lab: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(np.diff(np.concatenate([[0], lab, [0]])))
    return idx[1::2] - idx[::2]


def _type(lengths: np.ndarray) -> str:
    return "point" if (lengths == 1).all() else "seq" if (lengths > 1).all() else "both"


def gutentag_features() -> pd.DataFrame:
    """Anomaly type, median anomaly length and TSB-AD's window at ranks 1-3 per GutenTAG series."""
    from TSB_AD.utils.slidingWindows import find_length_rank

    listing = pd.read_csv(GT / "listing.csv")
    rows = {}
    for f in listing["file_name"]:
        df = pd.read_csv(GT / f)
        x = df["Data"].to_numpy(float).reshape(-1, 1)
        lengths = _segments(df["Label"].to_numpy(int))
        rows[f] = {"anomaly_type": _type(lengths), "median_anomaly_length": float(np.median(lengths)),
                   **{f"window_r{k}": int(find_length_rank(x, rank=k)) for k in (1, 2, 3)}}
    return pd.DataFrame(rows).T.join(listing.set_index("file_name")[["family", "kind_classes"]])


def _ranks() -> dict[str, int]:
    from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict as HP

    return {d: int(HP[d]["periodicity"]) for d in SUB_DETECTORS}


def p1(gt: pd.DataFrame) -> dict[str, Any]:
    feats = json.loads((ART / "series_features.json").read_text())
    ranks = _ranks()
    tsb = []
    for name, pat in (("eva", "TSB-AD-U/detnorm_shard*.json"), ("rest", "TSB-AD-U/detnorm_rest_shard*.json")):
        det = keyed(pat)
        f = pd.DataFrame(feats[name]).T.reindex(det.index)
        for d in SUB_DETECTORS:
            win = f[{1: "window", 2: "window_r2", 3: "window_r3"}[ranks[d]]].astype(float)
            tsb.append(pd.DataFrame({"detector": d, "gain": det[f"{d}|global"] - det[f"{d}|window"],
                                     "type": f["anomaly_type"],
                                     "rbin": pd.cut(f["median_anomaly_length"].astype(float) / win, R_BINS,
                                                    labels=R_LABELS).astype(str)}).dropna())
    tsb_all = pd.concat(tsb)
    out: dict[str, Any] = {"min_stratum": MIN_STRATUM, "by_detector": {}}
    for d in SUB_DETECTORS:
        t = tsb_all[tsb_all.detector == d]
        strata = t.groupby(["type", "rbin"])["gain"].agg(["mean", "size"])
        type_mean = t.groupby("type")["gain"].mean()
        win = gt[f"window_r{ranks[d]}"].astype(float)
        g_type = gt["anomaly_type"]
        g_bin = pd.cut(gt["median_anomaly_length"].astype(float) / win, R_BINS, labels=R_LABELS).astype(str)
        preds = []
        for ty, rb in zip(g_type, g_bin):
            if (ty, rb) in strata.index and strata.loc[(ty, rb), "size"] >= MIN_STRATUM:
                preds.append(float(strata.loc[(ty, rb), "mean"]))
            else:
                preds.append(float(type_mean[ty]))
        out["by_detector"][d] = {"stratified": float(np.mean(preds)), "pooled_tsbad": float(t["gain"].mean()),
                                 "n_tsbad": int(len(t)),
                                 "gutentag_strata": {f"{a}|{b}": int(n) for (a, b), n in
                                                     pd.Series(list(zip(g_type, g_bin))).value_counts().items()}}
    return out


def p2(gt: pd.DataFrame) -> dict[str, Any]:
    single = gt[gt["kind_classes"].map(lambda s: len(json.loads(s)) == 1)]
    cls = single["kind_classes"].map(lambda s: json.loads(s)[0])
    return {"groups": {c: sorted(single.index[cls == c]) for c in ("level", "scale", "shape", "point")}}


def p3(gt: pd.DataFrame) -> dict[str, Any]:
    eva = keyed("TSB-AD-U/detnorm_shard*.json")
    cols = [f"{d}|{v}" for d in SUB_DETECTORS for v in ("window", "global")]
    eva = eva[cols]
    src = pd.Series([s.split("_")[1] for s in eva.index], index=eva.index)
    sizes = gt["family"].value_counts()
    w = sizes / sizes.sum()
    o1, o2 = [], []
    for a, b in itertools.combinations(cols, 2):
        dd = (eva[a] - eva[b]).dropna()
        f = fit(dd, src)
        if f["d"] == 0:
            continue
        o2.append({"a": a, "b": b, **predict(f, int(sizes.sum()), float((w ** 2).sum()))})
        for fam, n in sizes.items():
            o1.append({"a": a, "b": b, "source": fam, **predict(f, int(n), 1.0)})
    return {"n_conclusions": len(o2), "families": {k: int(v) for k, v in sizes.items()}, "O1": o1, "O2": o2}


def build() -> dict[str, Any]:
    gt = gutentag_features()
    return {"gutentag_series": int(len(gt)), "anomaly_type_counts": gt["anomaly_type"].value_counts().to_dict(),
            "features": {k: {c: (v[c] if isinstance(v[c], str) else float(v[c])) for c in v.index}
                         for k, v in gt.iterrows()},
            "P1": p1(gt), "P2": p2(gt), "P3": p3(gt)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    res = json.loads(json.dumps(build()))
    if args.check:
        same = res == json.loads(OUT.read_text())
        LOGGER.info("recomputed predictions %s the committed file", "match" if same else "DIFFER from")
        raise SystemExit(0 if same else 1)
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    LOGGER.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
