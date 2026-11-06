"""VUS-PR of a constant score on every TSB-AD-U-Eva series (and, with --split, on the 472 held-out
series or the H11 TSB-UAD sets), with TSB-AD's own get_metrics.

At the commit of the uploaded results, TSB-AD's univariate IForest and LOF z-score each time step
across its single channel, which maps every value to 0 and yields a constant score. This script
computes what such a score attains under the official metric and sliding-window parameter, to
compare with the published values of those two detectors.

For the held-out series and the H11 sets it is the chance floor against which their VUS-PR is read
(exploratory, added 2026-11-06 after a mock review).

Output: artifacts/official_protocol/constant_score[_<split>].json (per-shard parts under
results/official_protocol/TSB-AD-U/constant[_<split>]_shard<k>.json, merged by --merge)

    python -m scripts.constant_score_vus [--split rest|uad_primary|uad_secondary] --shard k --num-shards K
    python -m scripts.constant_score_vus [--split ...] --merge
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_official_protocol import DATA, OFFICIAL, OUT as RES, SPLITS, official_metrics

LOGGER = logging.getLogger(__name__)
OUT = Path("artifacts/official_protocol/constant_score.json")


def main() -> None:
    from TSB_AD.utils.slidingWindows import find_length_rank

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--split", choices=("eva", "rest", "uad_primary", "uad_secondary"), default="eva")
    args = ap.parse_args()
    tag = "" if args.split == "eva" else f"_{args.split}"
    out_file = OUT.with_name(f"constant_score{tag}.json")
    if args.merge:
        out: dict[str, dict[str, float]] = {}
        for f in sorted((RES / "TSB-AD-U").glob(f"constant{tag}_shard*.json")):
            out.update(json.loads(f.read_text()))
        out_file.write_text(json.dumps(dict(sorted(out.items())), indent=1) + "\n")
        LOGGER.info("wrote %s (%d series, mean VUS-PR %.3f)", out_file, len(out),
                    np.mean([v["VUS-PR"] for v in out.values()]))
        return
    data_dir = DATA / "TSB-AD-U"
    if args.split == "eva":
        files = pd.read_csv(OFFICIAL / SPLITS["TSB-AD-U"])["file_name"].tolist()
    elif args.split == "rest":
        used = set(pd.read_csv(OFFICIAL / SPLITS["TSB-AD-U"])["file_name"])
        used |= set(pd.read_csv(OFFICIAL / "TSB-AD-U-Tuning.csv")["file_name"])
        files = sorted(f.name for f in data_dir.glob("*.csv") if f.name not in used)
    else:
        uad = pd.read_csv(Path("data/tsb_uad/converted/listing.csv"))
        files = uad.loc[uad["set"] == args.split[4:], "file_name"].tolist()
        data_dir = Path("data/tsb_uad/converted") / args.split[4:]
    mine = [f for i, f in enumerate(files) if i % args.num_shards == args.shard]
    path = RES / "TSB-AD-U" / f"constant{tag}_shard{args.shard}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    out = json.loads(path.read_text()) if path.exists() else {}
    for k, f in enumerate(mine, start=1):
        if f in out:
            continue
        df = pd.read_csv(data_dir / f).dropna()
        x = df.iloc[:, 0].to_numpy(dtype=float)
        labels = df["Label"].astype(int).to_numpy()
        sliding = int(find_length_rank(x.reshape(-1, 1), rank=1))
        m = official_metrics(np.zeros_like(x), labels, sliding)
        out[f] = {"VUS-PR": m["VUS-PR"], "AUC-PR": m["AUC-PR"], "anomaly_ratio": float(labels.mean())}
        path.write_text(json.dumps(out, indent=1) + "\n")
        LOGGER.info("%d/%d %s", k, len(mine), f)


if __name__ == "__main__":
    main()
