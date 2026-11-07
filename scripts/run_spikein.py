"""Spike-in test: known anomalies injected into real normal stretches of YAHOO and of the other
sources (registered SP, artifacts/preregistration/spikein_hypothesis.md).

On the 472 held-out series the toggle's sign is decided mostly by YAHOO, whose anomalies are short,
often synthetic and whose labels have been questioned. This separates the series' background from
its labelled anomalies: the same level, scale and shape anomalies as the synthetic test B1-B4 are
injected into label-free stretches of real series, in two groups,

  yahoo    TSB-AD-U YAHOO series (any split)
  other19  TSB-AD-U-Eva series of the 19 sources the 472 held-out series do not contain

and TSB-AD's Sub-KNN and Sub-OCSVM score them with per-window or global normalization, exactly as in
the toggle (TSB-AD 1.5 classes, tuned hyperparameters, run_detector_normalization's replacement).

  plan   choose backgrounds and injection positions from values and labels only (no detector)
         -> artifacts/official_protocol/spikein_plan.json
  run    score every planned series -> results/official_protocol/synthetic/spikein_shard<k>.json

    python -m scripts.run_spikein plan
    python -m scripts.run_spikein run --shard k --num-shards n
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.run_contract import open_run, save_records, sha256_file
from scripts.run_detector_normalization import TUNED, VARIANTS
from scripts.run_official_protocol import DATA, OFFICIAL, OUT, SPLITS, official_metrics
from scripts.run_retune import _score

LOGGER = logging.getLogger(__name__)
PLAN = Path("artifacts/official_protocol/spikein_plan.json")
N, N_TRAIN = 1000, 500          # background length and training part (YAHOO's own training part is 500)
W_MAX = 60                      # so that a 4W anomaly fits in the test part with margins
PER_GROUP, PER_SOURCE = 20, 3
RATIOS = (0.5, 4)
KINDS = ("level", "scale", "shape")   # level +3 sigma, scale x3, shape: sawtooth (as B1-B4)
HELD_OUT_SOURCES = {"YAHOO", "UCR", "WSD", "Stock"}
DETECTORS = ("Sub_KNN", "Sub_OCSVM")
SEED = 0


def _longest_normal(labels: np.ndarray) -> tuple[int, int]:
    best, start, cur = (0, 0), 0, 0
    for i, v in enumerate(np.append(labels, 1)):
        if v == 0:
            if cur == 0:
                start = i
            cur += 1
        else:
            if cur > best[1]:
                best = (start, cur)
            cur = 0
    return best


def _candidates(group: str) -> list[str]:
    files = sorted(f.name for f in (DATA / "TSB-AD-U").glob("*.csv"))
    if group == "yahoo":
        return [f for f in files if f.split("_")[1] == "YAHOO"]
    eva = pd.read_csv(OFFICIAL / SPLITS["TSB-AD-U"])["file_name"].tolist()
    return sorted(f for f in eva if f.split("_")[1] not in HELD_OUT_SOURCES)


def plan() -> dict[str, Any]:
    from TSB_AD.utils.slidingWindows import find_length_rank

    rng = np.random.default_rng(SEED)
    backgrounds: dict[str, list[dict[str, Any]]] = {}
    for group in ("yahoo", "other19"):
        cands = _candidates(group)
        order = rng.permutation(len(cands))
        chosen, per_source = [], {}
        for i in order:
            f = cands[i]
            src = f.split("_")[1]
            if group == "other19" and per_source.get(src, 0) >= PER_SOURCE:
                continue
            df = pd.read_csv(DATA / "TSB-AD-U" / f).dropna()
            a, length = _longest_normal(df["Label"].astype(int).to_numpy())
            if length < N:
                continue
            bg = df["Data"].to_numpy(dtype=float)[a:a + N]
            if bg[:N_TRAIN].std() == 0:
                continue
            w = int(find_length_rank(bg.reshape(-1, 1), rank=1))
            if w > W_MAX:
                continue
            per_source[src] = per_source.get(src, 0) + 1
            m, s = float(bg[:N_TRAIN].mean()), float(bg[:N_TRAIN].std())
            chosen.append({"file": f, "source": src, "offset": int(a), "W": w, "train_mean": m, "train_std": s,
                           "drift_level": float((bg[N_TRAIN:].mean() - m) / s),
                           "drift_scale": float(bg[N_TRAIN:].std() / s)})
            if len(chosen) == PER_GROUP:
                break
        backgrounds[group] = chosen
    items = []
    for group, bgs in backgrounds.items():
        for b_i, b in enumerate(bgs):
            for kind in KINDS:
                for r in RATIOS:
                    length = max(2, int(round(r * b["W"])))
                    start = int(rng.integers(N_TRAIN + 50, N - length - 50 + 1))
                    items.append({"id": f"{group}|{b['file']}|{kind}|{r}", "group": group, "background": b_i,
                                  "kind": kind, "ratio": r, "start": start, "length": length})
    return {"N": N, "N_TRAIN": N_TRAIN, "W_MAX": W_MAX, "seed": SEED, "backgrounds": backgrounds, "items": items}


def build(item: dict[str, Any], bg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(DATA / "TSB-AD-U" / bg["file"]).dropna()
    x = df["Data"].to_numpy(dtype=float)[bg["offset"]:bg["offset"] + N].copy()
    seg = slice(item["start"], item["start"] + item["length"])
    m, s = bg["train_mean"], bg["train_std"]
    if item["kind"] == "level":
        x[seg] += 3.0 * s
    elif item["kind"] == "scale":
        x[seg] = m + 3.0 * (x[seg] - m)
    else:
        old = x[seg]
        period = max(4, bg["W"] // 2)
        saw = ((np.arange(item["length"]) % period) / period) - 0.5
        saw = (saw - saw.mean()) / (saw.std() + 1e-12)
        x[seg] = old.mean() + old.std() * saw
    lab = np.zeros(N, dtype=int)
    lab[seg] = 1
    return x, lab


def run_item(item: dict[str, Any], bg: dict[str, Any]) -> dict[str, Any]:
    x, lab = build(item, bg)
    rec: dict[str, Any] = {"series": item["id"], **{k: item[k] for k in ("group", "kind", "ratio", "length")},
                           "source": bg["source"], "W": bg["W"]}
    for det in DETECTORS:
        for variant in VARIANTS:
            key = f"{det}|{variant}"
            try:
                rec[key] = official_metrics(_score(det, variant, x[:, None], N_TRAIN, TUNED[det]), lab, bg["W"])
            except Exception as exc:
                rec[key] = None
                rec.setdefault("errors", {})[key] = f"{type(exc).__name__}: {str(exc)[:160]}"
    return rec


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("phase", choices=("plan", "run"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    if args.phase == "plan":
        p = plan()
        PLAN.write_text(json.dumps(p, indent=1) + "\n")
        LOGGER.info("wrote %s: %d + %d backgrounds, %d items", PLAN, len(p["backgrounds"]["yahoo"]),
                    len(p["backgrounds"]["other19"]), len(p["items"]))
        return
    p = json.loads(PLAN.read_text())
    mine = [it for i, it in enumerate(p["items"]) if i % args.num_shards == args.shard]
    out_path = OUT / "synthetic" / f"spikein_shard{args.shard}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = open_run(out_path, {"plan_sha256": sha256_file(PLAN), "detectors": list(DETECTORS),
                                  "hyperparameters": {d: TUNED[d] for d in DETECTORS},
                                  "shard": [args.shard, args.num_shards]},
                       [it["id"] for it in mine], __file__, PLAN)
    done = {r["series"] for r in records}
    for i, it in enumerate(mine, start=1):
        if it["id"] in done:
            continue
        records.append(run_item(it, p["backgrounds"][it["group"]][it["background"]]))
        save_records(out_path, records)
        LOGGER.info("%d/%d %s", i, len(mine), it["id"])


if __name__ == "__main__":
    main()
