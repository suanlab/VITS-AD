"""Re-tune both normalization variants of TSB-AD's Sub- detectors (registered RT,
artifacts/preregistration/retune_hypothesis.md).

The toggle in run_detector_normalization.py gives both variants the hyperparameters TSB-AD tuned
under per-window normalization. Here each variant gets its own: TSB-AD's own grid
(Uni_algo_HP_dict of the installed TSB-AD 1.5) is searched on the 48 series of TSB-AD-U-Tuning,
separately for the per-window and the global variant, and the configuration with the highest
mean VUS-PR is kept (a failed run scores 0, as in TSB-AD's HP_Tuning_U.py; ties go to the first
configuration in grid order). The kept configurations are then run on the evaluation split (350)
and on the 472 series in neither split.

    python -m scripts.run_retune grid  --shard k --num-shards n     # tuning split, every config
    python -m scripts.run_retune select                             # -> retune_selection.json
    python -m scripts.run_retune apply --split eva|rest --shard k --num-shards n

Outputs (results/official_protocol/TSB-AD-U/):
  retune_grid_shard<k>.json      per tuning series: {"<det>|<variant>|<config>": VUS-PR or null}
  retune_selection.json          per detector and variant: kept configuration, its tuning mean,
                                 and whether it equals TSB-AD's tuned one
  retune_<split>_shard<k>.json   per series: {"<det>|<variant>": metrics} for each detector and
                                 variant whose kept configuration differs from TSB-AD's tuned one;
                                 where it is the same, the toggle's existing result is that value
                                 (detnorm*_shard*.json; repeated runs were identical, see
                                 repeat_determinism in paper_numbers.json)

Shard files carry a --part suffix so that processes running different detectors on the same
shard write different files; readers merge them by series name.
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

from scripts.run_contract import open_run, require_inputs, save_records, sha256_file
from scripts.run_detector_normalization import SEMI_SUPERVISED, TUNED, VARIANTS, _patch, _runner
from scripts.run_official_protocol import DATA, OFFICIAL, OUT, SPLITS, TUNING_SPLITS, official_metrics

LOGGER = logging.getLogger(__name__)
SUBSET = "TSB-AD-U"
DETECTORS = ("Sub_KNN", "Sub_MCD", "Sub_OCSVM", "Sub_HBOS", "Sub_IForest", "Sub_LOF")
SELECTION = OUT / SUBSET / "retune_selection.json"


def grid(det: str) -> list[dict[str, Any]]:
    """TSB-AD's own search grid for ``det``, in its itertools.product order."""
    from TSB_AD.HP_list import Uni_algo_HP_dict

    keys, values = zip(*Uni_algo_HP_dict[det].items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def config_key(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True)


def _load(path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    from TSB_AD.utils.slidingWindows import find_length_rank

    df = pd.read_csv(path).dropna()
    x = df.iloc[:, 0:-1].to_numpy(dtype=float)
    labels = df["Label"].astype(int).to_numpy()
    n_train = int(path.stem.split("_")[-3])
    sliding = int(find_length_rank(x[:, 0].reshape(-1, 1), rank=1))
    return x, labels, n_train, sliding


def _score(det: str, variant: str, x: np.ndarray, n_train: int, params: dict[str, Any]) -> np.ndarray:
    _patch(variant)
    if variant == "global":
        x = (x - x[:n_train].mean(axis=0)) / (x[:n_train].std(axis=0) + 1e-8)
    run = _runner(det)
    score = run(x[:n_train], x, **params) if det in SEMI_SUPERVISED else run(x, **params)
    score = np.asarray(score, dtype=float).ravel()
    if score.size != x.shape[0]:
        raise ValueError(f"score vector has size {score.size}, series has {x.shape[0]}")
    if not np.all(np.isfinite(score)):
        raise ValueError(f"{int((~np.isfinite(score)).sum())} non-finite scores of {score.size}")
    return score


def run_grid_series(path: Path, detectors: list[str], configs: slice = slice(None)) -> dict[str, Any]:
    x, labels, n_train, sliding = _load(path)
    rec: dict[str, Any] = {"series": path.name, "n": int(x.shape[0]), "n_train": n_train}
    for det in detectors:
        for variant in VARIANTS:
            for params in grid(det)[configs]:
                key = f"{det}|{variant}|{config_key(params)}"
                try:
                    rec[key] = official_metrics(_score(det, variant, x, n_train, params), labels, sliding)["VUS-PR"]
                except Exception as exc:
                    rec[key] = None
                    rec.setdefault("errors", {})[key] = f"{type(exc).__name__}: {str(exc)[:160]}"
    return rec


def select(records: list[dict[str, Any]], detectors: list[str]) -> dict[str, Any]:
    """Highest mean VUS-PR over the tuning series, failures scored 0, ties to grid order."""
    out: dict[str, Any] = {"n_series": len(records), "rule": "max mean VUS-PR, failed run = 0, ties: first in grid order"}
    for det in detectors:
        for variant in VARIANTS:
            means = []
            for params in grid(det):
                key = f"{det}|{variant}|{config_key(params)}"
                vals = [r.get(key) for r in records]
                means.append((float(np.mean([0.0 if v is None else v for v in vals])),
                              sum(v is None for v in vals), params))
            best = max(range(len(means)), key=lambda i: (means[i][0], -i))
            out[f"{det}|{variant}"] = {
                "params": means[best][2], "tuning_mean": means[best][0], "failures": means[best][1],
                "equals_tsbad_tuned": means[best][2] == TUNED[det],
                "tsbad_tuned_mean": next(m for m, _, p in means if p == TUNED[det]),
                "grid": [{"params": p, "mean": m, "failures": f} for m, f, p in means],
            }
    return out


def run_apply_series(path: Path, keys: list[str], chosen: dict[str, Any]) -> dict[str, Any] | None:
    x, labels, n_train, sliding = _load(path)
    if labels.max() == 0:
        return None
    rec: dict[str, Any] = {"series": path.name, "n": int(x.shape[0]), "n_train": n_train}
    for key in keys:
        det, variant = key.split("|")
        try:
            rec[key] = official_metrics(_score(det, variant, x, n_train, chosen[key]["params"]), labels, sliding)
        except Exception as exc:
            LOGGER.warning("%s [%s] failed: %s: %s", path.name, key, type(exc).__name__, str(exc)[:160])
            rec[key] = None
            rec.setdefault("errors", {})[key] = f"{type(exc).__name__}: {str(exc)[:160]}"
    return rec


def _files(split: str) -> tuple[list[str], Path | None]:
    if split == "rest":
        used = set(pd.read_csv(OFFICIAL / SPLITS[SUBSET])["file_name"])
        used |= set(pd.read_csv(OFFICIAL / TUNING_SPLITS[SUBSET])["file_name"])
        return sorted(f.name for f in (DATA / SUBSET).glob("*.csv") if f.name not in used), None
    listing = OFFICIAL / (SPLITS if split == "eva" else TUNING_SPLITS)[SUBSET]
    return pd.read_csv(listing)["file_name"].tolist(), listing


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("phase", choices=("grid", "select", "apply"))
    ap.add_argument("--split", choices=("eva", "rest"), default="eva", help="apply: series to run the kept configs on")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--detectors", default=",".join(DETECTORS))
    ap.add_argument("--out-dir", type=Path, default=OUT / SUBSET, help="where shard files go")
    ap.add_argument("--part", default="", help="output-file suffix, e.g. the detector a process runs")
    ap.add_argument("--selection", type=Path, default=None,
                    help="select: where to write; apply: which selection to run (default retune_selection.json). "
                         "Lets the detectors whose grid is complete start before the others")
    ap.add_argument("--series", default=None,
                    help="grid: comma-separated tuning series to run instead of a shard (splits one long series' work)")
    ap.add_argument("--configs", default=None,
                    help="grid: a slice 'i:j' of each detector's grid, so one series' configurations can be split "
                         "across processes; readers merge the records by series")
    args = ap.parse_args()
    detectors = [d for d in args.detectors.split(",") if d]

    if args.phase == "select":
        records: dict[str, dict[str, Any]] = {}
        for f in sorted(args.out_dir.glob("retune_grid_shard*.json")):
            for r in json.loads(f.read_text()):
                old = records.setdefault(r["series"], {})
                # Files of processes that split a series' work must agree on any key they share.
                clash = [k for k in r if "|" in k and k in old and old[k] != r[k]
                         and not (old[k] is None and r[k] is None)]
                if clash:
                    raise SystemExit(f"{f.name}: {r['series']} disagrees with an earlier file on {clash[:3]}")
                old.update(r)
        expected, _ = _files("tuning")
        if sorted(records) != sorted(expected):
            raise SystemExit(f"grid covers {len(records)} of {len(expected)} tuning series; finish it first")
        # Work may be split across processes by configuration: every series must carry every key
        # (a failed run is stored as None and scores 0; a missing key means the run has not happened).
        need = {f"{d}|{v}|{config_key(c)}" for d in detectors for v in VARIANTS for c in grid(d)}
        gaps = {s: len(need - set(r)) for s, r in records.items() if need - set(r)}
        if gaps:
            raise SystemExit(f"{len(gaps)} tuning series lack some configurations, e.g. {list(gaps.items())[:3]}")
        sel = select(list(records.values()), detectors)
        sel["grid_files_sha256"] = {f.name: sha256_file(f) for f in sorted(args.out_dir.glob("retune_grid_shard*.json"))}
        sel_out = args.selection or args.out_dir / SELECTION.name
        sel_out.write_text(json.dumps(sel, indent=1) + "\n")
        LOGGER.info("wrote %s", sel_out)
        return

    split = "tuning" if args.phase == "grid" else args.split
    files, listing = _files(split)
    files, missing = require_inputs(files, DATA / SUBSET)
    mine = [f for i, f in enumerate(files) if i % args.num_shards == args.shard]
    if args.series:
        mine = [f for f in files if f in set(args.series.split(","))]
    configs = slice(*[int(v) if v else None for v in args.configs.split(":")]) if args.configs else slice(None)
    chosen: dict[str, Any] = {}
    config: dict[str, Any] = {"phase": args.phase, "split": split, "detectors": detectors,
                              "shard": [args.shard, args.num_shards], "series": args.series, "configs": args.configs}
    if args.phase == "apply":
        sel_path = args.selection or args.out_dir / SELECTION.name
        chosen = json.loads(sel_path.read_text())
        config["selection_sha256"] = sha256_file(sel_path)
        changed = [k for d in detectors for k in (f"{d}|window", f"{d}|global") if not chosen[k]["equals_tsbad_tuned"]]
        config["kept"] = {k: chosen[k]["params"] for k in changed}
        if not changed:
            LOGGER.info("every kept configuration equals TSB-AD's tuned one; nothing to run")
            return
    prefix = "retune_grid" if args.phase == "grid" else f"retune_{split}"
    out_path = args.out_dir / f"{prefix}_shard{args.shard}{args.part}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records_list = open_run(out_path, config, mine, __file__, listing, missing)
    done = {r["series"] for r in records_list}
    for i, f in enumerate(mine, start=1):
        if f in done:
            continue
        rec = (run_grid_series(DATA / SUBSET / f, detectors, configs) if args.phase == "grid"
               else run_apply_series(DATA / SUBSET / f, list(config["kept"]), chosen))
        if rec is not None:
            records_list.append(rec)
        save_records(out_path, records_list)
        LOGGER.info("%d/%d %s", i, len(mine), f)


if __name__ == "__main__":
    main()
