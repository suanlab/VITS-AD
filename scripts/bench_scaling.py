"""Run time and peak memory of Raw-1NN and Raw-LW as the reference, the stream and the channels grow.

Complements bench_official_cost.py (per-series cost on TSB-AD series). Here the inputs are
synthetic, because cost depends only on their shapes: n_ref reference windows and n_eval scored
windows of W=100 steps and D channels, flattened to W*D numbers, drawn i.i.d. Gaussian with a fixed
seed. Each configuration runs in a fresh process so that its peak resident memory (ru_maxrss) is
its own; that process times fit and scoring with the same code as run_official_protocol.py
(_nn_distance for Raw-1NN, WoodburyLedoitWolfScorer for Raw-LW), single-threaded, after one
untimed warm-up run; reported is the run with the smallest fit-plus-score time of --repeats runs,
with the median and all runs stored. Data generation is outside the timer; peak memory covers the
whole process. A configuration whose estimated memory (inputs, plus the n_ref x n_ref matrix that
the Woodbury form inverts for Raw-LW) exceeds --max-gb is skipped and recorded as skipped.

The server is shared with other jobs; the CPU model and load at run time are stored with the results.

Output: artifacts/official_protocol/scaling.json

    python -m scripts.bench_scaling
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

LOGGER = logging.getLogger(__name__)
OUT = Path("artifacts/official_protocol/scaling.json")
W = 100
GRID = [  # (method, n_ref, n_eval, channels)
    *[(m, r, 3000, 1) for m in ("raw_knn", "raw_lw") for r in (1000, 10000, 100000)],
    *[(m, 10000, e, 1) for m in ("raw_knn", "raw_lw") for e in (30000, 100000)],
    *[(m, 3000, 3000, d) for m in ("raw_knn", "raw_lw") for d in (25, 51)],
    *[(m, 10000, 3000, d) for m in ("raw_knn", "raw_lw") for d in (25, 51)],
]


def _one(method: str, n_ref: int, n_eval: int, channels: int, repeats: int) -> dict[str, Any]:
    from scripts.run_official_protocol import _nn_distance
    from src.scoring.woodbury_mahalanobis import WoodburyLedoitWolfScorer

    rng = np.random.default_rng(0)
    dim = W * channels
    ref = rng.standard_normal((n_ref, dim))
    query = rng.standard_normal((n_eval, dim))
    fit_s, score_s = [], []
    for rep in range(repeats + 1):
        if method == "raw_knn":
            t0 = time.perf_counter()
            _nn_distance(ref, query)            # index build and query are one call there
            if rep:                             # run 0 is an untimed warm-up
                fit_s.append(0.0)
                score_s.append(time.perf_counter() - t0)
        else:
            t0 = time.perf_counter()
            scorer = WoodburyLedoitWolfScorer().fit(ref)
            t1 = time.perf_counter()
            scorer.score(query)
            if rep:
                fit_s.append(t1 - t0)
                score_s.append(time.perf_counter() - t1)
    totals = [f + c for f, c in zip(fit_s, score_s)]
    best = int(np.argmin(totals))
    return {"method": method, "n_ref": n_ref, "n_eval": n_eval, "channels": channels, "dim": dim,
            "fit_s": fit_s[best], "score_s": score_s[best], "total_s": totals[best],
            "total_s_median": float(np.median(totals)), "total_s_all": totals,
            "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
            "repeats": repeats}


def _cpu() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-gb", type=float, default=12.0)
    ap.add_argument("--child", nargs=4, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        m, r, e, d = args.child
        print(json.dumps(_one(m, int(r), int(e), int(d), args.repeats)))
        return
    rows: list[dict[str, Any]] = []
    for m, r, e, d in GRID:
        gb = ((r + e) * W * d + (3 * r * r if m == "raw_lw" else 0)) * 8 / 1024 ** 3
        if gb > args.max_gb:
            LOGGER.info("skip %s n_ref=%d n_eval=%d D=%d (about %.0f GB)", m, r, e, d, gb)
            rows.append({"method": m, "n_ref": r, "n_eval": e, "channels": d,
                         "skipped": f"estimated {gb:.0f} GB > {args.max_gb:.0f} GB"})
            continue
        res = subprocess.run([sys.executable, "-m", "scripts.bench_scaling", "--repeats",
                              str(args.repeats), "--child", m, str(r), str(e), str(d)],
                             capture_output=True, text=True, check=True)
        row = json.loads(res.stdout.strip().splitlines()[-1])
        rows.append(row)
        LOGGER.info("%s", row)
        OUT.write_text(json.dumps({"threads": 1, "cpu": _cpu(), "load_at_start": os.getloadavg(),
                                   "statistic": "run with the smallest fit+score time; median stored",
                                   "window": W, "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
