"""Behavioural preprocessing checks on a second harness: TimeEval's algorithm implementations.

The paper's checks (scripts/check_preprocessing.py) call TSB-AD's detector interface. Here the same
kinds of verdict are obtained through TimeEval's own interface: each algorithm is a script that
reads a JSON config (executionType train/execute, dataInput, dataOutput, modelInput/modelOutput,
customParameters) and a CSV (timestamp, value-0, is_anomaly), and writes one score per point or per
window. We call those scripts directly (no Docker) with their default parameters on small synthetic
inputs, and map window scores to points by averaging the windows that cover a point (TimeEval's
reverse windowing).

Checks and what a verdict means (as in the paper, Sec. 8):
  label_independence  contract: scores do not change when the labels in the input file are zeroed
                      (several PyOD-based scripts read the labels to set `contamination`)
  batch_context       contract, semi-supervised only: a window's score does not change with the
                      other data scored alongside it (execute on the test series alone and after
                      a prefix of other data); unsupervised scripts fit on the scored data by design
  non_constant        degeneracy diagnostic: scores are not constant on a non-degenerate input
  level_shift         capability probe: a +3 sd level shift of 300 points is detected (point ROC-AUC
                      >= 0.8); a warning unless the algorithm declares level invariance
  affine_invariance   description, no verdict: Spearman correlation of scores on 5x+3 and on x

Input: a clone of github.com/TimeEval/TimeEval-algorithms (per-algorithm licences; not
redistributed) and a Python environment with its dependencies (see README).
Output: artifacts/timeeval_checks/checks.json and summary.md

    python -m scripts.check_timeeval --src data/timeeval_algorithms/src --python data/timeeval_algorithms/venv/bin/python
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

LOGGER = logging.getLogger(__name__)
OUT = Path("artifacts/timeeval_checks")
UNSUPERVISED = ("subsequence_lof", "subsequence_if", "subsequence_knn", "knn", "lof", "hbos", "iforest", "cof",
                "copod", "cblof", "kmeans", "pci", "phasespace_svm", "fft", "median_method", "dwt_mlead",
                "triple_es", "if_lof")
SEMI_SUPERVISED = ("subsequence_fast_mcd", "fast_mcd", "robust_pca")
N_TRAIN, N_TEST, PERIOD, SEED = 1500, 1500, 50, 0
TIMEOUT_S = 600


def _series(n: int, rng: np.random.Generator, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n)
    return np.sin(2 * np.pi * t / PERIOD + phase) + 0.1 * rng.standard_normal(n)


def _write(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    pd.DataFrame({"timestamp": np.arange(x.size), "value-0": x, "is_anomaly": y.astype(int)}).to_csv(path, index=False)


def _to_points(scores: np.ndarray, n: int) -> np.ndarray:
    """Window scores to point scores (mean of the windows covering a point); point scores unchanged."""
    if scores.size == n:
        return scores
    w = n - scores.size + 1
    if w < 1:
        raise ValueError(f"{scores.size} scores for {n} points")
    acc = np.convolve(scores, np.ones(w))
    cnt = np.convolve(np.ones(scores.size), np.ones(w))
    return acc / cnt


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    r = stats.rankdata(s)
    pos = y == 1
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


class Runner:
    def __init__(self, src: Path, python: Path, work: Path) -> None:
        self.src, self.python, self.work = src, python, work

    def run(self, algo: str, data: Path, mode: str = "execute", model: Path | None = None) -> np.ndarray:
        out = self.work / f"{algo}_{data.stem}_{mode}.out"
        cfg = {"executionType": mode, "dataInput": str(data), "dataOutput": str(out),
               "modelInput": str(model or self.work / f"{algo}.model"),
               "modelOutput": str(model or self.work / f"{algo}.model"), "customParameters": {}}
        proc = subprocess.run([str(self.python), "algorithm.py", json.dumps(cfg)], cwd=self.src / algo,
                              capture_output=True, text=True, timeout=TIMEOUT_S)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"exit {proc.returncode}")
        if mode == "train":
            return np.empty(0)
        return np.loadtxt(out, delimiter=",", ndmin=1).ravel()

    def score(self, algo: str, x: np.ndarray, y: np.ndarray, tag: str, train: tuple[np.ndarray, np.ndarray] | None,
              ) -> np.ndarray:
        test = self.work / f"{tag}.csv"
        _write(test, x, y)
        if algo in SEMI_SUPERVISED:
            tr = self.work / f"{tag}_train.csv"
            _write(tr, *train)
            model = self.work / f"{algo}_{tag}.model"
            self.run(algo, tr, "train", model)
            return self.run(algo, test, "execute", model)
        return self.run(algo, test)


def check(runner: Runner, algo: str) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    train = (_series(N_TRAIN, rng), np.zeros(N_TRAIN))
    base = _series(N_TEST, rng)
    y = np.zeros(N_TEST)
    lo, hi = 700, 1000
    y[lo:hi] = 1
    level = base.copy()
    level[lo:hi] += 3 * base.std()
    out: dict[str, Any] = {"learning_type": "semi-supervised" if algo in SEMI_SUPERVISED else "unsupervised"}
    try:
        s_level = runner.score(algo, level, y, f"{algo}_level", train)
    except Exception as exc:  # noqa: BLE001 - any failure of a third-party script is recorded, not raised
        out["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return out
    if not np.all(np.isfinite(s_level)):
        # A script that returns NaN or inf here most likely meets a library version other than its
        # Docker image's; no verdict is given for it.
        out["error"] = f"non-finite scores in this environment ({int((~np.isfinite(s_level)).sum())} of {s_level.size})"
        return out
    p_level = _to_points(s_level, N_TEST)
    out["output"] = "points" if s_level.size == N_TEST else f"windows (w={N_TEST - s_level.size + 1})"
    out["non_constant"] = {"status": "pass" if np.ptp(s_level) > 1e-12 else "diagnostic", "kind": "degeneracy"}
    if np.ptp(s_level) <= 1e-12:
        # A constant output is the degeneracy above; its ROC-AUC would only measure rounding noise.
        out["level_shift"] = {"status": "n/a (constant output)", "kind": "capability", "roc_auc": None}
    else:
        auc = _auc(y, p_level)
        out["level_shift"] = {"status": "pass" if auc >= 0.8 else "warning", "kind": "capability", "roc_auc": auc}
    try:
        s_nolab = runner.score(algo, level, np.zeros(N_TEST), f"{algo}_nolabel", train)
        same = s_nolab.size == s_level.size
        diff = float(np.max(np.abs(s_nolab - s_level))) if same else float("inf")
        shift = s_nolab - s_level if same else None
        out["label_independence"] = {"status": "pass" if diff <= 1e-9 else "violation", "kind": "contract",
                                     "max_abs_diff": diff,
                                     # A constant shift leaves every ranking measure unchanged.
                                     "constant_shift": bool(same and np.ptp(shift) <= 1e-9),
                                     "same_ranks": bool(same and np.array_equal(stats.rankdata(s_nolab),
                                                                                 stats.rankdata(s_level)))}
    except Exception as exc:  # noqa: BLE001
        out["label_independence"] = {"status": "not run", "error": str(exc)[:200]}
    try:
        s_aff = runner.score(algo, 5 * level + 3, y, f"{algo}_affine", (5 * train[0] + 3, train[1]))
        rho = float(stats.spearmanr(s_aff, s_level).statistic) if s_aff.size == s_level.size else float("nan")
        out["affine_invariance"] = {"spearman": rho}
    except Exception as exc:  # noqa: BLE001
        out["affine_invariance"] = {"error": str(exc)[:200]}
    if algo in SEMI_SUPERVISED:
        prefix = _series(500, rng, phase=1.0) + 5.0       # other data at another level
        try:
            s_ctx = runner.score(algo, np.concatenate([prefix, level]), np.concatenate([np.zeros(500), y]),
                                 f"{algo}_context", train)
            tail = s_ctx[-s_level.size:]
            diff = float(np.max(np.abs(tail - s_level)))
            out["batch_context"] = {"status": "pass" if diff <= 1e-9 * max(1.0, float(np.max(np.abs(s_level))))
                                    else "violation", "kind": "contract", "max_abs_diff": diff}
        except Exception as exc:  # noqa: BLE001
            out["batch_context"] = {"status": "not run", "error": str(exc)[:200]}
    else:
        out["batch_context"] = {"status": "n/a", "note": "fits on the scored data by design"}
    return out


def summary(res: dict[str, Any]) -> str:
    lines = ["# Behavioural checks on TimeEval's algorithm implementations", "",
             "Generated by `scripts/check_timeeval.py`; see its docstring for what each verdict means.", "",
             "| algorithm | learning | output | label independence | batch context | non-constant | level shift (AUC) | affine Spearman |",
             "|---|---|---|---|---|---|---|---|"]
    for a, r in res["algorithms"].items():
        if "error" in r:
            lines.append(f"| {a} | {r['learning_type']} | not run: {r['error'][:60]} | | | | | |")
            continue
        lv = r["level_shift"]
        li = r["label_independence"]
        shifted = li["status"] == "violation" and li.get("constant_shift") and li.get("same_ranks")
        lab = li["status"] + (" (constant shift; same ranks)" if shifted else "")
        auc = "" if lv["roc_auc"] is None else f" ({lv['roc_auc']:.2f})"
        lines.append(f"| {a} | {r['learning_type']} | {r['output']} | {lab} | "
                     f"{r['batch_context']['status']} | {r['non_constant']['status']} | {lv['status']}{auc} | "
                     f"{r['affine_invariance'].get('spearman', float('nan')):.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path)
    ap.add_argument("--python", type=Path)
    ap.add_argument("--algorithms", default=None)
    ap.add_argument("--summary-only", action="store_true", help="rewrite summary.md from checks.json")
    args = ap.parse_args()
    if args.summary_only:
        (OUT / "summary.md").write_text(summary(json.loads((OUT / "checks.json").read_text())))
        return
    algos = [a for a in (args.algorithms.split(",") if args.algorithms else UNSUPERVISED + SEMI_SUPERVISED)]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.src, capture_output=True, text=True).stdout.strip()
    res: dict[str, Any] = {"timeeval_algorithms_commit": commit, "seed": SEED, "algorithms": {}}
    with tempfile.TemporaryDirectory() as tmp:
        runner = Runner(args.src.absolute(), args.python.absolute(), Path(tmp))  # a venv python must not be resolved
        for a in algos:
            res["algorithms"][a] = check(runner, a)
            LOGGER.info("%s: %s", a, {k: v.get("status") for k, v in res["algorithms"][a].items() if isinstance(v, dict)})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checks.json").write_text(json.dumps(res, indent=1) + "\n")
    (OUT / "summary.md").write_text(summary(res))
    LOGGER.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
