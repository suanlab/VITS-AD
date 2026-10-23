"""Follow-up to the registered boundary test: clean negatives (F1) and duration at fixed
prevalence (F2). Design: artifacts/preregistration/boundary_followup.md.

F1 reruns the B1-B4 series (scripts/run_boundary_synthetic.py, unchanged) and adds the interior
ROC-AUC against normal points at least W from every anomalous point. F2 uses longer series in which
the total number of anomalous test points is fixed while the duration of each anomaly varies.

Output: results/official_protocol/synthetic/boundary_followup.json

    python -m scripts.run_boundary_followup
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.run_boundary_synthetic import CELLS, RATIOS, SEEDS, make_series
from scripts.run_official_protocol import OUT, W, _nn_distance, _window_z, official_metrics, to_points, windows_at

LOGGER = logging.getLogger(__name__)
N_TRAIN, N_TEST = 3000, 12000
PREVALENCE = (400, 1600)
DURATIONS = (25, 100, 400, 1600)
F2_KINDS = (("level", 1.0), ("scale", 2.0), ("shape", 0.0))
F2_SEEDS = 10


def score(x: np.ndarray, n_train: int, variant: str) -> np.ndarray:
    n = x.size
    z = (x - x[:n_train].mean()) / x[:n_train].std()
    starts, train = np.arange(0, n - W + 1), np.arange(0, n_train - W + 1)
    aw = windows_at(z[:, None], starts).reshape(len(starts), -1)
    tw = windows_at(z[:, None], train).reshape(len(train), -1)
    if variant == "per_window":
        aw, tw = _window_z(aw), _window_z(tw)
    return to_points(_nn_distance(tw, aw), starts, n)


def _segments(lab: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(np.diff(np.concatenate([[0], lab, [0]])))
    return list(zip(idx[::2], idx[1::2]))


def masks(lab: np.ndarray, n_train: int) -> dict[str, np.ndarray]:
    n = lab.size
    pos = np.arange(n)
    inside = lab.astype(bool)
    edge = np.zeros(n, dtype=bool)
    dist = np.full(n, np.inf)
    for a, b in _segments(lab):
        edge[a:b] |= (pos[a:b] - a < W // 2) | (b - 1 - pos[a:b] < W // 2)
        dist = np.minimum(dist, np.maximum(0, np.maximum(a - pos, pos - (b - 1))))
    normal = (pos >= n_train) & ~inside
    return {"interior": inside & ~edge, "boundary": edge, "normal": normal,
            "clean": normal & (dist >= W), "test": pos >= n_train}


def _auc(s: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> float | None:
    if not pos.any() or not neg.any():
        return None
    y = np.r_[np.ones(pos.sum()), np.zeros(neg.sum())]
    return float(roc_auc_score(y, np.r_[s[pos], s[neg]]))


def measures(s: np.ndarray, lab: np.ndarray, n_train: int) -> dict[str, Any]:
    m = masks(lab, n_train)
    t = m["test"]
    return {"auc_interior": _auc(s, m["interior"], m["normal"]),
            "auc_interior_clean": _auc(s, m["interior"], m["clean"]),
            "auc_boundary_clean": _auc(s, m["boundary"], m["clean"]),
            "test_roc_auc": float(roc_auc_score(lab[t], s[t])),
            "test_auc_pr": float(average_precision_score(lab[t], s[t])),
            "VUS-PR": official_metrics(s, lab, 50)["VUS-PR"]}


def f2_series(kind: str, mag: float, total: int, d: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_TEST
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 50) + 0.5 * np.sin(2 * np.pi * t / 170) + 0.2 * rng.standard_normal(n)
    sigma, m = x[:N_TRAIN].std(), x[:N_TRAIN].mean()
    k = total // d
    slot = N_TEST // k
    lab = np.zeros(n, dtype=int)
    for j in range(k):
        lo = N_TRAIN + j * slot + W          # at least W normal steps before each anomaly
        hi = N_TRAIN + (j + 1) * slot - d    # ends inside its slot, so anomalies are >= W apart
        a = int(rng.integers(lo, hi + 1))
        seg = slice(a, a + d)
        if kind == "level":
            x[seg] += mag * sigma
        elif kind == "scale":
            x[seg] = m + mag * (x[seg] - m)
        else:
            old = x[seg]
            saw = ((np.arange(d) % 50) / 50.0) - 0.5
            saw = (saw - saw.mean()) / (saw.std() + 1e-12)
            x[seg] = old.mean() + old.std() * saw
        lab[seg] = 1
    return x, lab


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    recs: list[dict[str, Any]] = []
    for kind, mag in CELLS:
        for ratio in RATIOS:
            for seed in range(SEEDS):
                x, lab = make_series(kind, mag, ratio, seed)
                for variant in ("global", "per_window"):
                    recs.append({"part": "F1", "kind": kind, "mag": mag, "ratio": ratio, "seed": seed,
                                 "variant": variant, **measures(score(x, 3000, variant), lab, 3000)})
            LOGGER.info("F1 %s %.1f r=%s", kind, mag, ratio)
    for kind, mag in F2_KINDS:
        for total in PREVALENCE:
            for d in DURATIONS:
                if d > total:
                    continue
                for seed in range(F2_SEEDS):
                    x, lab = f2_series(kind, mag, total, d, seed)
                    for variant in ("global", "per_window"):
                        recs.append({"part": "F2", "kind": kind, "mag": mag, "total": total, "d": d,
                                     "seed": seed, "variant": variant,
                                     **measures(score(x, N_TRAIN, variant), lab, N_TRAIN)})
                LOGGER.info("F2 %s P=%d d=%d", kind, total, d)
    path = OUT / "synthetic" / "boundary_followup.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(recs, indent=1) + "\n")
    LOGGER.info("wrote %s (%d records)", path, len(recs))


if __name__ == "__main__":
    main()
