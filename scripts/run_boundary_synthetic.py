"""Synthetic test of where per-window normalization loses anomalies: edges vs interior, by duration.

Design and decision rules: artifacts/preregistration/boundary_hypothesis.md (B1-B4). Raw-1NN with
global or per-window normalization on synthetic series with one level, scale or shape anomaly whose
duration is 0.25, 1, 4 or 16 window lengths.

Output: results/official_protocol/synthetic/boundary.json (one record per series and variant)

    python -m scripts.run_boundary_synthetic
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.run_official_protocol import OUT, W, _nn_distance, _window_z, official_metrics, to_points, windows_at

LOGGER = logging.getLogger(__name__)
N, N_TRAIN, SEEDS = 6000, 3000, 20
RATIOS = (0.25, 1, 4, 16)
CELLS = [("level", 1.0), ("level", 3.0), ("scale", 2.0), ("scale", 3.0), ("shape", 0.0)]


def make_series(kind: str, mag: float, ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(N)
    x = np.sin(2 * np.pi * t / 50) + 0.5 * np.sin(2 * np.pi * t / 170) + 0.2 * rng.standard_normal(N)
    sigma, m = x[:N_TRAIN].std(), x[:N_TRAIN].mean()
    d = int(ratio * W)
    a = int(rng.integers(3200, N - d - 200 + 1))
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
    lab = np.zeros(N, dtype=int)
    lab[seg] = 1
    return x, lab


def score(x: np.ndarray, variant: str) -> np.ndarray:
    z = (x - x[:N_TRAIN].mean()) / x[:N_TRAIN].std()
    starts = np.arange(0, N - W + 1)
    train = np.arange(0, N_TRAIN - W + 1)
    aw = windows_at(z[:, None], starts).reshape(len(starts), -1)
    tw = windows_at(z[:, None], train).reshape(len(train), -1)
    if variant == "per_window":
        aw, tw = _window_z(aw), _window_z(tw)
    return to_points(_nn_distance(tw, aw), starts, N)


def regions(lab: np.ndarray) -> dict[str, np.ndarray]:
    idx = np.flatnonzero(lab)
    a, b = idx[0], idx[-1]
    pos = np.arange(N)
    inside = lab.astype(bool)
    edge = inside & ((pos - a < W // 2) | (b - pos < W // 2))
    return {"boundary": edge, "interior": inside & ~edge, "normal": (pos >= N_TRAIN) & ~inside}


def run() -> list[dict[str, Any]]:
    out = []
    for kind, mag in CELLS:
        for ratio in RATIOS:
            for seed in range(SEEDS):
                x, lab = make_series(kind, mag, ratio, seed)
                reg = regions(lab)
                for variant in ("global", "per_window"):
                    s = score(x, variant)
                    rec: dict[str, Any] = {"kind": kind, "mag": mag, "ratio": ratio, "seed": seed,
                                           "variant": variant}
                    for r in ("boundary", "interior"):
                        if reg[r].any():
                            y = np.r_[np.ones(reg[r].sum()), np.zeros(reg["normal"].sum())]
                            rec[f"auc_{r}"] = float(roc_auc_score(y, np.r_[s[reg[r]], s[reg["normal"]]]))
                    rec["VUS-PR"] = official_metrics(s, lab, 50)["VUS-PR"]
                    out.append(rec)
            LOGGER.info("%s %.2f r=%s done", kind, mag, ratio)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    recs = run()
    path = OUT / "synthetic" / "boundary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(recs, indent=1) + "\n")
    LOGGER.info("wrote %s (%d records)", path, len(recs))


if __name__ == "__main__":
    main()
