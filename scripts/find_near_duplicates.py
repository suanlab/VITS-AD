"""Find TSB-AD-U series that share a base signal with another series.

Two series are near-duplicates at threshold r if they have the same length and the Pearson
correlation of their raw values is above r in absolute value. Only raw values are read: no labels,
detector scores or results. The UCR archive, for one, contains several anomalies injected into the
same recording, and TSB-AD keeps such variants in different splits.

Outputs artifacts/official_protocol/near_duplicates.json:
  pairs           every pair with |r| > 0.9, with r and the split of both series
  split_of        the split of each series (eva, tuning, rest)
  rest_with_partner[r]  rest series with a partner in eva or tuning above r (r = 0.9, 0.99, 0.999)
  clusters_eva_099      connected components of eva series at |r| > 0.99 (for cluster bootstraps)

    python -m scripts.find_near_duplicates
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_detector_normalization import DATA, OFFICIAL, SPLITS, TUNING_SPLITS

LOGGER = logging.getLogger(__name__)
OUT = Path("artifacts/official_protocol/near_duplicates.json")
THRESHOLDS = (0.9, 0.99, 0.999)


def _components(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    parent = {v: v for v in nodes}

    def find(v: str) -> str:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for a, b in edges:
        parent[find(a)] = find(b)
    groups: dict[str, list[str]] = {}
    for v in nodes:
        groups.setdefault(find(v), []).append(v)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: (-len(g), g[0]))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    subset = "TSB-AD-U"
    eva = set(pd.read_csv(OFFICIAL / SPLITS[subset])["file_name"])
    tun = set(pd.read_csv(OFFICIAL / TUNING_SPLITS[subset])["file_name"])
    files = sorted(f.name for f in (DATA / subset).glob("*.csv"))
    split_of = {f: "eva" if f in eva else "tuning" if f in tun else "rest" for f in files}
    zs: dict[str, np.ndarray] = {}
    by_len: dict[int, list[str]] = {}
    for f in files:
        x = pd.read_csv(DATA / subset / f).iloc[:, 0].to_numpy(dtype=float)
        by_len.setdefault(len(x), []).append(f)
        if x.std() > 0:
            zs[f] = (x - x.mean()) / x.std()
    pairs = []
    for group in by_len.values():
        group = [f for f in group if f in zs]
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                r = float(zs[a] @ zs[b] / len(zs[a]))
                if abs(r) > THRESHOLDS[0]:
                    pairs.append({"a": a, "b": b, "r": round(r, 6),
                                  "split_a": split_of[a], "split_b": split_of[b]})
    rest_with_partner = {}
    for th in THRESHOLDS:
        hit = set()
        for p in pairs:
            if abs(p["r"]) <= th:
                continue
            for x, y in ((p["a"], p["b"]), (p["b"], p["a"])):
                if split_of[x] == "rest" and split_of[y] != "rest":
                    hit.add(x)
        rest_with_partner[str(th)] = sorted(hit)
        LOGGER.info("|r| > %s: %d rest series with an eva/tuning partner", th, len(hit))
    eva_edges = [(p["a"], p["b"]) for p in pairs
                 if abs(p["r"]) > 0.99 and p["split_a"] == p["split_b"] == "eva"]
    clusters = _components(sorted(eva), eva_edges)
    LOGGER.info("eva: %d series in %d clusters at |r| > 0.99", len(eva), len(clusters))
    OUT.write_text(json.dumps({"pairs": pairs, "split_of": split_of,
                               "rest_with_partner": rest_with_partner,
                               "clusters_eva_099": clusters}, indent=1) + "\n")
    LOGGER.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
