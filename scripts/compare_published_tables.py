"""How TSB-AD's published per-series VUS-PR tables changed between commits.

Downloads the univariate and multivariate tables at the upload commit (35ba9c3b, 2024-11-29), at the
commit that regenerated part of the multivariate table (82484f04, 2024-12-05, two days after the
model fix ac710895), and at the pin used by the paper (303dffbf), and records, per detector column
present at the upload, how many series changed and by how much. Values are compared, not bytes:
later commits reformat the files.

Output: artifacts/official_protocol/published_table_history.json

    python -m scripts.compare_published_tables
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)
OUT = Path("artifacts/official_protocol/published_table_history.json")
COMMITS = {"upload": "35ba9c3b", "regenerated_multi": "82484f04", "pinned": "303dffbf"}
FILES = {"uni": "benchmark_exp/benchmark_eval_results/uni_mergedTable_VUS-PR.csv",
         "multi": "benchmark_exp/benchmark_eval_results/multi_mergedTable_VUS-PR.csv"}


def _fetch(commit: str, path: str) -> bytes:
    url = f"https://raw.githubusercontent.com/TheDatumOrg/TSB-AD/{commit}/{path}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out: dict[str, Any] = {"commits": COMMITS, "sha256": {}, "changes": {}}
    for kind, path in FILES.items():
        tables = {}
        for name, commit in COMMITS.items():
            raw = _fetch(commit, path)
            out["sha256"][f"{kind}@{commit}"] = hashlib.sha256(raw).hexdigest()
            tables[name] = pd.read_csv(io.BytesIO(raw)).set_index("file")
        base = tables["upload"]
        for name in ("regenerated_multi", "pinned"):
            other = tables[name]
            idx = base.index.intersection(other.index)
            changed = {}
            for col in base.columns:
                if col not in other.columns or base[col].dtype == object:
                    continue
                d = (base.loc[idx, col] - other.loc[idx, col]).abs()
                if d.max() > 1e-9:
                    changed[col] = {"series_changed": int((d > 1e-9).sum()), "n": int(len(idx)),
                                    "max_abs_diff": float(d.max()),
                                    "mean_upload": float(base.loc[idx, col].mean()),
                                    f"mean_{name}": float(other.loc[idx, col].mean())}
            out["changes"][f"{kind}: upload -> {name}"] = changed
            LOGGER.info("%s upload -> %s: %s", kind, name, sorted(changed) or "no value changes")
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    LOGGER.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
