"""Behavioural checks of a TSB-AD code state's preprocessing, on small synthetic inputs.

The audit in artifacts/normalization_audit/ found four kinds of preprocessing error by reading code.
These checks detect them by running the code, so a benchmark can run them on every commit:

  non_constant       a non-degenerate univariate series with a level-shift anomaly must not get a
                     constant score (catches normalizations that are degenerate on one channel)
  fit_score_match    for detectors that score training data like any other data (not the
                     neighbour-based ones, whose fit-time scores exclude each point itself), the
                     scores computed while fitting must equal decision_function on the same data,
                     after aligning window scores with their padded point positions (catches
                     transforms applied in one path only)
  deep_input_norm    a dataset class asked to normalize must hand normalized windows to the model
                     (catches a normalized copy that is never used)
  channel_shift      a short level shift (20 steps, fewer than the detectors' neighbourhoods)
                     common to both channels of a 2-channel series must be detectable (ROC-AUC
                     >= 0.8; catches normalizations that erase amplitude)
  affine_order       informational: Kendall tau between scores of x and 3x + 5 (detectors are not
                     all meant to be invariant; HBOS, for one, is not)
  fingerprint        hash of the rounded scores, compared across code states by --summarize

Each check passes, fails or errors per detector; nothing here uses benchmark data or labels.

    PYTHONPATH=<tsbad_src> python -m scripts.check_preprocessing --tsbad-src <tsbad_src> --tag A
    python -m scripts.check_preprocessing --summarize A B C

Output: artifacts/normalization_audit/checks_<tag>.json and checks_summary.md
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np

LOGGER = logging.getLogger(__name__)
OUT = Path("artifacts/normalization_audit")
UNI = ("IForest", "LOF", "HBOS", "Sub_IForest", "Sub_LOF", "Sub_KNN", "Sub_HBOS", "Sub_OCSVM",
       "Sub_MCD", "Sub_PCA")
SEMI = {"Sub_OCSVM", "Sub_MCD", "OCSVM", "MCD"}
CLASSES = ("KNN", "LOF", "HBOS", "OCSVM", "MCD", "PCA", "IForest")


def _series(n: int = 1500, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 50) + 0.1 * rng.standard_normal(n)
    lab = np.zeros(n, dtype=int)
    lab[1100:1160] = 1
    x[1100:1160] += 3.0
    return x.reshape(-1, 1), lab


def _auc(score: np.ndarray, lab: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(lab, score))


def _check_is_fitted(estimator: Any, attributes: list[str] | None = None, **_: Any) -> None:
    missing = [a for a in (attributes or []) if not hasattr(estimator, a)]
    if missing:
        raise ValueError(f"{type(estimator).__name__} is not fitted: missing {missing}")


def _call(wrapper: Any, name: str, x: np.ndarray, hp: dict[str, Any]) -> np.ndarray:
    run = getattr(wrapper, f"run_{name}")
    params = set(inspect.signature(run).parameters)
    kw = {k: v for k, v in hp.items() if k in params}
    if name in SEMI:
        return np.asarray(run(x[:800], x, **kw), dtype=float).ravel()
    return np.asarray(run(x, **kw), dtype=float).ravel()


def _guard(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return fn()
    except Exception as exc:  # a crash is itself a finding
        return {"status": "error", "detail": f"{type(exc).__name__}: {str(exc)[:160]}"}


def run_checks(tsbad_src: Path) -> dict[str, Any]:
    import TSB_AD

    origin = Path(TSB_AD.__file__).resolve()
    if tsbad_src.resolve() not in origin.parents:
        raise RuntimeError(f"TSB_AD imported from {origin}, not from {tsbad_src}; set PYTHONPATH")
    for m in CLASSES:
        mod = importlib.import_module(f"TSB_AD.models.{m}")
        if hasattr(mod, "check_is_fitted"):
            mod.check_is_fitted = _check_is_fitted
    wrapper = importlib.import_module("TSB_AD.model_wrapper")
    hp_mod = importlib.import_module("TSB_AD.HP_list")
    uni_hp, multi_hp = hp_mod.Optimal_Uni_algo_HP_dict, hp_mod.Optimal_Multi_algo_HP_dict
    x, lab = _series()
    out: dict[str, Any] = {"non_constant": {}, "affine_order": {}, "fingerprint": {},
                           "fit_score_match": {}, "deep_input_norm": {}, "channel_shift": {}}

    for name in UNI:
        def one(name: str = name) -> dict[str, Any]:
            s = _call(wrapper, name, x, uni_hp.get(name, {}))
            out["fingerprint"][name] = hashlib.sha256(np.round(s, 6).tobytes()).hexdigest()[:16]
            ok = bool(np.std(s) > 1e-12)
            return {"status": "pass" if ok else "fail", "std": float(np.std(s)), "auc": _auc(s, lab)}

        out["non_constant"][name] = _guard(one)

        def aff(name: str = name) -> dict[str, Any]:
            from scipy.stats import kendalltau

            a = _call(wrapper, name, x, uni_hp.get(name, {}))
            b = _call(wrapper, name, 3 * x + 5, uni_hp.get(name, {}))
            tau = float(kendalltau(a, b).statistic) if np.std(a) > 0 and np.std(b) > 0 else float("nan")
            return {"status": "info", "kendall_tau": tau}

        out["affine_order"][name] = _guard(aff)

    for cls_name in ("OCSVM", "MCD", "HBOS", "PCA"):
        def fsm(cls_name: str = cls_name) -> dict[str, Any]:
            cls = getattr(importlib.import_module(f"TSB_AD.models.{cls_name}"), cls_name)
            params = set(inspect.signature(cls).parameters)
            kw = {"slidingWindow": 50} if "slidingWindow" in params else {}
            model = cls(**kw)
            model.fit(x[:800])
            fit_scores = np.asarray(model.decision_scores_, dtype=float).ravel()
            again = np.asarray(model.decision_function(x[:800]), dtype=float).ravel()
            off = (again.size - fit_scores.size + 1) // 2 if again.size > fit_scores.size else 0
            aligned = again[off:off + fit_scores.size]
            m = min(fit_scores.size, aligned.size)
            diff = float(np.max(np.abs(fit_scores[:m] - aligned[:m])))
            scale = float(np.max(np.abs(fit_scores[:m]))) or 1.0
            return {"status": "pass" if diff <= 1e-6 * max(1.0, scale) else "fail", "max_abs_diff": diff}

        out["fit_score_match"][cls_name] = _guard(fsm)

    def deep() -> dict[str, Any]:
        ds_mod = importlib.import_module("TSB_AD.utils.dataset")
        data = 100.0 + 5.0 * np.random.default_rng(1).standard_normal((600, 1))
        res = {}
        for cls_name in ("ReconstructDataset", "ForecastDataset"):
            cls = getattr(ds_mod, cls_name)
            ds = cls(data, 20, normalize=True) if cls_name == "ReconstructDataset" else cls(data, 20, 1, normalize=True)
            mean = float(np.asarray(ds.samples, dtype=float).mean())
            res[cls_name] = {"status": "pass" if abs(mean) < 1.0 else "fail", "window_mean": mean}
        return res

    try:
        out["deep_input_norm"] = deep()
    except Exception as exc:
        out["deep_input_norm"] = {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}

    rng = np.random.default_rng(2)
    t = np.arange(1500)
    x2 = np.stack([np.sin(2 * np.pi * t / 50), np.cos(2 * np.pi * t / 37)], axis=1)
    x2 += 0.1 * rng.standard_normal(x2.shape)
    x2[1100:1120] += 3.0
    lab2 = np.zeros(1500, dtype=int)
    lab2[1100:1120] = 1
    for name in ("KNN", "LOF", "HBOS", "IForest", "PCA"):
        def chan(name: str = name) -> dict[str, Any]:
            s = _call(wrapper, name, x2, multi_hp.get(name, {}))
            auc = _auc(s, lab2)
            return {"status": "pass" if auc >= 0.8 else "fail", "auc": auc}

        out["channel_shift"][name] = _guard(chan)
    return out


def summarize(tags: list[str]) -> str:
    res = {t: json.loads((OUT / f"checks_{t}.json").read_text()) for t in tags}
    lines = ["| check | detector | " + " | ".join(tags) + " |", "|---|---|" + "---|" * len(tags)]
    for check in ("non_constant", "fit_score_match", "channel_shift", "affine_order"):
        names = sorted({n for t in tags for n in res[t]["checks"][check]})
        for n in names:
            cells = []
            for t in tags:
                r = res[t]["checks"][check].get(n, {})
                st = r.get("status", "-")
                if check == "affine_order" and "kendall_tau" in r:
                    st = f"tau {r['kendall_tau']:.3f}"
                cells.append(st)
            lines.append(f"| {check} | {n} | " + " | ".join(cells) + " |")
    for n in ("ReconstructDataset", "ForecastDataset"):
        cells = [res[t]["checks"]["deep_input_norm"].get(n, {}).get("status", "-") for t in tags]
        lines.append(f"| deep_input_norm | {n} | " + " | ".join(cells) + " |")
    names = sorted({n for t in tags for n in res[t]["checks"]["fingerprint"]})
    for n in names:
        fp = [res[t]["checks"]["fingerprint"].get(n, "-") for t in tags]
        cells = ["same as previous" if i and fp[i] == fp[i - 1] else fp[i] for i in range(len(fp))]
        lines.append(f"| fingerprint | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsbad-src", type=Path)
    ap.add_argument("--tag")
    ap.add_argument("--summarize", nargs="+")
    args = ap.parse_args()
    if args.summarize:
        md = summarize(args.summarize)
        (OUT / "checks_summary.md").write_text(md)
        print(md)
        return
    checks = run_checks(args.tsbad_src)
    (OUT / f"checks_{args.tag}.json").write_text(
        json.dumps({"tsbad_src": args.tsbad_src.name, "checks": checks}, indent=1) + "\n")
    LOGGER.info("wrote %s", OUT / f"checks_{args.tag}.json")


if __name__ == "__main__":
    main()
