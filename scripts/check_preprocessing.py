"""Behavioural checks of a TSB-AD code state's preprocessing, on small synthetic inputs.

The audit in artifacts/normalization_audit/ found four kinds of preprocessing error by reading code.
These checks detect them by running the code, so a benchmark can run them on every commit:

  non_constant       a constant score on the specified synthetic series is a degeneracy diagnostic,
                     not a universal implementation-error verdict
  fit_score_match    for detectors that score training data like any other data (not the
                     neighbour-based ones, whose fit-time scores exclude each point itself), the
                     scores computed while fitting must equal decision_function on the same data,
                     after aligning window scores with their padded point positions (catches
                     transforms applied in one path only)
  deep_input_norm    a dataset class asked to normalize must hand the model windows with mean near
                     0 and standard deviation near 1 (input: mean 100, std 5; catches a normalized
                     copy that is never used; checks location and scale only, not the axis)
  channel_shift      a short level shift (20 steps, fewer than the detectors' neighbourhoods)
                     common to both channels is a capability probe (ROC-AUC >= 0.8); failure
                     indicates task mismatch, not necessarily an implementation error
  batch_context      for detectors fitted on a training part (OCSVM, MCD) and the deep models'
                     dataset classes: the score (or window) of the same data must not change when
                     other data is scored with it. The training part is scored alone and again
                     followed by a copy shifted by +5; the shared positions must agree (catches
                     standardizing with the statistics of the scored batch instead of the fitted
                     ones; detectors that declare whole-series processing are out of its scope)
  affine_order       informational: Kendall tau between scores of x and 3x + 5 (detectors are not
                     all meant to be invariant; HBOS, for one, is not)
  fingerprint        hash of the rounded scores, compared across code states by --summarize

The checks are of different kinds and their failures mean different things:

  contract     fit_score_match, deep_input_norm, batch_context: the code does not do what it declares (a
               violation, whatever the detector is meant to detect)
  degeneracy   non_constant: a diagnostic on this input; a constant score is not always a bug
  capability   channel_shift: a task probe with a threshold we chose (ROC-AUC 0.8); a detector
               that declares invariance to a common shift is expected to fail it, and for others a
               failure is a warning about fitness for the task, not an implementation error
  version      fingerprint: a change between commits, neither good nor bad
  info         affine_order

Controls run with every state, independent of TSB-AD: a toy 1-NN that removes the common shift on
purpose and declares it (capability failure expected, not a violation), the same 1-NN without
that step (passes), and a toy detector that scales its input only when fitting (contract
violation expected) with its consistent counterpart (passes), and a toy detector that standardizes
with the scored batch's statistics (batch-context violation expected) with one that keeps the
fitted statistics (passes).

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
KIND = {"non_constant": "degeneracy", "fit_score_match": "contract", "deep_input_norm": "contract",
        "batch_context": "contract",
        "channel_shift": "capability", "affine_order": "info", "fingerprint": "version"}
# Invariances a detector declares; none of TSB-AD's detectors declares invariance to a shift
# common to all channels, so a failed channel_shift probe is a warning for them.
DECLARED_COMMON_SHIFT_INVARIANT = {"control_shift_invariant_1nn"}


def _two_channel(seed: int = 2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(1500)
    x2 = np.stack([np.sin(2 * np.pi * t / 50), np.cos(2 * np.pi * t / 37)], axis=1)
    x2 += 0.1 * rng.standard_normal(x2.shape)
    x2[1100:1120] += 3.0
    lab2 = np.zeros(1500, dtype=int)
    lab2[1100:1120] = 1
    return x2, lab2


def _knn_point_scores(x: np.ndarray, train: int = 800) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=1).fit(x[:train])
    return nn.kneighbors(x)[0][:, 0]


class _FitOnlyScaling:
    """Control: min-max scales its input when fitting but not when scoring (a contract violation)."""

    def __init__(self, consistent: bool = False) -> None:
        self.consistent = consistent

    def _t(self, x: np.ndarray) -> np.ndarray:
        return (x - self.lo) / (self.hi - self.lo)

    def fit(self, x: np.ndarray) -> "_FitOnlyScaling":
        self.lo, self.hi = x.min(axis=0), x.max(axis=0)
        self.ref = self._t(x)
        self.decision_scores_ = np.abs(self.ref - self.ref.mean(axis=0)).sum(axis=1)
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        z = self._t(x) if self.consistent else x
        return np.abs(z - self.ref.mean(axis=0)).sum(axis=1)


class _BatchStats:
    """Control: z-scores with the statistics of the batch it scores (a batch-context violation)
    or, if ``fitted``, with the statistics stored at fit time."""

    def __init__(self, fitted: bool = False) -> None:
        self.fitted = fitted

    def fit(self, x: np.ndarray) -> "_BatchStats":
        self.mu, self.sd = x.mean(axis=0), x.std(axis=0) + 1e-8
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        mu, sd = (self.mu, self.sd) if self.fitted else (x.mean(axis=0), x.std(axis=0) + 1e-8)
        return np.abs((x - mu) / sd).sum(axis=1)


def _context_diff(score_alone: np.ndarray, score_with: np.ndarray, n: int, margin: int) -> float:
    """Largest difference on positions [margin, n - margin) scored alone and with other data after."""
    a, b = np.asarray(score_alone, dtype=float).ravel(), np.asarray(score_with, dtype=float).ravel()
    sl = slice(margin, n - margin)
    scale = max(1.0, float(np.max(np.abs(a[sl]))))
    return float(np.max(np.abs(a[sl] - b[sl])) / scale)


def run_controls() -> dict[str, Any]:
    x2, lab2 = _two_channel()
    centred = x2 - x2.mean(axis=1, keepdims=True)
    out: dict[str, Any] = {"channel_shift": {}, "fit_score_match": {}, "batch_context": {}}
    for name, data in (("control_shift_invariant_1nn", centred), ("control_plain_1nn", x2)):
        auc = _auc(_knn_point_scores(data), lab2)
        out["channel_shift"][name] = {"status": "pass" if auc >= 0.8 else "fail", "auc": auc}
    x, _ = _series()
    for name, consistent in (("control_fit_only_scaling", False), ("control_consistent_scaling", True)):
        m = _FitOnlyScaling(consistent).fit(x[:800])
        diff = float(np.max(np.abs(m.decision_scores_ - m.decision_function(x[:800]))))
        out["fit_score_match"][name] = {"status": "pass" if diff <= 1e-6 else "fail", "max_abs_diff": diff}
    for name, fitted in (("control_batch_statistics", False), ("control_fitted_statistics", True)):
        m = _BatchStats(fitted).fit(x[:800])
        both = np.concatenate([x[:800], x[:800] + 5.0])
        diff = _context_diff(m.decision_function(x[:800]), m.decision_function(both), 800, 0)
        out["batch_context"][name] = {"status": "pass" if diff <= 1e-6 else "fail", "max_rel_diff": diff}
    return out


def verdict(check: str, name: str, status: str) -> str:
    """Turn a pass/fail into what it means for this kind of check."""
    if status in ("pass", "-", "info") or status.startswith("tau"):
        return status
    if status == "error":
        return "error (crash)"
    kind = KIND[check]
    if kind == "contract":
        return "VIOLATION"
    if kind == "degeneracy":
        return "degenerate"
    if kind == "capability":
        return "fail (declared invariance)" if name in DECLARED_COMMON_SHIFT_INVARIANT else "warning"
    return status


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
                           "fit_score_match": {}, "deep_input_norm": {}, "channel_shift": {}, "batch_context": {}}

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

    for cls_name in ("OCSVM", "MCD"):
        def ctx(cls_name: str = cls_name) -> dict[str, Any]:
            cls = getattr(importlib.import_module(f"TSB_AD.models.{cls_name}"), cls_name)
            params = set(inspect.signature(cls).parameters)
            kw = {"slidingWindow": 50} if "slidingWindow" in params else {}
            model = cls(**kw)
            model.fit(x[:800])
            both = np.concatenate([x[:800], x[:800] + 5.0])
            diff = _context_diff(model.decision_function(x[:800]), model.decision_function(both), 800, 50)
            return {"status": "pass" if diff <= 1e-6 else "fail", "max_rel_diff": diff}

        out["batch_context"][cls_name] = _guard(ctx)

    def deep_ctx() -> dict[str, Any]:
        ds_mod = importlib.import_module("TSB_AD.utils.dataset")
        data = 100.0 + 5.0 * np.random.default_rng(1).standard_normal((600, 1))
        both = np.concatenate([data, data + 50.0])
        res = {}
        for cls_name in ("ReconstructDataset", "ForecastDataset"):
            cls = getattr(ds_mod, cls_name)
            make = (lambda d: cls(d, 20, normalize=True)) if cls_name == "ReconstructDataset" else \
                (lambda d: cls(d, 20, 1, normalize=True))
            a, b = np.asarray(make(data).samples, dtype=float), np.asarray(make(both).samples, dtype=float)
            m = min(len(a), 500)
            diff = float(np.max(np.abs(a[:m] - b[:m])) / max(1.0, float(np.max(np.abs(a[:m])))))
            res[cls_name] = {"status": "pass" if diff <= 1e-6 else "fail", "max_rel_diff": diff}
        return res

    try:
        out["batch_context"].update(deep_ctx())
    except Exception as exc:
        out["batch_context"]["datasets"] = {"status": "error", "detail": f"{type(exc).__name__}: {str(exc)[:160]}"}

    def deep() -> dict[str, Any]:
        ds_mod = importlib.import_module("TSB_AD.utils.dataset")
        data = 100.0 + 5.0 * np.random.default_rng(1).standard_normal((600, 1))
        res = {}
        for cls_name in ("ReconstructDataset", "ForecastDataset"):
            cls = getattr(ds_mod, cls_name)
            ds = cls(data, 20, normalize=True) if cls_name == "ReconstructDataset" else cls(data, 20, 1, normalize=True)
            w = np.asarray(ds.samples, dtype=float)
            mean, std = float(w.mean()), float(w.std())
            ok = abs(mean) < 1.0 and 0.5 < std < 2.0       # input has mean 100 and std 5
            res[cls_name] = {"status": "pass" if ok else "fail", "window_mean": mean, "window_std": std}
        return res

    try:
        out["deep_input_norm"] = deep()
    except Exception as exc:
        out["deep_input_norm"] = {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}

    x2, lab2 = _two_channel()
    for name in ("KNN", "LOF", "HBOS", "IForest", "PCA"):
        def chan(name: str = name) -> dict[str, Any]:
            s = _call(wrapper, name, x2, multi_hp.get(name, {}))
            auc = _auc(s, lab2)
            return {"status": "pass" if auc >= 0.8 else "fail", "auc": auc}

        out["channel_shift"][name] = _guard(chan)
    return out


def summarize(tags: list[str]) -> str:
    res = {t: json.loads((OUT / f"checks_{t}.json").read_text()) for t in tags}
    lines = ["| check (kind) | detector | " + " | ".join(tags) + " |", "|---|---|" + "---|" * len(tags)]
    for check in ("fit_score_match", "batch_context", "non_constant", "channel_shift", "affine_order"):
        names = sorted({n for t in tags for n in res[t]["checks"].get(check, {})})
        for n in names:
            cells = []
            for t in tags:
                r = res[t]["checks"].get(check, {}).get(n, {})
                st = r.get("status", "-")
                if check == "affine_order" and "kendall_tau" in r:
                    st = f"tau {r['kendall_tau']:.3f}"
                cells.append(verdict(check, n, st))
            lines.append(f"| {check} ({KIND[check]}) | {n} | " + " | ".join(cells) + " |")
    for n in ("ReconstructDataset", "ForecastDataset"):
        cells = [verdict("deep_input_norm", n, res[t]["checks"]["deep_input_norm"].get(n, {}).get("status", "-"))
                 for t in tags]
        lines.append(f"| deep_input_norm (contract) | {n} | " + " | ".join(cells) + " |")
    ctrl = res[tags[-1]].get("controls", {})
    for check, rows in ctrl.items():
        for n, r in rows.items():
            lines.append(f"| {check} ({KIND[check]}) | {n} | " + " | ".join([verdict(check, n, r["status"])] * len(tags))
                         + " |")
    names = sorted({n for t in tags for n in res[t]["checks"]["fingerprint"]})
    for n in names:
        fp = [res[t]["checks"]["fingerprint"].get(n, "-") for t in tags]
        cells = ["unchanged" if i and fp[i] == fp[i - 1] else ("changed" if i else fp[i]) for i in range(len(fp))]
        lines.append(f"| fingerprint (version) | {n} | " + " | ".join(cells) + " |")
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
        json.dumps({"tsbad_src": args.tsbad_src.name, "kinds": KIND, "checks": checks,
                    "controls": run_controls()}, indent=1) + "\n")
    LOGGER.info("wrote %s", OUT / f"checks_{args.tag}.json")


if __name__ == "__main__":
    main()
