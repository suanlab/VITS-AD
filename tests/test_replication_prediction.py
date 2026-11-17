"""Tests for the replication-prediction rule (scripts/replication_prediction.py) on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.replication_prediction import decision, fit, instances, predict, score


def _make_sources(effects: dict[str, float], n: int, noise: float, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    vals, src, idx = [], [], []
    for s, e in effects.items():
        for i in range(n):
            vals.append(e + noise * rng.standard_normal())
            src.append(s)
            idx.append(f"{s}_{i}")
    return pd.Series(vals, index=idx), pd.Series(src, index=idx)


def test_fit_recovers_between_source_variance() -> None:
    eff = {f"s{k}": v for k, v in enumerate(np.linspace(-0.3, 0.5, 10))}
    d, s = _make_sources(eff, 30, 0.05)
    f = fit(d, s)
    assert f["k_sources"] == 10
    assert f["tau2"] == pytest.approx(np.var(list(eff.values()), ddof=1), rel=0.3)
    homog, s2 = _make_sources({f"s{k}": 0.1 for k in range(10)}, 30, 0.05)
    assert fit(homog, s2)["tau2"] < 1e-3


def test_heterogeneity_lowers_the_new_source_probability() -> None:
    het, s = _make_sources({"a": 0.4, "b": 0.3, "c": -0.2, "d": -0.1, "e": 0.3}, 40, 0.05)
    p = predict(fit(het, s), 40, 1.0)
    assert p["p_new"] < p["p_iid"] <= p["p_sig"]


def test_instances_and_scores() -> None:
    rng = np.random.default_rng(1)
    rows = {}
    for s in [f"s{k}" for k in range(8)]:
        shift = rng.normal(0, 0.2)
        for i in range(20):
            rows[f"{s}_{i}"] = {"x": 0.5 + 0.05 * rng.standard_normal(), "y": 0.5 + shift + 0.05 * rng.standard_normal()}
    t = pd.DataFrame(rows).T
    src = pd.Series([i.split("_")[0] for i in t.index], index=t.index)
    disc = src.isin(["s0", "s1", "s2", "s3"])
    o1, o2 = instances(t[disc], t[~disc], src[disc], src[~disc], [("x", "y")])
    assert len(o2) == 1 and len(o1) == 4
    sc = score(o1)
    assert sc["n"] == 4 and set(sc) >= {"p_new", "p_iid", "p_sig"}


def test_decision_rule() -> None:
    good = {"O1": {"p_new": {"brier": 0.10, "auc": 0.80}, "p_iid": {"brier": 0.15, "auc": 0.70}}}
    flat = {"O1": {"p_new": {"brier": 0.15, "auc": 0.70}, "p_iid": {"brier": 0.15, "auc": 0.70}}}
    assert decision({"a": good, "b": good, "c": flat})["decision"] == "supported"
    assert decision({"a": flat, "b": flat, "c": flat})["decision"] == "refuted"
    assert decision({"a": good, "b": flat, "c": flat})["decision"] == "inconclusive"


def test_exploratory_variants_separate_centre_and_variance() -> None:
    from scripts.replication_prediction_explore import variants

    het, s = _make_sources({"a": 0.4, "b": 0.3, "c": -0.2, "d": -0.1, "e": 0.3}, 40, 0.05)
    f = fit(het, s)
    v = variants(f, 40)
    assert v["p_new"] == pytest.approx(predict(f, 40, 1.0)["p_new"])
    # dropping tau^2 from the variance sharpens the forecast; a t on k-2 df widens it
    assert abs(v["p_mu_novar"] - 0.5) >= abs(v["p_new"] - 0.5)
    assert abs(v["p_new_t"] - 0.5) <= abs(v["p_new"] - 0.5)
    assert v["p_d_revar"] <= v["p_iid"]


def test_reml_recovers_between_source_variance() -> None:
    from scripts.replication_prediction_explore import reml

    eff = {f"s{k}": v for k, v in enumerate(np.linspace(-0.3, 0.5, 10))}
    d, s = _make_sources(eff, 30, 0.05)
    tau2, mu, se_mu = reml(d, s)
    assert tau2 == pytest.approx(np.var(list(eff.values()), ddof=1), rel=0.3)
    assert mu == pytest.approx(np.mean(list(eff.values())), abs=0.05) and se_mu > 0
