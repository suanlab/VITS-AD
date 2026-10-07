from __future__ import annotations

import numpy as np
import pytest
from sklearn.covariance import LedoitWolf

from src.scoring.woodbury_mahalanobis import WoodburyLedoitWolfScorer


def _dense_scores(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    lw = LedoitWolf().fit(train)
    diff = test - train.mean(axis=0)
    return np.einsum("ij,jk,ik->i", diff, lw.precision_, diff)


@pytest.mark.parametrize(("n", "d"), [(300, 40), (60, 400), (200, 200)])
def test_matches_dense_ledoit_wolf(n: int, d: int) -> None:
    rng = np.random.default_rng(0)
    train = rng.standard_normal((n, d)) @ rng.standard_normal((d, d)) * 0.1 + rng.standard_normal(d)
    test = rng.standard_normal((50, d)) * 1.5
    fast = WoodburyLedoitWolfScorer().fit(train).score(test)
    np.testing.assert_allclose(fast, _dense_scores(train, test), rtol=1e-6, atol=1e-8)


def test_shrinkage_matches_sklearn() -> None:
    rng = np.random.default_rng(1)
    train = rng.standard_normal((80, 120))
    scorer = WoodburyLedoitWolfScorer().fit(train)
    assert scorer.shrinkage_ == pytest.approx(LedoitWolf().fit(train).shrinkage_, rel=1e-10)


def test_requires_fit() -> None:
    with pytest.raises(RuntimeError):
        WoodburyLedoitWolfScorer().score(np.zeros((2, 3)))
