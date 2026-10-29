"""Ledoit-Wolf Mahalanobis scoring without forming the d x d precision matrix.

The Ledoit-Wolf estimate is a rank-deficient sample covariance plus a scaled identity,

    Sigma = (1 - delta) * Xc^T Xc / n + delta * mu * I,    mu = trace(Xc^T Xc / n) / d,

so the Woodbury identity gives Sigma^{-1} through an n x n solve instead of a d x d inverse.
The scores equal sklearn's ``LedoitWolf`` (same shrinkage, same covariance) as long as the floor
``eps`` is inactive, i.e. delta * mu > eps and (1 - delta) > 0; for inputs with near-zero variance
the floor changes the scores. The cost is O(n^2 d + n^3) rather than O(n d^2 + d^3), which removes the need to cap the flattened
feature dimension W * D when the number of training windows n is moderate.
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt
from sklearn.covariance import ledoit_wolf_shrinkage

FloatArray = npt.NDArray[np.float64]


class WoodburyLedoitWolfScorer:
    """Squared Mahalanobis distance under the Ledoit-Wolf covariance, via Woodbury.

    Args:
        eps: Floor for the identity term when the shrinkage target vanishes.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self.mean_: FloatArray | None = None
        self._xc: FloatArray | None = None
        self._nu = float("nan")
        self._inner_inv: FloatArray | None = None
        self.shrinkage_ = float("nan")

    def fit(self, features: FloatArray) -> "WoodburyLedoitWolfScorer":
        """Fit on training rows.

        Args:
            features: Training matrix of shape (n, d).

        Returns:
            The fitted scorer.

        Raises:
            ValueError: If fewer than two rows are given.
        """
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] < 2:
            raise ValueError(f"need a 2-D array with at least 2 rows, got {x.shape}")
        n, d = x.shape
        self.mean_ = x.mean(axis=0)
        xc = x - self.mean_
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            delta = float(ledoit_wolf_shrinkage(xc, assume_centered=True))
        mu = float(np.einsum("ij,ij->", xc, xc)) / (n * d)
        a = (1.0 - delta) / n
        nu = max(delta * mu, self.eps)
        gram = xc @ xc.T
        inner = (np.eye(n) / a if a > 0 else np.eye(n) / self.eps) + gram / nu
        self._xc = xc
        self._nu = nu
        self._inner_inv = np.linalg.inv(inner)
        self.shrinkage_ = delta
        return self

    def score(self, features: FloatArray, batch: int = 4096) -> FloatArray:
        """Squared Mahalanobis distance of each row from the training mean.

        Args:
            features: Matrix of shape (m, d).
            batch: Rows scored per block, to bound memory.

        Returns:
            Scores of shape (m,).

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if self._xc is None or self.mean_ is None or self._inner_inv is None:
            raise RuntimeError("fit() must be called first")
        x = np.asarray(features, dtype=np.float64)
        out = np.empty(x.shape[0], dtype=np.float64)
        for lo in range(0, x.shape[0], batch):
            v = x[lo:lo + batch] - self.mean_
            p = v @ self._xc.T
            quad = np.einsum("ij,ij->i", p @ self._inner_inv, p)
            out[lo:lo + batch] = np.einsum("ij,ij->i", v, v) / self._nu - quad / self._nu**2
        return out
