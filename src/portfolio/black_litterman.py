"""Black-Litterman posterior expected returns.

Market-implied equilibrium returns as the prior, factor scores as views
with confidence, posterior feeds MVO. Taught because no retail content
does it and it's close to what desks actually run.

    pi          = delta * Sigma * w_mkt                 (implied prior)
    mu_BL = [ (tau*Sigma)^-1 + Pᵀ Omega^-1 P ]^-1
            [ (tau*Sigma)^-1 pi + Pᵀ Omega^-1 Q ]
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def implied_returns(
    cov: pd.DataFrame, w_market: pd.Series, delta: float = 2.5
) -> pd.Series:
    w = w_market.reindex(cov.index).fillna(0.0).values
    pi = delta * cov.values @ w
    return pd.Series(pi, index=cov.index, name="pi")


def black_litterman(
    cov: pd.DataFrame,
    w_market: pd.Series,
    views: pd.Series,
    *,
    tau: float = 0.05,
    delta: float = 2.5,
    view_confidence: float = 0.50,
) -> pd.Series:
    """Absolute view per asset (``views``, e.g. scaled factor scores).

    P = I (one view per asset). Omega is diagonal; lower
    ``view_confidence`` -> larger Omega -> posterior stays nearer the
    market prior.
    """
    assets = list(cov.index)
    Sigma = cov.values
    pi = implied_returns(cov, w_market, delta).values
    q = views.reindex(assets).fillna(0.0).values

    n = len(assets)
    P = np.eye(n)
    tS_inv = np.linalg.inv(tau * Sigma)
    # Omega scaled by view (un)confidence; +eps keeps it invertible.
    omega_diag = (1.0 / max(view_confidence, 1e-6)) * np.diag(tau * Sigma) + 1e-10
    Omega_inv = np.diag(1.0 / omega_diag)

    A = tS_inv + P.T @ Omega_inv @ P
    b = tS_inv @ pi + P.T @ Omega_inv @ q
    mu_bl = np.linalg.solve(A, b)
    return pd.Series(mu_bl, index=assets, name="mu_bl")
