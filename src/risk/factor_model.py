"""Factor vs specific risk decomposition.

A statistical factor model (top-k PCA of asset returns) splits portfolio
variance into a factor part and a specific (stock-selection) part. The
spec targets 70-85% specific risk: returns should come from picking
names, not from a levered factor bet. >30% factor risk -> flag rebalance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskDecomp:
    total_var: float
    factor_var: float
    specific_var: float

    @property
    def specific_share(self) -> float:
        return self.specific_var / self.total_var if self.total_var > 0 else 0.0

    @property
    def factor_share(self) -> float:
        return 1.0 - self.specific_share

    @property
    def flag_rebalance(self) -> bool:
        return self.factor_share > 0.30


def decompose(
    weights: pd.Series, returns_history: pd.DataFrame, n_factors: int = 5
) -> RiskDecomp:
    """PCA factor model: Sigma ~= B F Bᵀ + D (D = specific diagonal)."""
    r = returns_history.dropna(axis=1, how="any")
    w = weights.reindex(r.columns).fillna(0.0).values
    if r.shape[1] < 2 or r.shape[0] < 3:
        raise ValueError("need >=2 assets and >=3 observations")

    Sigma = np.cov(r.values, rowvar=False)
    k = min(n_factors, Sigma.shape[0] - 1)
    vals, vecs = np.linalg.eigh(Sigma)
    order = np.argsort(vals)[::-1]
    B = vecs[:, order[:k]]                 # factor loadings (top-k PCs)
    F = np.diag(vals[order[:k]])           # factor variances
    factor_cov = B @ F @ B.T
    D = np.diag(np.clip(np.diag(Sigma) - np.diag(factor_cov), 0.0, None))

    total = float(w @ Sigma @ w)
    fvar = float(w @ factor_cov @ w)
    svar = float(w @ D @ w)
    # numerical guard: keep parts non-negative and consistent with total
    fvar = max(fvar, 0.0)
    svar = max(total - fvar, 0.0)
    return RiskDecomp(total_var=total, factor_var=fvar, specific_var=svar)
