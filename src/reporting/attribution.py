"""P&L attribution by OLS on explanatory return streams.

Regress portfolio daily returns on a set of factors (market, sector
baskets, the 5 style factors — whatever the caller supplies). Each
factor's contribution = beta * mean(factor) annualised; the intercept is
alpha. Pure and deterministic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def attribute(
    port: pd.Series, factors: pd.DataFrame
) -> dict[str, float]:
    """``port``: daily portfolio returns. ``factors``: aligned daily
    return streams (columns = factor names). Returns annualised
    contribution per factor plus ``alpha`` and ``r_squared``."""
    df = pd.concat([port.rename("y"), factors], axis=1).dropna()
    if len(df) < len(factors.columns) + 2:
        raise ValueError("not enough overlapping observations to attribute")

    y = df["y"].values
    X = df[factors.columns].values
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)

    alpha_daily = beta[0]
    loadings = beta[1:]
    contrib = {
        col: float(loadings[i] * X[:, i].mean() * TRADING_DAYS)
        for i, col in enumerate(factors.columns)
    }
    contrib["alpha"] = float(alpha_daily * TRADING_DAYS)

    resid = y - X1 @ beta
    ss_tot = ((y - y.mean()) ** 2).sum()
    contrib["r_squared"] = float(
        1.0 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else 0.0
    )
    return contrib
