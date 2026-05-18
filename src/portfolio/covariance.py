"""Ledoit-Wolf shrinkage covariance.

A 500-asset sample covariance from ~1-3y of daily returns is severely
ill-conditioned and makes the optimiser chase noise. Ledoit-Wolf shrinks
the sample matrix toward a structured target, which is well-conditioned
and the standard institutional choice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252


def ledoit_wolf_cov(returns: pd.DataFrame, annualise: bool = True) -> pd.DataFrame:
    """Shrunk covariance from a (time x asset) daily-returns frame.

    Columns with any NaN are dropped (an asset without a full history
    can't enter the covariance cleanly). Returns a labelled, symmetric
    DataFrame; annualised by default.
    """
    r = returns.dropna(axis=1, how="any")
    if r.shape[1] < 2 or r.shape[0] < 2:
        raise ValueError("need >=2 assets and >=2 observations for covariance")
    lw = LedoitWolf().fit(r.values)
    cov = lw.covariance_
    if annualise:
        cov = cov * TRADING_DAYS
    cov = (cov + cov.T) / 2.0  # guard tiny asymmetry from float error
    return pd.DataFrame(cov, index=r.columns, columns=r.columns)
