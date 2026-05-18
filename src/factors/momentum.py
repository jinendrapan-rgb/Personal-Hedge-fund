"""Momentum — Jegadeesh & Titman (1993), 12-1, risk-adjusted.

Cumulative return from t-12m to t-1m (skip the most recent month: it
mean-reverts), divided by trailing 12m volatility. Returns a raw per-ticker
Series; sector-neutralisation/z-scoring happens in the engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_M = 21  # trading days per month (approx)


def momentum_12_1(close: pd.DataFrame, as_of: str | pd.Timestamp) -> pd.Series:
    """Risk-adjusted 12-1 momentum per ticker.

    ``close``: DataFrame indexed by date, one column per ticker. Only rows
    with ``date <= as_of`` are used — no lookahead, by construction.
    """
    asof = pd.Timestamp(as_of).normalize()
    px = close.loc[close.index <= asof].sort_index()
    if len(px) < 12 * _M + 1:
        return pd.Series(np.nan, index=close.columns, name="momentum")

    p_skip = px.iloc[-1 - _M]      # price ~1 month ago  (t-1m)
    p_start = px.iloc[-1 - 12 * _M]  # price ~12 months ago (t-12m)
    ret_12_1 = p_skip / p_start - 1.0

    daily = px.pct_change()
    vol = daily.iloc[-12 * _M:].std(ddof=0) * np.sqrt(252.0)

    score = ret_12_1 / vol.replace(0.0, np.nan)
    score.name = "momentum"
    return score
