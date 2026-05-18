"""Low Volatility — Ang, Hodrick, Xing, Zhang (2006).

Trailing 252-day realised volatility, **inverted** so low vol = high
score (consistent with the engine-wide "higher = more attractive" rule).
The factor most retail builds skip; one of the most persistent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def low_volatility(close: pd.DataFrame, as_of: str | pd.Timestamp) -> pd.Series:
    """Negative trailing 252d realised vol per ticker (no lookahead)."""
    asof = pd.Timestamp(as_of).normalize()
    px = close.loc[close.index <= asof].sort_index()
    if len(px) < 252:
        return pd.Series(np.nan, index=close.columns, name="low_vol")

    daily = px.pct_change().iloc[-252:]
    vol = daily.std(ddof=0) * np.sqrt(252.0)
    score = -vol  # invert: lower realised vol -> higher score
    score.name = "low_vol"
    return score
