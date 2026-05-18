"""Long/short decile portfolio construction.

Top decile long, bottom decile short, equal-weighted within each leg.
1x long + 1x short = 2x gross, ~0 net. Pure function of a score Series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_long_short(scores: pd.Series, decile: float = 0.10) -> pd.Series:
    """Target weights from cross-sectional ``scores`` (higher = long).

    Returns a weight Series over the same index: longs sum to +1, shorts
    sum to -1, everything else 0. Names with NaN score are excluded.
    """
    s = scores.dropna()
    n = len(s)
    if n < 2:
        return pd.Series(0.0, index=scores.index)

    k = max(1, int(np.floor(n * decile)))
    ranked = s.sort_values(ascending=False)
    longs = ranked.index[:k]
    shorts = ranked.index[-k:]

    w = pd.Series(0.0, index=scores.index)
    w.loc[longs] = 1.0 / k
    w.loc[shorts] = -1.0 / k
    return w


def gross(w: pd.Series) -> float:
    return float(w.abs().sum())


def net(w: pd.Series) -> float:
    return float(w.sum())
