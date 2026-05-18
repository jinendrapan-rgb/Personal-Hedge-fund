"""Cross-sectional utilities: winsorise, z-score, sector-neutralise, rank.

Sign convention across the whole factor engine: **higher score = more
attractive (long side)**. Each raw factor is built to obey this, then
sector-neutralised so the composite expresses stock selection, not sector
bets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def winsorise(s: pd.Series, limit: float = 0.02) -> pd.Series:
    """Clip both tails to the ``limit`` / ``1 - limit`` quantiles."""
    if s.dropna().empty:
        return s
    lo, hi = s.quantile(limit), s.quantile(1 - limit)
    return s.clip(lower=lo, upper=hi)


def zscore(s: pd.Series) -> pd.Series:
    """Standardise to mean 0, std 1 (NaNs preserved). Zero-variance -> 0."""
    x = s.astype(float)
    mu = x.mean()
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index).where(x.notna())
    return (x - mu) / sd


def sector_neutralise(values: pd.Series, sectors: pd.Series) -> pd.Series:
    """Demean within sector, then global z-score.

    Demeaning within sector forces each sector's mean to ~0; the trailing
    z-score only rescales, so per-sector means stay ~0. That is the
    property the spec's test checks.
    """
    v = values.astype(float)
    demeaned = v - v.groupby(sectors).transform("mean")
    return zscore(demeaned)


def rank_within_sector(values: pd.Series, sectors: pd.Series) -> pd.Series:
    """Percentile rank within sector, centred to [-0.5, +0.5].

    Used to combine value sub-metrics on a common scale before z-scoring.
    Higher input -> higher rank (feed already-oriented signals).
    """
    return values.groupby(sectors).rank(pct=True) - 0.5


def combine_zscores(frame: pd.DataFrame) -> pd.Series:
    """Equal-weight mean across columns, ignoring NaNs per row.

    A row with *no* observations -> NaN (don't fabricate a 0 signal).
    """
    return frame.mean(axis=1, skipna=True)
