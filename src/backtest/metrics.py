"""Tear-sheet metrics. Pure functions on daily-return Series."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _clean(r: pd.Series) -> pd.Series:
    return r.replace([np.inf, -np.inf], np.nan).dropna()


def ann_return(r: pd.Series) -> float:
    r = _clean(r)
    if r.empty:
        return 0.0
    return float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1)


def ann_vol(r: pd.Series) -> float:
    r = _clean(r)
    return float(r.std(ddof=0) * np.sqrt(TRADING_DAYS))


def sharpe(r: pd.Series, rf_daily: float = 0.0) -> float:
    r = _clean(r) - rf_daily
    sd = r.std(ddof=0)
    if r.empty or sd == 0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS))


def sortino(r: pd.Series, rf_daily: float = 0.0) -> float:
    r = _clean(r) - rf_daily
    downside = r[r < 0]
    dd = downside.std(ddof=0)
    if r.empty or dd == 0:
        return 0.0
    return float(r.mean() / dd * np.sqrt(TRADING_DAYS))


def max_drawdown(r: pd.Series) -> tuple[float, int]:
    """Return (max drawdown as a negative fraction, duration in days)."""
    r = _clean(r)
    if r.empty:
        return 0.0, 0
    eq = (1 + r).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    trough = dd.idxmin()
    mdd = float(dd.min())
    peak_idx = eq.loc[:trough].idxmax()
    duration = int((eq.index.get_loc(trough) - eq.index.get_loc(peak_idx)))
    return mdd, duration


def calmar(r: pd.Series) -> float:
    mdd, _ = max_drawdown(r)
    if mdd == 0:
        return 0.0
    return float(ann_return(r) / abs(mdd))


def information_ratio(r: pd.Series, bench: pd.Series) -> float:
    a, b = _clean(r).align(_clean(bench), join="inner")
    excess = a - b
    sd = excess.std(ddof=0)
    if excess.empty or sd == 0:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def hit_rate(r: pd.Series) -> float:
    r = _clean(r)
    return float((r > 0).mean()) if not r.empty else 0.0


def win_loss(r: pd.Series) -> tuple[float, float]:
    r = _clean(r)
    wins, losses = r[r > 0], r[r < 0]
    return (
        float(wins.mean()) if not wins.empty else 0.0,
        float(losses.mean()) if not losses.empty else 0.0,
    )


def turnover(positions: pd.DataFrame) -> float:
    """Mean per-rebalance one-way turnover from a (rebalance x ticker)
    target-weight frame."""
    if len(positions) < 2:
        return 0.0
    return float(positions.fillna(0.0).diff().abs().sum(axis=1).iloc[1:].mean())


def summary(
    total: pd.Series,
    long_leg: pd.Series,
    short_leg: pd.Series,
    bench: pd.Series,
    positions: pd.DataFrame,
) -> dict[str, float]:
    mdd, dur = max_drawdown(total)
    aw, al = win_loss(total)
    return {
        "ann_return": round(ann_return(total), 4),
        "ann_vol": round(ann_vol(total), 4),
        "sharpe": round(sharpe(total), 3),
        "sortino": round(sortino(total), 3),
        "calmar": round(calmar(total), 3),
        "max_drawdown": round(mdd, 4),
        "max_dd_duration_days": dur,
        "information_ratio_vs_bench": round(information_ratio(total, bench), 3),
        "long_only_sharpe": round(sharpe(long_leg), 3),
        "short_only_sharpe": round(sharpe(short_leg), 3),
        "hit_rate": round(hit_rate(total), 4),
        "avg_win": round(aw, 5),
        "avg_loss": round(al, 5),
        "avg_monthly_turnover": round(turnover(positions), 4),
        "ann_return_vs_bench": round(ann_return(total) - ann_return(bench), 4),
    }
