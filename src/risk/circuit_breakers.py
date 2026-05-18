"""Circuit breakers — escalating halts driven by realised P&L.

* daily   -2.5%  -> halt new trades for the day
* weekly  -5.0%  -> halt for the week, force review
* peak DD -8.0%  -> full system stop, manual restart required
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class BreakerState(Enum):
    GREEN = "green"
    HALT_DAY = "halt_day"
    HALT_WEEK = "halt_week"
    SYSTEM_STOP = "system_stop"


@dataclass(frozen=True)
class BreakerConfig:
    daily_loss_limit: float = -0.025
    weekly_loss_limit: float = -0.050
    max_drawdown_limit: float = -0.080


@dataclass(frozen=True)
class BreakerStatus:
    state: BreakerState
    daily_pnl: float
    weekly_pnl: float
    drawdown: float
    reason: str

    @property
    def can_trade(self) -> bool:
        return self.state == BreakerState.GREEN


def evaluate(
    daily_returns: pd.Series, cfg: BreakerConfig | None = None
) -> BreakerStatus:
    """Most severe breach wins. Empty history -> GREEN."""
    cfg = cfg or BreakerConfig()
    r = daily_returns.dropna()
    if r.empty:
        return BreakerStatus(BreakerState.GREEN, 0, 0, 0, "no history")

    daily = float(r.iloc[-1])
    weekly = float((1 + r.iloc[-5:]).prod() - 1)
    eq = (1 + r).cumprod()
    dd = float(eq.iloc[-1] / eq.cummax().iloc[-1] - 1)

    if dd <= cfg.max_drawdown_limit:
        st, why = BreakerState.SYSTEM_STOP, (
            f"drawdown {dd:.2%} <= {cfg.max_drawdown_limit:.1%} — manual restart")
    elif weekly <= cfg.weekly_loss_limit:
        st, why = BreakerState.HALT_WEEK, (
            f"weekly {weekly:.2%} <= {cfg.weekly_loss_limit:.1%} — review")
    elif daily <= cfg.daily_loss_limit:
        st, why = BreakerState.HALT_DAY, (
            f"daily {daily:.2%} <= {cfg.daily_loss_limit:.1%} — halt today")
    else:
        st, why = BreakerState.GREEN, "within limits"
    return BreakerStatus(st, daily, weekly, dd, why)
