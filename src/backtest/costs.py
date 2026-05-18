"""Transaction cost model.

Three components, per the spec:

* spread        : 5 bps round-trip for large caps, 15 bps for small caps
* market impact : square-root model, 10 bps per 1% of ADV traded
* borrow        : 50 bps annualised on short notional (baseline)

All functions are pure and operate on fractional notional (weights), so a
"trade" of 0.03 means trading 3% of the book in that name.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# A name is "small" within the S&P 500 below this dollar ADV.
SMALL_CAP_ADV_USD = 50_000_000.0


@dataclass(frozen=True)
class CostModel:
    spread_bps_large: float = 5.0
    spread_bps_small: float = 15.0
    impact_bps_per_pct_adv: float = 10.0
    borrow_bps_annual: float = 50.0

    def spread_cost(self, traded: pd.Series, adv_usd: pd.Series) -> float:
        """Half-spread paid on |traded| notional fraction (one side)."""
        bps = np.where(
            adv_usd.reindex(traded.index).fillna(0.0) < SMALL_CAP_ADV_USD,
            self.spread_bps_small,
            self.spread_bps_large,
        )
        return float((traded.abs() * (bps / 1e4) * 0.5).sum())

    def impact_cost(
        self, traded: pd.Series, book_usd: float, adv_usd: pd.Series
    ) -> float:
        """Square-root impact: bps scales with sqrt(participation)."""
        adv = adv_usd.reindex(traded.index).fillna(np.inf)
        trade_usd = traded.abs() * book_usd
        participation = np.where(adv > 0, trade_usd / adv, 0.0)  # fraction of ADV
        bps = self.impact_bps_per_pct_adv * np.sqrt(participation * 100.0)
        return float((trade_usd / book_usd * bps / 1e4).sum())

    def borrow_cost_daily(self, short_weight_abs: float) -> float:
        """One day of borrow on the short book (annual / 252)."""
        return short_weight_abs * (self.borrow_bps_annual / 1e4) / 252.0


ZERO_COST = CostModel(0.0, 0.0, 0.0, 0.0)
