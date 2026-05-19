"""Factor engine orchestrator.

Computes the five factors as-of a date using only data available on that
date (price history is truncated at ``as_of``; fundamentals come through
``DataManager.get_fundamentals`` which enforces the point-in-time
contract). Output columns:

    date, ticker, momentum, quality, value, low_vol, revisions, composite

Every per-factor column is a sector-neutral z-score; ``composite`` is the
equal-weight mean of the five (NaN-skipping per row). Higher = long.
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from src.factors.low_vol import low_volatility
from src.factors.momentum import momentum_12_1
from src.factors.neutralise import (
    combine_zscores,
    rank_within_sector,
    sector_neutralise,
    winsorise,
    zscore,
)
from src.factors.quality import quality_signals
from src.factors.revisions import revisions_signals
from src.factors.sectors import SectorProvider
from src.factors.value import value_signals

log = logging.getLogger("greymatter.factors")

_LOOKBACK_DAYS = 460          # ~13 months of trading history for momentum
_PRIOR_YEAR = pd.Timedelta(days=365)
_PRIOR_QUARTER = pd.Timedelta(days=92)
FACTORS = ["momentum", "quality", "value", "low_vol", "revisions"]


class FactorEngine:
    def __init__(self, dm, sectors: SectorProvider | None = None, estimates=None):
        self.dm = dm
        self.sectors = sectors or SectorProvider()
        # estimates: optional provider with .get(ticker, as_of) -> dict
        # (fy1_eps, n_up, n_down, n_total). Absent -> revisions factor NaN.
        self.estimates = estimates

    # ---- assemble inputs ----------------------------------------------
    def _close_matrix(self, tickers: list[str], as_of: pd.Timestamp) -> pd.DataFrame:
        start = (as_of - pd.Timedelta(days=_LOOKBACK_DAYS)).date().isoformat()
        cols = {}
        for t in tickers:
            try:
                px = self.dm.get_prices(t, start, as_of.date().isoformat())
            except Exception as e:  # one bad ticker must not sink the run
                log.warning("price fetch failed %s: %r", t, e)
                continue
            if not px.empty:
                cols[t] = px["close"]
        if not cols:
            return pd.DataFrame()
        return pd.DataFrame(cols).sort_index()

    def _fund(self, t: str, when) -> dict:
        try:
            return self.dm.get_fundamentals(t, when)
        except Exception as e:
            log.warning("fundamentals failed %s @ %s: %r", t, when, e)
            return {}

    # ---- main ----------------------------------------------------------
    def compute(
        self, as_of: str | dt.date, universe: list[str] | None = None
    ) -> pd.DataFrame:
        asof = pd.Timestamp(as_of).normalize()
        # Explicit universe lets the pipeline operate without a PIT
        # membership file; falls back to the provider when not given.
        tickers = list(universe) if universe else list(
            self.dm.get_universe(asof.date().isoformat()))
        if not tickers:
            return self._empty()

        close = self._close_matrix(tickers, asof)
        present = [t for t in tickers if t in close.columns] if not close.empty else []
        if not present:
            return self._empty()
        sectors = self.sectors.get(present)

        # price-based factors
        mom = momentum_12_1(close[present], asof).reindex(present)
        lvol = low_volatility(close[present], asof).reindex(present)
        last_px = close[present].ffill().iloc[-1]

        # fundamentals-based per-ticker sub-signals
        q_rows, v_rows, r_rows = {}, {}, {}
        for t in present:
            curr = self._fund(t, asof.date().isoformat())
            prev = self._fund(t, (asof - _PRIOR_YEAR).date().isoformat())
            q_rows[t] = quality_signals(curr, prev)

            so = curr.get("shares_outstanding")
            mcap = (so * last_px[t]) if (so and np.isfinite(last_px[t])) else None
            v_rows[t] = value_signals(curr, mcap)

            if self.estimates is not None:
                e_now = self.estimates.get(t, asof.date().isoformat()) or {}
                e_old = self.estimates.get(
                    t, (asof - _PRIOR_QUARTER).date().isoformat()
                ) or {}
                r_rows[t] = revisions_signals(e_now, e_old)
            else:
                r_rows[t] = {"eps_change": np.nan, "breadth": np.nan}

        idx = pd.Index(present, name="ticker")
        qdf = pd.DataFrame(q_rows).T.reindex(idx).astype(float)
        vdf = pd.DataFrame(v_rows).T.reindex(idx).astype(float)
        rdf = pd.DataFrame(r_rows).T.reindex(idx).astype(float)

        # --- neutralise each factor to a sector-neutral z-score ---------
        z = pd.DataFrame(index=idx)
        z["momentum"] = sector_neutralise(winsorise(mom.reindex(idx)), sectors)
        z["low_vol"] = sector_neutralise(winsorise(lvol.reindex(idx)), sectors)

        # quality: z each sub-signal sector-neutral, average, re-neutralise.
        # The final pass matters: sub-signals have different NaN patterns, so
        # their row-wise mean is not automatically sector-neutral.
        q_sub = pd.DataFrame(
            {c: sector_neutralise(qdf[c], sectors) for c in qdf.columns}, index=idx
        )
        z["quality"] = sector_neutralise(combine_zscores(q_sub), sectors)

        # value: rank sub-metrics within sector, average, then z
        v_sub = pd.DataFrame(
            {c: rank_within_sector(vdf[c], sectors) for c in vdf.columns}, index=idx
        )
        z["value"] = sector_neutralise(combine_zscores(v_sub), sectors)

        # revisions: z each sub-signal sector-neutral, then average
        if rdf.notna().any().any():
            r_sub = pd.DataFrame(
                {c: sector_neutralise(rdf[c], sectors) for c in rdf.columns},
                index=idx,
            )
            z["revisions"] = sector_neutralise(combine_zscores(r_sub), sectors)
        else:
            z["revisions"] = np.nan

        z["composite"] = combine_zscores(z[FACTORS])

        z = z[[*FACTORS, "composite"]]  # canonical column order
        out = z.reset_index()
        out.insert(0, "date", asof)
        return out.sort_values("composite", ascending=False).reset_index(drop=True)

    @staticmethod
    def _empty() -> pd.DataFrame:
        return pd.DataFrame(
            columns=["date", "ticker", *FACTORS, "composite"]
        )
