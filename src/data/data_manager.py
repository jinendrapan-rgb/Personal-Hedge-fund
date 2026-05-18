"""Unified, OpenBB-style data interface with incremental parquet caching.

    dm = DataManager()
    dm.get_prices("AAPL", "2023-01-01", "2023-12-31")
    dm.get_fundamentals("AAPL", "2020-01-15")   # PIT — see xbrl_parser
    dm.get_universe("2024-06-30")

First run for a ticker downloads everything; later runs fetch only the
missing date tails. Fundamentals are cached as the full tidy XBRL frame
and the point-in-time filter is applied at read time.
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from src.config import settings
from src.data.polygon_client import PolygonClient
from src.data.sec_edgar import SecEdgarClient
from src.data.universe import UniverseProvider
from src.data.xbrl_parser import flatten, fundamentals_as_of

log = logging.getLogger("greymatter.data.manager")

_ONE_DAY = pd.Timedelta(days=1)


class DataManager:
    def __init__(
        self,
        polygon: PolygonClient | None = None,
        sec: SecEdgarClient | None = None,
        universe: UniverseProvider | None = None,
        fundamentals_max_age_days: int = 7,
    ) -> None:
        self._poly = polygon or PolygonClient()
        self._sec = sec or SecEdgarClient()
        self._uni = universe or UniverseProvider()
        self._fund_max_age = pd.Timedelta(days=fundamentals_max_age_days)
        settings.paths.ensure()

    # ---- prices --------------------------------------------------------
    def get_prices(
        self, ticker: str, start: str | dt.date, end: str | dt.date
    ) -> pd.DataFrame:
        ticker = ticker.upper()
        start_ts, end_ts = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        path = settings.paths.prices / f"{ticker}.parquet"

        cached = pd.read_parquet(path) if path.exists() else None
        if cached is not None and not cached.empty:
            cmin, cmax = cached.index.min(), cached.index.max()
            fetched = [cached]
            if start_ts < cmin:
                fetched.append(
                    self._poly.get_daily_bars(ticker, start_ts, cmin - _ONE_DAY)
                )
            if end_ts > cmax:
                fetched.append(
                    self._poly.get_daily_bars(ticker, cmax + _ONE_DAY, end_ts)
                )
            merged = pd.concat(fetched)
        else:
            merged = self._poly.get_daily_bars(ticker, start_ts, end_ts)

        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        if not merged.empty:
            merged.to_parquet(path)
        return merged.loc[(merged.index >= start_ts) & (merged.index <= end_ts)]

    # ---- fundamentals (point-in-time) ---------------------------------
    def _tidy_facts(self, ticker: str) -> pd.DataFrame:
        ticker = ticker.upper()
        path = settings.paths.fundamentals / f"{ticker}.parquet"
        fresh = (
            path.exists()
            and (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s"))
            < self._fund_max_age
        )
        if fresh:
            return pd.read_parquet(path)
        tidy = flatten(self._sec.get_company_facts(ticker))
        if not tidy.empty:
            tidy.to_parquet(path)
        return tidy

    def get_fundamentals(
        self, ticker: str, as_of_date: str | dt.date
    ) -> dict[str, float | None]:
        """Standard fundamentals for ``ticker`` exactly as known on
        ``as_of_date`` (no restatement leakage)."""
        return fundamentals_as_of(self._tidy_facts(ticker), as_of_date)

    # ---- universe ------------------------------------------------------
    def get_universe(self, date: str | dt.date) -> list[str]:
        return self._uni.get_universe(date)

    def all_tickers_ever(self) -> list[str]:
        return self._uni.all_tickers_ever()
