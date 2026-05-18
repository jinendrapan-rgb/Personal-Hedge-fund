"""Polygon.io daily bars.

Uses the aggregates v2 endpoint with ``adjusted=true`` so splits and
dividends are handled by Polygon. Delisted tickers are returned by Polygon
on the Starter plan, which is what keeps the backtest survivorship-free.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import pandas as pd

from src.config import settings
from src.data._http import RateLimiter, get_json

log = logging.getLogger("greymatter.data.polygon")

_BASE = "https://api.polygon.io"
_COLS = ["open", "high", "low", "close", "volume", "vwap", "transactions"]


def _to_ms(d: str | dt.date) -> str:
    if isinstance(d, dt.date):
        return d.isoformat()
    return d


class PolygonClient:
    """Thin client for daily OHLCV. One instance is rate-limit shared."""

    def __init__(self, api_key: str | None = None, rate_per_sec: float | None = None) -> None:
        self._key = api_key if api_key is not None else settings.polygon_api_key
        self._limiter = RateLimiter(rate_per_sec or settings.polygon_rate_per_sec)

    def get_daily_bars(
        self, ticker: str, start: str | dt.date, end: str | dt.date
    ) -> pd.DataFrame:
        """Daily adjusted OHLCV for ``ticker`` in ``[start, end]``.

        Returns a DataFrame indexed by tz-naive ``date`` with columns
        ``open, high, low, close, volume, vwap, transactions``. Empty
        DataFrame (correct schema) if the ticker has no data in range.
        """
        settings.require("polygon_api_key")
        url = (
            f"{_BASE}/v2/aggs/ticker/{ticker.upper()}/range/1/day/"
            f"{_to_ms(start)}/{_to_ms(end)}"
        )
        rows: list[dict[str, Any]] = []
        params: dict[str, Any] = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self._key,
        }
        next_url: str | None = url
        while next_url:
            payload = get_json(
                next_url,
                limiter=self._limiter,
                params=params if next_url == url else {"apiKey": self._key},
            )
            rows.extend(payload.get("results", []) or [])
            next_url = payload.get("next_url")
        return self._frame(rows)

    @staticmethod
    def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
        if not rows:
            empty = pd.DataFrame(columns=_COLS)
            empty.index = pd.DatetimeIndex([], name="date")
            return empty
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["t"], unit="ms").dt.normalize()
        out = df.rename(
            columns={
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "vw": "vwap",
                "n": "transactions",
            }
        ).set_index("date")[_COLS]
        out.index.name = "date"
        return out.sort_index()
