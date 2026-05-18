"""Sector lookup for neutralisation.

True GICS sectors (point-in-time) require a paid mapping. We use an
explicit ``data/universe/sectors.csv`` (``ticker,sector``) so the source
is swappable and auditable; unknown tickers fall into ``"UNKNOWN"`` (still
neutralised among themselves rather than silently dropped). Document this
limitation — a leakage-free run wants PIT sectors as of the rebalance date.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import settings

_FILE = settings.paths.universe / "sectors.csv"


class SectorProvider:
    def __init__(self, file: Path | None = None) -> None:
        self._file = file or _FILE
        self._map: dict[str, str] | None = None

    def _load(self) -> dict[str, str]:
        if self._map is not None:
            return self._map
        if self._file.exists():
            df = pd.read_csv(self._file, dtype=str)
            self._map = {
                str(t).upper(): str(s)
                for t, s in zip(df["ticker"], df["sector"])
            }
        else:
            self._map = {}
        return self._map

    def get(self, tickers: list[str]) -> pd.Series:
        m = self._load()
        return pd.Series(
            {t: m.get(t.upper(), "UNKNOWN") for t in tickers}, name="sector"
        )
