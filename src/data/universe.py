"""Point-in-time S&P 500 membership.

True PIT membership (including historical removals) cannot be derived from
the *current* Wikipedia table — that table is itself survivorship-biased.
So membership is driven by a local CSV with explicit add/remove dates:

    ticker,name,date_added,date_removed

``date_removed`` blank == still a member. ``get_universe(d)`` returns the
set of tickers that were members on date ``d``.

``bootstrap_current_from_wikipedia`` can seed the *current* constituents
(network) as a convenience, but it is explicitly a snapshot, not history;
for a leakage-free backtest, supply a real historical membership file
(e.g. the well-known fja05680/sp500 dataset, or a paid PIT source).
"""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import pandas as pd

from src.config import settings

log = logging.getLogger("greymatter.data.universe")

_MEMBERSHIP_FILE = settings.paths.universe / "sp500_membership.csv"
_COLS = ["ticker", "name", "date_added", "date_removed"]


class UniverseProvider:
    def __init__(self, membership_file: Path | None = None) -> None:
        self._file = membership_file or _MEMBERSHIP_FILE
        self._df: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        if not self._file.exists():
            raise FileNotFoundError(
                f"Membership file not found: {self._file}. Provide a PIT "
                f"membership CSV with columns {_COLS}, or call "
                f"bootstrap_current_from_wikipedia() for a current snapshot."
            )
        df = pd.read_csv(self._file, dtype={"ticker": str})
        df["date_added"] = pd.to_datetime(df["date_added"])
        df["date_removed"] = pd.to_datetime(df["date_removed"], errors="coerce")
        self._df = df
        return df

    def get_universe(self, date: str | dt.date | pd.Timestamp) -> list[str]:
        """Tickers that were S&P 500 members on ``date`` (sorted)."""
        d = pd.Timestamp(date).normalize()
        df = self._load()
        member = (df["date_added"] <= d) & (
            df["date_removed"].isna() | (df["date_removed"] > d)
        )
        return sorted(df.loc[member, "ticker"].str.upper().unique().tolist())

    def all_tickers_ever(self) -> list[str]:
        """Every ticker that was *ever* a member — includes delisted names.

        This is the correct iteration set for a survivorship-free backtest.
        """
        return sorted(self._load()["ticker"].str.upper().unique().tolist())


def bootstrap_current_from_wikipedia(dest: Path | None = None) -> Path:
    """Seed a *current* membership CSV from Wikipedia (requires network).

    Snapshot only — every row gets a ``date_added`` from Wikipedia's
    column and a blank ``date_removed``. Not valid for historical PIT.
    """
    dest = dest or _MEMBERSHIP_FILE
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)  # lxml-backed; network call
    t = tables[0]
    out = pd.DataFrame(
        {
            "ticker": t["Symbol"].astype(str).str.replace(".", "-", regex=False),
            "name": t["Security"].astype(str),
            "date_added": pd.to_datetime(t.get("Date added"), errors="coerce"),
            "date_removed": pd.NaT,
        }
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    log.warning("Wrote CURRENT-SNAPSHOT membership to %s (not PIT history)", dest)
    return dest
