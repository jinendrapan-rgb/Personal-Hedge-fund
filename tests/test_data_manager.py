"""Incremental price caching: second call must fetch only the missing tail."""
import types

import pandas as pd
import pytest

from src.data import data_manager as dm_mod
from src.data.data_manager import DataManager


class FakePolygon:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def get_daily_bars(self, ticker, start, end):
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        self.calls.append((s.date().isoformat(), e.date().isoformat()))
        idx = pd.bdate_range(s, e)
        if len(idx) == 0:
            empty = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "vwap", "transactions"]
            )
            empty.index = pd.DatetimeIndex([], name="date")
            return empty
        df = pd.DataFrame(
            {
                "open": 1.0, "high": 1.0, "low": 1.0,
                "close": range(len(idx)), "volume": 1, "vwap": 1.0, "transactions": 1,
            },
            index=idx,
        )
        df.index.name = "date"
        return df


@pytest.fixture
def patched_settings(tmp_path, monkeypatch):
    paths = types.SimpleNamespace(
        prices=tmp_path / "prices",
        fundamentals=tmp_path / "fundamentals",
        ensure=lambda: (
            (tmp_path / "prices").mkdir(exist_ok=True),
            (tmp_path / "fundamentals").mkdir(exist_ok=True),
        ),
    )
    (tmp_path / "prices").mkdir()
    (tmp_path / "fundamentals").mkdir()
    fake = types.SimpleNamespace(paths=paths)
    monkeypatch.setattr(dm_mod, "settings", fake)
    return fake


def test_incremental_fetches_only_missing_tail(patched_settings):
    poly = FakePolygon()
    dm = DataManager(polygon=poly, sec=object(), universe=object())

    a = dm.get_prices("AAPL", "2023-02-01", "2023-02-28")
    assert len(poly.calls) == 1
    assert not a.empty

    # Widen on both sides — should fetch two tails only, not the middle.
    b = dm.get_prices("AAPL", "2023-01-01", "2023-03-31")
    assert len(poly.calls) == 3
    assert poly.calls[1] == ("2023-01-01", "2023-01-31")
    assert poly.calls[2] == ("2023-03-01", "2023-03-31")
    assert b.index.min() >= pd.Timestamp("2023-01-01")
    assert b.index.max() <= pd.Timestamp("2023-03-31")

    # Fully cached range — no new network calls.
    dm.get_prices("AAPL", "2023-02-05", "2023-02-20")
    assert len(poly.calls) == 3


def test_fundamentals_uses_pit_filter(patched_settings, monkeypatch):
    facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"end": "2018-12-31", "val": 1000, "form": "10-K",
                             "filed": "2019-02-15", "fy": 2018, "fp": "FY", "accn": "a1"},
                            {"end": "2018-12-31", "val": 800, "form": "10-K/A",
                             "filed": "2019-11-01", "fy": 2018, "fp": "FY", "accn": "a2"},
                        ]
                    }
                }
            }
        }
    }
    fake_sec = types.SimpleNamespace(get_company_facts=lambda t: facts)
    dm = DataManager(polygon=object(), sec=fake_sec, universe=object())

    assert dm.get_fundamentals("AAPL", "2019-06-01")["revenue"] == 1000.0
    assert dm.get_fundamentals("AAPL", "2020-01-01")["revenue"] == 800.0
