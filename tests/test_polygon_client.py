"""Polygon client: network mocked, no API key needed."""
import pandas as pd
import pytest

from src.config import Settings
from src.data import polygon_client as pc


@pytest.fixture(autouse=True)
def _no_credential_gate(monkeypatch):
    # require() is a method on the frozen Settings class; patch the class.
    monkeypatch.setattr(Settings, "require", lambda self, *a, **k: None)


def test_get_daily_bars_parses_and_paginates(monkeypatch):
    pages = [
        {
            "results": [
                {"t": 1672617600000, "o": 1, "h": 2, "l": 0.5, "c": 1.5,
                 "v": 100, "vw": 1.2, "n": 10},
            ],
            "next_url": "https://api.polygon.io/next",
        },
        {
            "results": [
                {"t": 1672704000000, "o": 1.5, "h": 2.5, "l": 1, "c": 2,
                 "v": 200, "vw": 1.8, "n": 20},
            ]
        },
    ]
    calls = {"i": 0}

    def fake_get_json(url, **kw):
        p = pages[calls["i"]]
        calls["i"] += 1
        return p

    monkeypatch.setattr(pc, "get_json", fake_get_json)
    df = pc.PolygonClient(api_key="x").get_daily_bars("AAPL", "2023-01-01", "2023-01-31")

    assert list(df.columns) == ["open", "high", "low", "close", "volume", "vwap", "transactions"]
    assert len(df) == 2
    assert df.index.is_monotonic_increasing
    assert df["close"].tolist() == [1.5, 2.0]


def test_empty_results_typed(monkeypatch):
    monkeypatch.setattr(pc, "get_json", lambda url, **kw: {"results": []})
    df = pc.PolygonClient(api_key="x").get_daily_bars("ZZZZ", "2023-01-01", "2023-01-31")
    assert df.empty
    assert df.index.name == "date"
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "vwap", "transactions"]
