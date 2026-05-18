"""PIT universe membership, including delisted-name handling."""
import pandas as pd

from src.data.universe import UniverseProvider

CSV = """ticker,name,date_added,date_removed
AAPL,Apple,1982-11-30,
MSFT,Microsoft,1994-06-01,
LB,L Brands,1983-01-01,2021-08-02
WCG,WellCare,2016-03-25,2020-01-23
"""


def _provider(tmp_path):
    f = tmp_path / "membership.csv"
    f.write_text(CSV)
    return UniverseProvider(membership_file=f)


def test_member_inside_window(tmp_path):
    u = _provider(tmp_path)
    assert u.get_universe("2019-06-30") == ["AAPL", "LB", "MSFT", "WCG"]


def test_delisted_excluded_after_removal(tmp_path):
    u = _provider(tmp_path)
    members = u.get_universe("2022-01-01")
    assert "LB" not in members and "WCG" not in members
    assert members == ["AAPL", "MSFT"]


def test_not_yet_added(tmp_path):
    u = _provider(tmp_path)
    assert u.get_universe("2015-01-01") == ["AAPL", "LB", "MSFT"]


def test_all_tickers_ever_includes_delisted(tmp_path):
    u = _provider(tmp_path)
    # survivorship-free iteration set must keep LB and WCG
    assert u.all_tickers_ever() == ["AAPL", "LB", "MSFT", "WCG"]
