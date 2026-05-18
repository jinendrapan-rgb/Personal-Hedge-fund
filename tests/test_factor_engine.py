"""Engine end-to-end on a synthetic in-memory dataset.

Verifies schema, composite definition, ordering, determinism, engine-level
sector-neutrality, and that future prices never leak into an as-of run.
"""
import numpy as np
import pandas as pd
import pytest

from src.factors.factor_engine import FACTORS, FactorEngine
from src.factors.sectors import SectorProvider

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
ASOF = "2023-06-30"


class FakeDM:
    """get_prices returns the FULL series incl. data AFTER `end`, so the
    engine/factors must do their own as-of truncation to stay leak-free."""

    def __init__(self, extend_future_days=0, spike=1.0):
        self.extend = extend_future_days
        self.spike = spike

    def get_universe(self, date):
        return TICKERS

    def get_prices(self, t, start, end):
        rng = np.random.default_rng(abs(hash(t)) % 2**32)
        n = 560  # base series ends ~2023-07, i.e. AFTER ASOF
        idx = pd.bdate_range("2021-06-01", periods=n)
        drift = {"AAA": 0.0012, "BBB": 0.0009, "CCC": 0.0001,
                 "DDD": -0.0003, "EEE": 0.0006, "FFF": 0.0002}[t]
        px = 100 * np.cumprod(1 + rng.normal(drift, 0.012, n))
        df = pd.DataFrame({"close": px}, index=idx)
        df.index.name = "date"
        if self.extend:
            fut = pd.bdate_range(idx[-1] + pd.Timedelta(days=1),
                                 periods=self.extend)
            df = pd.concat(
                [df, pd.DataFrame({"close": px[-1] * self.spike}, index=fut)]
            )
            df.index.name = "date"
        return df

    def get_fundamentals(self, t, when):
        base = {"AAA": 1.2, "BBB": 1.0, "CCC": 0.8, "DDD": 0.6,
                "EEE": 1.1, "FFF": 0.9}[t]
        prior = pd.Timestamp(when) < pd.Timestamp("2023-01-01")
        k = base * (0.9 if prior else 1.0)
        return {
            "net_income": 10 * k, "total_assets": 100, "revenue": 80 * k,
            "operating_cash_flow": 12 * k, "gross_profit": 40 * k,
            "operating_income": 15 * k, "long_term_debt": 10,
            "stockholders_equity": 50, "shares_outstanding": 1000,
            "cash": 5, "capex": 3, "current_assets": 40,
            "current_liabilities": 20,
        }


@pytest.fixture
def sectors(tmp_path):
    f = tmp_path / "sectors.csv"
    f.write_text(
        "ticker,sector\n"
        "AAA,Tech\nBBB,Tech\nCCC,Tech\nDDD,Energy\nEEE,Energy\nFFF,Energy\n"
    )
    return SectorProvider(file=f)


def test_schema_and_one_row_per_ticker(sectors):
    df = FactorEngine(FakeDM(), sectors=sectors).compute(ASOF)
    assert list(df.columns) == ["date", "ticker", *FACTORS, "composite"]
    assert sorted(df["ticker"]) == sorted(TICKERS)
    assert (df["date"] == pd.Timestamp(ASOF)).all()


def test_composite_is_mean_of_five(sectors):
    df = FactorEngine(FakeDM(), sectors=sectors).compute(ASOF).set_index("ticker")
    expected = df[FACTORS].mean(axis=1, skipna=True)
    pd.testing.assert_series_equal(
        df["composite"], expected, check_names=False
    )


def test_sorted_descending(sectors):
    df = FactorEngine(FakeDM(), sectors=sectors).compute(ASOF)
    assert df["composite"].is_monotonic_decreasing


def test_deterministic(sectors):
    a = FactorEngine(FakeDM(), sectors=sectors).compute(ASOF)
    b = FactorEngine(FakeDM(), sectors=sectors).compute(ASOF)
    pd.testing.assert_frame_equal(a, b)


def test_engine_level_sector_neutrality(sectors):
    df = FactorEngine(FakeDM(), sectors=sectors).compute(ASOF)
    df = df.set_index("ticker")
    sec = sectors.get(list(df.index))
    for col in ["momentum", "low_vol", "quality", "value"]:
        means = df[col].groupby(sec).mean()
        assert np.allclose(means.values, 0.0, atol=1e-8), f"{col} not neutral"


def test_no_future_leak_at_engine_level(sectors):
    """Identical as-of result whether or not the DM exposes future prices."""
    clean = FactorEngine(FakeDM(extend_future_days=0), sectors=sectors).compute(ASOF)
    poisoned = FactorEngine(
        FakeDM(extend_future_days=120, spike=4.0), sectors=sectors
    ).compute(ASOF)
    pd.testing.assert_frame_equal(clean, poisoned)


def test_revisions_nan_without_provider(sectors):
    df = FactorEngine(FakeDM(), sectors=sectors).compute(ASOF)
    assert df["revisions"].isna().all()  # no estimates provider wired
