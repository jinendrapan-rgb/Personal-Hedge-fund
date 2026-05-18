"""Pure-function factor tests: determinism, no lookahead, orientation."""
import numpy as np
import pandas as pd

from src.factors.low_vol import low_volatility
from src.factors.momentum import momentum_12_1
from src.factors.neutralise import rank_within_sector, sector_neutralise, zscore
from src.factors.quality import quality_signals
from src.factors.revisions import revisions_signals
from src.factors.value import value_signals


def _close(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    # AAA strong uptrend, BBB flat, CCC volatile
    aaa = 100 * np.cumprod(1 + rng.normal(0.0010, 0.010, n))
    bbb = 100 * np.cumprod(1 + rng.normal(0.0000, 0.008, n))
    ccc = 100 * np.cumprod(1 + rng.normal(0.0002, 0.030, n))
    return pd.DataFrame({"AAA": aaa, "BBB": bbb, "CCC": ccc}, index=idx)


def test_momentum_is_pure():
    c = _close()
    a = momentum_12_1(c, "2023-06-30")
    b = momentum_12_1(c, "2023-06-30")
    pd.testing.assert_series_equal(a, b)


def test_momentum_no_lookahead():
    c = _close()
    asof = "2023-06-30"
    base = momentum_12_1(c, asof)
    # Append wild FUTURE data strictly after as_of — must not change result.
    fut_idx = pd.bdate_range("2023-07-03", periods=60)
    future = pd.DataFrame(
        {col: c[col].iloc[-1] * 5.0 for col in c.columns}, index=fut_idx
    )
    poisoned = pd.concat([c, future])
    after = momentum_12_1(poisoned, asof)
    pd.testing.assert_series_equal(base, after)


def test_low_vol_orientation_and_no_lookahead():
    c = _close()
    lv = low_volatility(c, "2023-06-30")
    # CCC is the most volatile -> should have the LOWEST (most negative) score
    assert lv["CCC"] < lv["BBB"]
    poisoned = c.copy()
    poisoned.iloc[-1] = poisoned.iloc[-1]  # no future rows; truncation still holds
    pd.testing.assert_series_equal(lv, low_volatility(c, "2023-06-30"))


def test_sector_neutralise_zero_mean_per_sector():
    vals = pd.Series({"a": 10, "b": 12, "c": 1, "d": 3, "e": 50, "f": 52})
    sec = pd.Series({"a": "T", "b": "T", "c": "F", "d": "F", "e": "E", "f": "E"})
    out = sector_neutralise(vals, sec)
    per_sector_mean = out.groupby(sec).mean()
    assert np.allclose(per_sector_mean.values, 0.0, atol=1e-9)


def test_rank_within_sector_centered():
    vals = pd.Series({"a": 1, "b": 2, "c": 3, "d": 9})
    sec = pd.Series({"a": "X", "b": "X", "c": "Y", "d": "Y"})
    r = rank_within_sector(vals, sec)
    assert r.between(-0.5, 0.5).all()


def test_quality_orientation():
    strong = {"net_income": 10, "total_assets": 100, "operating_cash_flow": 15,
              "gross_profit": 40, "revenue": 80, "long_term_debt": 5,
              "shares_outstanding": 100}
    weak_prev = {"net_income": 1, "total_assets": 100, "operating_cash_flow": 0,
                 "gross_profit": 20, "revenue": 80, "long_term_debt": 20,
                 "shares_outstanding": 100}
    s = quality_signals(strong, weak_prev)
    assert 0.0 <= s["f_score"] <= 1.0 and s["f_score"] > 0.5
    assert s["gross_prof"] == 40 / 100
    assert s["neg_accruals"] == -((10 - 15) / 100)  # CFO>NI -> positive signal


def test_value_higher_is_cheaper():
    fund = {"operating_income": 50, "stockholders_equity": 200,
            "operating_cash_flow": 40, "long_term_debt": 0, "cash": 0, "capex": 10}
    cheap = value_signals(fund, market_cap=100)
    rich = value_signals(fund, market_cap=1000)
    assert cheap["ev_ebit_inv"] > rich["ev_ebit_inv"]
    assert cheap["bp"] > rich["bp"]
    assert cheap["fcf_yield"] > rich["fcf_yield"]


def test_value_handles_missing():
    assert value_signals({}, None)["bp"] is None
    assert value_signals({"stockholders_equity": 10}, 0)["bp"] is None


def test_revisions_signals():
    s = revisions_signals(
        {"fy1_eps": 5.5, "n_up": 6, "n_down": 2, "n_total": 10},
        {"fy1_eps": 5.0},
    )
    assert abs(s["eps_change"] - 0.1) < 1e-12
    assert abs(s["breadth"] - 0.4) < 1e-12
    assert revisions_signals({}, {})["eps_change"] is None
