"""Layer 6 — pre-trade, circuit breakers, factor risk, correlation, stress."""
import numpy as np
import pandas as pd
import pytest

from src.risk.circuit_breakers import BreakerState, evaluate
from src.risk.correlation import monitor
from src.risk.factor_model import decompose
from src.risk.pre_trade import pre_trade_check
from src.risk.stress_tests import run_stress

T = [f"S{i:02d}" for i in range(10)]
SEC = pd.Series({t: ["Tech", "Energy"][i % 2] for i, t in enumerate(T)})
BETA = pd.Series({t: 1.0 for t in T})


class FakeLocate:
    def __init__(self, hard_to_borrow=()):
        self.htb = set(hard_to_borrow)

    def is_shortable(self, ticker):
        return ticker not in self.htb


def _good_book():
    w = pd.Series(0.0, index=T)
    w[T[:2]] = 0.02         # longs in Tech & Energy
    w[T[1]] = 0.02
    w[T[2:4]] = -0.02       # shorts balance sectors
    # sector-net ~0: long S00(Tech),S01(Energy); short S02(Tech),S03(Energy)
    w[:] = 0.0
    w["S00"], w["S02"] = 0.02, -0.02     # Tech net 0
    w["S01"], w["S03"] = 0.02, -0.02     # Energy net 0
    return w


# ---- pre-trade --------------------------------------------------------
def test_pre_trade_passes_clean_book():
    w = _good_book()
    adv = pd.Series(1e9, index=T)
    res = pre_trade_check(w, SEC, BETA, adv_usd=adv, book_usd=1e7,
                          locate=FakeLocate())
    assert res.ok, res.violations


def test_pre_trade_flags_oversize_and_illiquid_and_borrow():
    w = pd.Series(0.0, index=T)
    w["S00"] = 0.09                       # > 4% box
    w["S01"] = -0.02                      # short, hard to borrow
    adv = pd.Series(1e9, index=T)
    adv["S00"] = 1.0                      # tiny ADV -> huge impact
    res = pre_trade_check(w, SEC, BETA, adv_usd=adv, book_usd=1e8,
                          locate=FakeLocate(hard_to_borrow={"S01"}))
    assert not res.ok
    joined = " ".join(res.violations)
    assert "position size" in joined and "liquidity" in joined and "borrow" in joined


# ---- circuit breakers -------------------------------------------------
def test_breaker_states():
    assert evaluate(pd.Series(dtype=float)).state == BreakerState.GREEN
    assert evaluate(pd.Series([0.001, -0.001, 0.0005])).state == BreakerState.GREEN
    assert evaluate(pd.Series([0.0, -0.03])).state == BreakerState.HALT_DAY
    wk = pd.Series([-0.011, -0.011, -0.011, -0.011, -0.011])
    assert evaluate(wk).state == BreakerState.HALT_WEEK
    crash = pd.Series([-0.03] * 4)
    s = evaluate(crash)
    assert s.state == BreakerState.SYSTEM_STOP and not s.can_trade


# ---- factor risk ------------------------------------------------------
def test_factor_decomp_specific_share_bounds():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2022-01-03", periods=260)
    f = rng.normal(0, 0.01, len(idx))
    # mostly idiosyncratic names -> high specific share
    data = {t: 0.2 * f + rng.normal(0, 0.02, len(idx)) for t in T}
    r = pd.DataFrame(data, index=idx)
    w = _good_book()
    d = decompose(w, r, n_factors=3)
    assert 0.0 <= d.specific_share <= 1.0
    assert abs(d.factor_share + d.specific_share - 1.0) < 1e-9
    assert d.specific_share > 0.5            # idiosyncratic-dominated book


# ---- correlation ------------------------------------------------------
def test_correlation_flags_pair_and_cluster():
    idx = pd.bdate_range("2022-01-03", periods=200)
    base = np.random.default_rng(1).normal(0, 0.01, len(idx))
    df = pd.DataFrame(index=idx)
    df["A"] = base
    df["B"] = base + np.random.default_rng(2).normal(0, 1e-4, len(idx))  # ~1.0 corr
    df["C"] = np.random.default_rng(3).normal(0, 0.01, len(idx))
    w = pd.Series({"A": 0.5, "B": 0.5, "C": 0.01})
    rep = monitor(w, df)
    assert any({"A", "B"} == {p[0], p[1]} for p in rep.high_corr_pairs)
    assert not rep.ok


# ---- stress -----------------------------------------------------------
def test_stress_empirical_and_linear():
    idx = pd.bdate_range("2020-02-01", "2020-04-30")  # covers COVID window
    rng = np.random.default_rng(5)
    df = pd.DataFrame({t: rng.normal(-0.002, 0.03, len(idx)) for t in T},
                      index=idx)
    w = _good_book()
    res = {r.scenario: r for r in run_stress(w, df, portfolio_beta=0.05)}
    assert res["COVID_2020"].method == "empirical"
    assert res["GFC_2008"].method == "linear"      # not in history window
    assert isinstance(res["COVID_2020"].portfolio_return, float)
