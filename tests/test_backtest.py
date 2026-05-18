"""Layer 3 — engine, portfolio, costs, metrics, and the three mandated
correctness tests: lookahead, survivorship, transaction-cost sensitivity.
"""
import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import ZERO_COST, CostModel
from src.backtest.engine import WalkForwardBacktest
from src.backtest.metrics import max_drawdown, sharpe, turnover
from src.backtest.portfolio import build_long_short

IDX = pd.bdate_range("2016-01-01", "2021-12-31")
TICKERS = ["AA", "BB", "CC", "DD", "EE", "FF", "GG", "HH"]


class FakePx:
    """Deterministic prices. Identical for all callers up to ``corrupt_after``;
    beyond it, optionally scaled garbage (unused by as-of-correct signals)."""

    def __init__(self, corrupt_after=None, tail_factor=1.0):
        self.corrupt_after = pd.Timestamp(corrupt_after) if corrupt_after else None
        self.tail_factor = tail_factor

    def _series(self, t):
        seed = abs(hash(t)) % 2**32
        rng = np.random.default_rng(seed)
        drift = {"AA": 0.0012, "BB": 0.0009, "CC": 0.0006, "DD": 0.0003,
                 "EE": 0.0001, "FF": -0.0002, "GG": 0.0005, "HH": 0.0007,
                 "SPY": 0.0004, "DEAD": -0.0030}.get(t, 0.0002)
        px = 100 * np.cumprod(1 + rng.normal(drift, 0.011, len(IDX)))
        s = pd.Series(px, index=IDX)
        if self.corrupt_after is not None and self.tail_factor != 1.0:
            mask = s.index > self.corrupt_after
            # Ticker-specific garbage so the cross-section actually changes
            # (a uniform scalar would cancel in any return ratio).
            grng = np.random.default_rng((seed ^ 0xBADF00D) % 2**32)
            n = int(mask.sum())
            s.loc[mask] = 100 * np.cumprod(
                1 + grng.normal(0.0, 0.05 * self.tail_factor, n)
            )
        return s

    def get_prices(self, t, start, end):
        s = self._series(t)
        s = s.loc[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
        return pd.DataFrame({"close": s, "volume": 5_000_000.0})


def universe_full(date):
    d = pd.Timestamp(date)
    base = list(TICKERS)
    if d < pd.Timestamp("2020-03-31"):   # DEAD delists 2020-03-31
        base = base + ["DEAD"]
    return base


def universe_survivors(date):
    return list(TICKERS)  # pretends delisted names never existed (biased)


def make_score_fn(px, lookback=120, peek=False):
    def score_fn(as_of, tickers):
        end = "2021-12-31" if peek else as_of  # peek = deliberate leakage
        out = {}
        for t in tickers:
            df = px.get_prices(t, "2016-01-01", end)
            c = df["close"]
            c = c.loc[c.index <= pd.Timestamp(as_of)] if not peek else c
            out[t] = (c.iloc[-1] / c.iloc[-lookback] - 1) if len(c) > lookback else np.nan
        return pd.Series(out)
    return score_fn


# ---- unit: portfolio / costs / metrics --------------------------------
def test_build_long_short_balanced():
    s = pd.Series({f"t{i}": i for i in range(10)})
    w = build_long_short(s, decile=0.2)
    assert abs(w[w > 0].sum() - 1.0) < 1e-12
    assert abs(w[w < 0].sum() + 1.0) < 1e-12
    assert w["t9"] > 0 and w["t8"] > 0 and w["t0"] < 0 and w["t1"] < 0
    assert (w[[f"t{i}" for i in range(2, 8)]] == 0).all()


def test_costs_zero_vs_real():
    traded = pd.Series({"a": 0.1, "b": -0.1})
    adv = pd.Series({"a": 1e9, "b": 1e9})
    assert ZERO_COST.spread_cost(traded, adv) == 0.0
    assert CostModel().spread_cost(traded, adv) > 0.0


def test_metrics_basic():
    r = pd.Series(np.r_[np.full(10, 0.01), np.full(5, -0.02)])
    assert sharpe(pd.Series(np.zeros(20))) == 0.0
    mdd, dur = max_drawdown(r)
    assert mdd < 0 and dur >= 1


# ---- mandated: lookahead ----------------------------------------------
def test_lookahead_positions_invariant_to_future():
    """As-of-correct signal: future corruption must NOT change decisions.
    A deliberately leaky signal MUST change — proving the test is sensitive."""
    clean = FakePx()
    poisoned = FakePx(corrupt_after="2020-06-30", tail_factor=8.0)

    def run(px, peek):
        bt = WalkForwardBacktest(
            px, universe_full, make_score_fn(px, peek=peek),
            cost_model=ZERO_COST, min_history_months=24, decile=0.5, min_names=6)
        # All decision/realised dates are BEFORE corrupt_after, so the
        # corrupted region is genuinely future w.r.t. every rebalance —
        # reachable only by a signal that illegitimately peeks ahead.
        return bt.run("2018-06-29", "2019-12-31", run_id="lh")

    p_clean = run(clean, peek=False).positions
    p_pois = run(poisoned, peek=False).positions
    pd.testing.assert_frame_equal(p_clean, p_pois)          # no leakage

    leak_clean = run(clean, peek=True).positions
    leak_pois = run(poisoned, peek=True).positions
    with pytest.raises(AssertionError):                      # test is sensitive
        pd.testing.assert_frame_equal(leak_clean, leak_pois)


# ---- mandated: survivorship -------------------------------------------
def test_survivorship_changes_results():
    px = FakePx()
    full = WalkForwardBacktest(px, universe_full, make_score_fn(px),
                               cost_model=ZERO_COST, min_history_months=24,
                               decile=0.5, min_names=6).run("2019-06-30", "2020-06-30",
                                                run_id="surv_full")
    surv = WalkForwardBacktest(px, universe_survivors, make_score_fn(px),
                               cost_model=ZERO_COST, min_history_months=24,
                               decile=0.5, min_names=6).run("2019-06-30", "2020-06-30",
                                                run_id="surv_only")
    assert full.metrics["ann_return"] != surv.metrics["ann_return"]


# ---- mandated: transaction costs --------------------------------------
def test_costs_drag_returns():
    px = FakePx()
    zero = WalkForwardBacktest(px, universe_full, make_score_fn(px),
                               cost_model=ZERO_COST, min_history_months=24,
                               decile=0.5, min_names=6).run("2019-06-30", "2021-06-30",
                                                run_id="cost_zero")
    real = WalkForwardBacktest(px, universe_full, make_score_fn(px),
                               cost_model=CostModel(), min_history_months=24,
                               decile=0.5, min_names=6).run("2019-06-30", "2021-06-30",
                                                run_id="cost_real")
    assert real.metrics["ann_return"] < zero.metrics["ann_return"]
    assert real.metrics["sharpe"] <= zero.metrics["sharpe"]


def test_artifacts_written(tmp_path):
    px = FakePx()
    res = WalkForwardBacktest(px, universe_full, make_score_fn(px),
                              cost_model=CostModel(), min_history_months=24,
                              decile=0.5, min_names=6).run("2019-06-30", "2020-12-31",
                                               run_id="artifacts")
    from pathlib import Path
    out = Path(res.out_dir)
    for f in ("returns.parquet", "positions.parquet", "metrics.json",
              "tearsheet.html"):
        assert (out / f).exists()
