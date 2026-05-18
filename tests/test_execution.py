"""Layer 7 — execution algorithm, fully offline with a virtual clock."""
import itertools

import pytest

from src.execution.broker import OrderState, Quote, guard_live
from src.execution.order_manager import ExecConfig, OrderManager, OrderRequest


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s          # virtual time, no real waiting


class FakeBroker:
    """Fills an order only once its age >= fill_after. To exercise the
    re-quote path, aggressive (limit != mid) orders never fill; orders
    priced exactly at mid fill quickly."""

    def __init__(self, clock, *, mid=100.0, spread=0.20,
                 fill_after=1.0, locate_ok=True, mid_only=False):
        self.clock = clock
        self.q = Quote(bid=mid - spread / 2, ask=mid + spread / 2)
        self.fill_after = fill_after
        self.locate_ok = locate_ok
        self.mid_only = mid_only
        self._orders = {}
        self._ids = itertools.count(1)
        self.submits = []

    def get_quote(self, s):
        return self.q

    def can_locate(self, s, qty):
        return self.locate_ok

    def submit_limit(self, symbol, qty, side, limit_price):
        oid = str(next(self._ids))
        self.submits.append((symbol, qty, side, round(limit_price, 4)))
        self._orders[oid] = {
            "symbol": symbol, "qty": qty, "side": side,
            "limit": limit_price, "t0": self.clock.now(),
        }
        return OrderState(oid, symbol, qty, side, limit_price, "new")

    def get_order(self, oid):
        o = self._orders[oid]
        age = self.clock.now() - o["t0"]
        at_mid = abs(o["limit"] - self.q.mid) < 1e-9
        fills = age >= self.fill_after and (at_mid or not self.mid_only)
        if fills:
            return OrderState(oid, o["symbol"], o["qty"], o["side"],
                              o["limit"], "filled", o["qty"], o["limit"])
        return OrderState(oid, o["symbol"], o["qty"], o["side"],
                          o["limit"], "new")

    def cancel(self, oid):
        self._orders[oid]["limit"] = -1  # ensure a stale order never fills


CFG = ExecConfig(poll_seconds=30, requote_after=300,
                 giveup_after=900, slice_spacing=300)


def test_guard_live():
    guard_live(False, False)              # paper: fine
    guard_live(True, True)                # live + confirm: fine
    with pytest.raises(RuntimeError):
        guard_live(True, False)           # live without confirm: refused


def test_fill_and_slippage_sign():
    ck = FakeClock()
    b = FakeBroker(ck, mid=100.0, spread=0.20, fill_after=1.0)
    om = OrderManager(b, ck, CFG)
    elog = om.execute([OrderRequest("AAA", "buy", 100, adv_shares=1e9)])
    assert len(elog.fills) == 1
    f = elog.fills[0]
    # bought at mid+half(=100.10) vs decision mid 100 -> +~10 bps cost
    assert f.fill_price == pytest.approx(100.10)
    assert f.slippage_bps == pytest.approx(10.0, abs=0.1)


def test_requote_at_mid_after_5min():
    ck = FakeClock()
    b = FakeBroker(ck, fill_after=1.0, mid_only=True)  # only mid-priced fills
    om = OrderManager(b, ck, CFG)
    elog = om.execute([OrderRequest("BBB", "buy", 10, adv_shares=1e9)])
    assert len(elog.fills) == 1
    # first submit aggressive, second submit at mid (the re-quote)
    assert b.submits[0][3] == pytest.approx(100.10)
    assert b.submits[1][3] == pytest.approx(100.00)
    assert ck.now() >= CFG.requote_after


def test_giveup_after_15min_skips():
    ck = FakeClock()
    b = FakeBroker(ck, fill_after=1e18)        # never fills
    om = OrderManager(b, ck, CFG)
    elog = om.execute([OrderRequest("CCC", "sell", 10, adv_shares=1e9)])
    assert elog.fills == []
    assert elog.skipped and "15m" in elog.skipped[0][1]
    assert ck.now() >= CFG.giveup_after


def test_short_without_locate_skipped():
    ck = FakeClock()
    b = FakeBroker(ck, locate_ok=False)
    om = OrderManager(b, ck, CFG)
    elog = om.execute([OrderRequest("DDD", "sell", 10, adv_shares=1e9,
                                    is_short=True)])
    assert elog.fills == []
    assert elog.skipped[0][1] == "no locate for short"


def test_adv_chunking_spaces_slices():
    ck = FakeClock()
    b = FakeBroker(ck, fill_after=1.0)
    om = OrderManager(b, ck, CFG)
    # 5000 shares vs ADV 100000 -> 1% cap = 1000 -> 5 slices
    elog = om.execute([OrderRequest("EEE", "buy", 5000, adv_shares=100_000)])
    assert len(elog.fills) == 5
    assert all(abs(f.qty - 1000) < 1e-9 for f in elog.fills)
    assert ck.now() >= 4 * CFG.slice_spacing      # 4 gaps between 5 slices


def test_daily_summary():
    ck = FakeClock()
    b = FakeBroker(ck, fill_after=1.0)
    om = OrderManager(b, ck, CFG)
    elog = om.execute([
        OrderRequest("AAA", "buy", 100, adv_shares=1e9),
        OrderRequest("BBB", "sell", 50, adv_shares=1e9, is_short=True),
    ])
    s = elog.daily_summary()
    assert s["trades"] == 2 and s["fill_rate"] == 1.0
    assert "avg_slippage_bps" in s
