"""Layer-7 verification: run the real execution algorithm against a
simulated broker on a virtual clock (no Alpaca key needed). If Alpaca
keys are present it reports paper/live availability — it does NOT place
real orders.
"""
from __future__ import annotations

import itertools
import sys

from src.config import settings
from src.execution.broker import OrderState, Quote
from src.execution.order_manager import ExecConfig, OrderManager, OrderRequest


class SimClock:
    def __init__(self): self.t = 0.0
    def now(self): return self.t
    def sleep(self, s): self.t += s


class SimBroker:
    def __init__(self, clock):
        self.clock = clock
        self.q = Quote(bid=49.95, ask=50.05)
        self._o = {}
        self._id = itertools.count(1)

    def get_quote(self, s): return self.q
    def can_locate(self, s, q): return s != "HTB"   # HTB = hard-to-borrow

    def submit_limit(self, symbol, qty, side, limit):
        oid = str(next(self._id))
        self._o[oid] = (symbol, qty, side, limit, self.clock.now())
        return OrderState(oid, symbol, qty, side, limit, "new")

    def get_order(self, oid):
        sym, qty, side, lim, t0 = self._o[oid]
        if self.clock.now() - t0 >= 60:           # fills after ~1 min
            return OrderState(oid, sym, qty, side, lim, "filled", qty, lim)
        return OrderState(oid, sym, qty, side, lim, "new")

    def cancel(self, oid): self._o[oid] = (*self._o[oid][:3], -1, self._o[oid][4])


def main() -> int:
    ck = SimClock()
    om = OrderManager(SimBroker(ck), ck,
                      ExecConfig(poll_seconds=15, requote_after=300,
                                 giveup_after=900, slice_spacing=300))
    book = [
        OrderRequest("AAA", "buy", 500, adv_shares=1_000_000),
        OrderRequest("BBB", "sell", 300, adv_shares=1_000_000, is_short=True),
        OrderRequest("CCC", "buy", 40_000, adv_shares=1_000_000),   # >1% ADV
        OrderRequest("HTB", "sell", 100, adv_shares=1_000_000, is_short=True),
    ]
    elog = om.execute(book)

    print("FILLS")
    for f in elog.fills:
        print(f"  {f.symbol} {f.side} {f.qty:.0f} @ {f.fill_price:.2f} "
              f"slip={f.slippage_bps:+.1f}bps")
    print("SKIPPED")
    for sym, why in elog.skipped:
        print(f"  {sym}: {why}")
    print("\nDAILY SUMMARY:", elog.daily_summary())
    print("  (CCC = 40k vs 1M ADV -> 4 slices of 1% ADV; HTB -> no locate)")

    has_keys = bool(settings.alpaca_api_key and settings.alpaca_secret_key)
    print(f"\nAlpaca keys present: {has_keys} | "
          f"mode: {'LIVE (needs confirm)' if settings.alpaca_live else 'paper'}")
    print("Live trading requires ALPACA_LIVE=true AND confirm_live=True.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
