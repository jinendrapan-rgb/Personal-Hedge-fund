"""Execution algorithm.

Rules (spec):
* limit orders only, never market
* limit = mid + spread/2 (buy) / mid - spread/2 (sell)  — cross half spread
* unfilled after 5 min  -> cancel & re-quote at mid
* unfilled after 15 min -> cancel, log, skip (do not chase)
* shorts require a confirmed locate, else skip & log
* any order > 1% ADV is sliced into <=1% pieces, 5 min apart

The clock is injected so all timing is deterministic under test (no real
sleeping).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol

from src.execution.broker import BrokerClient
from src.execution.slippage_tracker import ExecutionLog, Fill

log = logging.getLogger("greymatter.execution")


class Clock(Protocol):
    def now(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


class RealClock:
    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


@dataclass(frozen=True)
class ExecConfig:
    poll_seconds: float = 30.0
    requote_after: float = 300.0     # 5 min -> re-quote at mid
    giveup_after: float = 900.0      # 15 min -> skip
    slice_spacing: float = 300.0     # 5 min between ADV slices
    max_adv_participation: float = 0.01  # 1% of ADV per slice


@dataclass
class OrderRequest:
    symbol: str
    side: str            # "buy" | "sell"
    qty: float           # shares (positive)
    adv_shares: float    # 20-day average daily volume, shares
    is_short: bool = False


class OrderManager:
    def __init__(self, broker: BrokerClient, clock: Clock | None = None,
                 cfg: ExecConfig | None = None) -> None:
        self.b = broker
        self.clock = clock or RealClock()
        self.cfg = cfg or ExecConfig()

    # ---- slicing -------------------------------------------------------
    def _slices(self, qty: float, adv: float) -> list[float]:
        cap = max(self.cfg.max_adv_participation * adv, 0.0)
        if cap <= 0 or qty <= cap:
            return [qty]
        n = int(-(-qty // cap))                  # ceil
        base = qty / n
        return [base] * n

    # ---- one (sliced) order -------------------------------------------
    def _work_one(self, req: OrderRequest, elog: ExecutionLog) -> None:
        if req.is_short and not self.b.can_locate(req.symbol, req.qty):
            elog.record_skip(req.symbol, "no locate for short")
            log.warning("skip %s: no locate", req.symbol)
            return

        for i, slice_qty in enumerate(self._slices(req.qty, req.adv_shares)):
            if i > 0:
                self.clock.sleep(self.cfg.slice_spacing)
            self._fill_slice(req, slice_qty, elog)

    def _fill_slice(self, req: OrderRequest, qty: float,
                    elog: ExecutionLog) -> None:
        q = self.b.get_quote(req.symbol)
        decision_price = q.mid
        half = q.spread / 2.0
        limit = q.mid + half if req.side == "buy" else q.mid - half

        order = self.b.submit_limit(req.symbol, qty, req.side, limit)
        start = self.clock.now()
        requoted = False

        while True:
            st = self.b.get_order(order.id)
            if st.status == "filled":
                elog.record_fill(Fill(
                    req.symbol, req.side, st.filled_qty,
                    decision_price, st.filled_avg_price, self.clock.now()))
                return
            elapsed = self.clock.now() - start
            if elapsed >= self.cfg.giveup_after:
                self.b.cancel(order.id)
                elog.record_skip(req.symbol, "unfilled after 15m")
                log.warning("skip %s: unfilled 15m", req.symbol)
                return
            if elapsed >= self.cfg.requote_after and not requoted:
                self.b.cancel(order.id)
                order = self.b.submit_limit(req.symbol, qty, req.side, q.mid)
                requoted = True
            self.clock.sleep(self.cfg.poll_seconds)

    # ---- public --------------------------------------------------------
    def execute(self, requests: list[OrderRequest]) -> ExecutionLog:
        elog = ExecutionLog()
        for req in requests:
            if req.qty <= 0:
                continue
            try:
                self._work_one(req, elog)
            except Exception as e:                 # never let one name halt the book
                elog.record_skip(req.symbol, f"error: {e!r}")
                log.exception("execution error %s", req.symbol)
        return elog
