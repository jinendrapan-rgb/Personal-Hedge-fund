"""Broker abstraction.

``BrokerClient`` is the Protocol the order manager talks to — so the
execution algorithm is testable offline against a fake and runs live
against Alpaca unchanged. ``AlpacaBroker`` defaults to the paper endpoint;
live requires BOTH ``ALPACA_LIVE=true`` and an explicit confirmation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return max(self.ask - self.bid, 0.0)


@dataclass(frozen=True)
class OrderState:
    id: str
    symbol: str
    qty: float
    side: str            # "buy" | "sell"
    limit_price: float
    status: str          # "new" | "filled" | "partially_filled" | "canceled"
    filled_qty: float = 0.0
    filled_avg_price: float = 0.0


class BrokerClient(Protocol):
    def get_quote(self, symbol: str) -> Quote: ...
    def can_locate(self, symbol: str, qty: float) -> bool: ...
    def submit_limit(self, symbol: str, qty: float, side: str,
                     limit_price: float) -> OrderState: ...
    def get_order(self, order_id: str) -> OrderState: ...
    def cancel(self, order_id: str) -> None: ...


def guard_live(live: bool, confirm: bool) -> None:
    """Refuse real-money trading unless explicitly confirmed (spec)."""
    if live and not confirm:
        raise RuntimeError(
            "ALPACA_LIVE=true but confirm_live not passed. Refusing to "
            "trade real money without explicit confirmation."
        )


class AlpacaBroker:
    """Live/paper Alpaca adapter (thin; imports SDK lazily)."""

    def __init__(self, confirm_live: bool = False) -> None:
        from src.config import settings

        settings.require("alpaca_api_key", "alpaca_secret_key")
        guard_live(settings.alpaca_live, confirm_live)
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient

        self._paper = not settings.alpaca_live
        self._t = TradingClient(
            settings.alpaca_api_key, settings.alpaca_secret_key,
            paper=self._paper,
        )
        self._d = StockHistoricalDataClient(
            settings.alpaca_api_key, settings.alpaca_secret_key
        )

    def get_quote(self, symbol: str) -> Quote:
        from alpaca.data.requests import StockLatestQuoteRequest

        q = self._d.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=symbol)
        )[symbol]
        return Quote(bid=float(q.bid_price), ask=float(q.ask_price))

    def can_locate(self, symbol: str, qty: float) -> bool:
        try:
            a = self._t.get_asset(symbol)
            return bool(a.shortable and a.easy_to_borrow)
        except Exception:
            return False

    def submit_limit(self, symbol, qty, side, limit_price) -> OrderState:
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        o = self._t.submit_order(LimitOrderRequest(
            symbol=symbol, qty=abs(qty),
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            limit_price=round(limit_price, 2),
        ))
        return self._map(o)

    def get_order(self, order_id: str) -> OrderState:
        return self._map(self._t.get_order_by_id(order_id))

    def cancel(self, order_id: str) -> None:
        try:
            self._t.cancel_order_by_id(order_id)
        except Exception:
            pass

    @staticmethod
    def _map(o) -> OrderState:
        return OrderState(
            id=str(o.id), symbol=o.symbol, qty=float(o.qty),
            side=str(o.side).lower().split(".")[-1],
            limit_price=float(o.limit_price or 0.0),
            status=str(o.status).lower().split(".")[-1],
            filled_qty=float(o.filled_qty or 0.0),
            filled_avg_price=float(o.filled_avg_price or 0.0),
        )
