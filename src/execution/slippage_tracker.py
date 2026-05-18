"""Slippage logging and daily summary.

slippage_bps = (fill_price - decision_price) / decision_price * 1e4,
signed so POSITIVE always means worse-than-decision (a cost): for sells
the sign is flipped accordingly.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: str
    qty: float
    decision_price: float
    fill_price: float
    timestamp: float

    @property
    def slippage_bps(self) -> float:
        if self.decision_price <= 0:
            return 0.0
        raw = (self.fill_price - self.decision_price) / self.decision_price * 1e4
        return raw if self.side == "buy" else -raw


@dataclass
class ExecutionLog:
    fills: list[Fill] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (symbol, reason)

    def record_fill(self, f: Fill) -> None:
        self.fills.append(f)

    def record_skip(self, symbol: str, reason: str) -> None:
        self.skipped.append((symbol, reason))

    def daily_summary(self) -> dict:
        n = len(self.fills)
        total_slip = sum(f.slippage_bps for f in self.fills)
        attempted = n + len(self.skipped)
        return {
            "trades": n,
            "skipped": len(self.skipped),
            "fill_rate": round(n / attempted, 4) if attempted else 0.0,
            "total_slippage_bps": round(total_slip, 2),
            "avg_slippage_bps": round(total_slip / n, 2) if n else 0.0,
        }
