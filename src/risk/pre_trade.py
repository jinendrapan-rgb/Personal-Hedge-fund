"""Pre-trade risk checks. Every proposed trade list must pass before it
reaches execution.

Borrow availability is queried through a ``LocateProvider`` protocol so
this is testable offline and live against Alpaca's locate API when the
broker key is set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd

from src.portfolio.constraints import Constraints


class LocateProvider(Protocol):
    def is_shortable(self, ticker: str) -> bool: ...


@dataclass
class CheckResult:
    ok: bool
    violations: list[str] = field(default_factory=list)

    def add(self, msg: str) -> None:
        self.violations.append(msg)
        self.ok = False


def pre_trade_check(
    target_w: pd.Series,
    sectors: pd.Series,
    betas: pd.Series,
    *,
    adv_usd: pd.Series,
    book_usd: float,
    locate: LocateProvider | None = None,
    c: Constraints | None = None,
    max_price_move_bps: float = 25.0,
) -> CheckResult:
    """Validate a target book against position/neutrality/liquidity/borrow."""
    c = c or Constraints()
    r = CheckResult(ok=True)
    w = target_w[target_w != 0]

    if (w.abs() > c.pos_max + 1e-6).any():
        bad = w.index[w.abs() > c.pos_max + 1e-6].tolist()
        r.add(f"position size > {c.pos_max:.1%}: {bad}")
    if w.abs().sum() > c.gross_max + 1e-6:
        r.add(f"gross {w.abs().sum():.2f} > {c.gross_max}")
    if not (c.net_min - 1e-6 <= w.sum() <= c.net_max + 1e-6):
        r.add(f"net {w.sum():+.3f} outside [{c.net_min}, {c.net_max}]")

    sec = sectors.reindex(w.index).fillna("UNKNOWN")
    for s in sec.unique():
        net = w[sec == s].sum()
        if abs(net) > c.sector_net_tol + 1e-6:
            r.add(f"sector {s} net {net:+.3f} > ±{c.sector_net_tol:.0%}")

    port_beta = (betas.reindex(w.index).fillna(0.0) * w).sum()
    if not (c.beta_min - 1e-6 <= port_beta <= c.beta_max + 1e-6):
        r.add(f"portfolio beta {port_beta:+.3f} outside band")

    # liquidity: trade must not move price more than max_price_move_bps,
    # using the same sqrt-impact assumption as the cost model.
    adv = adv_usd.reindex(w.index)
    trade_usd = w.abs() * book_usd
    participation = (trade_usd / adv).where(adv > 0, 0.0)
    impact_bps = 10.0 * (participation * 100.0) ** 0.5
    illiquid = impact_bps[impact_bps > max_price_move_bps].index.tolist()
    if illiquid:
        r.add(f"liquidity: est. impact > {max_price_move_bps}bps for {illiquid}")

    # borrow availability for shorts
    if locate is not None:
        for tk in w.index[w < 0]:
            if not locate.is_shortable(tk):
                r.add(f"no borrow/locate for short {tk}")
    return r
