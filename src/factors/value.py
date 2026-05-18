"""Value — multi-metric composite (no raw P/E; it's noisy & sector-distorted).

Sub-metrics, all oriented so **higher = cheaper = more attractive**:

* ``ev_ebit_inv`` :  EBIT / EV   (inverse of EV/EBIT)
* ``bp``          :  book / price (inverse of P/B)
* ``fcf_yield``   :  free cash flow / EV

EV = market cap + long-term debt - cash. FCF = operating cash flow - capex
(falls back to OCF if capex tag is absent — documented approximation).
Pure over one PIT fundamentals dict plus a market cap.
"""
from __future__ import annotations

import math
from typing import Mapping

Num = float | int | None


def _ok(x: Num) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _safe_div(a: Num, b: Num) -> float | None:
    if _ok(a) and _ok(b) and b != 0:
        return a / b
    return None


def value_signals(
    fund: Mapping[str, Num], market_cap: Num
) -> dict[str, float | None]:
    if not _ok(market_cap) or market_cap <= 0:
        return {"ev_ebit_inv": None, "bp": None, "fcf_yield": None}

    ltd = fund.get("long_term_debt") if _ok(fund.get("long_term_debt")) else 0.0
    cash = fund.get("cash") if _ok(fund.get("cash")) else 0.0
    ev = market_cap + ltd - cash
    ev = ev if ev > 0 else None

    ebit = fund.get("operating_income")
    ev_ebit_inv = _safe_div(ebit, ev)  # EBIT/EV — higher = cheaper

    bp = _safe_div(fund.get("stockholders_equity"), market_cap)  # book/price

    ocf = fund.get("operating_cash_flow")
    capex = fund.get("capex") if _ok(fund.get("capex")) else 0.0
    fcf = (ocf - abs(capex)) if _ok(ocf) else None
    fcf_yield = _safe_div(fcf, ev)

    return {"ev_ebit_inv": ev_ebit_inv, "bp": bp, "fcf_yield": fcf_yield}
