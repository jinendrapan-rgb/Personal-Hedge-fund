"""Stress tests — replay current weights through historical crisis windows.

Empirical replay when the returns history covers a scenario window
(honest: actual constituent behaviour); otherwise a transparent linear
beta approximation. A market-neutral book should show small numbers —
that's the point of the construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# (start, end) inclusive. Replayed if covered by the supplied history.
SCENARIOS: dict[str, tuple[str, str]] = {
    "GFC_2008": ("2008-09-01", "2008-11-30"),
    "COVID_2020": ("2020-02-19", "2020-03-23"),
    "MOMENTUM_2022": ("2022-08-16", "2022-09-30"),
    "SVB_2023": ("2023-03-08", "2023-03-15"),
}


@dataclass(frozen=True)
class ScenarioResult:
    scenario: str
    method: str          # "empirical" | "linear"
    portfolio_return: float
    note: str = ""


def _empirical(weights: pd.Series, hist: pd.DataFrame, s: str, e: str) -> float | None:
    win = hist.loc[(hist.index >= pd.Timestamp(s)) & (hist.index <= pd.Timestamp(e))]
    cols = [c for c in weights.index if c in win.columns]
    if win.empty or not cols:
        return None
    # hold fixed weights across the window
    cum = (1 + win[cols].fillna(0.0)).prod() - 1.0
    return float((weights[cols] * cum).sum())


def run_stress(
    weights: pd.Series,
    returns_history: pd.DataFrame,
    *,
    portfolio_beta: float = 0.0,
    market_shocks: dict[str, float] | None = None,
) -> list[ScenarioResult]:
    market_shocks = market_shocks or {
        "GFC_2008": -0.30, "COVID_2020": -0.34,
        "MOMENTUM_2022": -0.09, "SVB_2023": -0.05,
    }
    out: list[ScenarioResult] = []
    held = weights[weights != 0]
    for name, (s, e) in SCENARIOS.items():
        emp = _empirical(held, returns_history, s, e)
        if emp is not None:
            out.append(ScenarioResult(name, "empirical", round(emp, 4)))
        else:
            shock = market_shocks.get(name, -0.10)
            approx = portfolio_beta * shock
            out.append(ScenarioResult(
                name, "linear", round(approx, 4),
                note="history did not cover window; beta×shock approximation"))
    return out
