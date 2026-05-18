"""Quality — Piotroski F-Score + Asness/Frazzini/Pedersen QMJ elements.

Pure over two point-in-time fundamentals dicts (current fiscal year and
one year prior, both from ``xbrl_parser.fundamentals_as_of``). Returns
sub-signals already oriented so **higher = better**:

* ``f_score``   : Piotroski checks that are computable from available tags,
                  reported as a *fraction* (0..1) of checks that passed so
                  missing balance-sheet tags don't penalise a name.
* ``gross_prof``: gross profit / total assets (Novy-Marx).
* ``neg_accruals``: -(net income - operating cash flow)/assets
                  (high accruals = low earnings quality -> negate).
"""
from __future__ import annotations

import math
from typing import Mapping

Num = float | int | None


def _ok(x: Num) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _ratio(a: Num, b: Num) -> float | None:
    if _ok(a) and _ok(b) and b != 0:
        return a / b
    return None


def quality_signals(
    curr: Mapping[str, Num], prev: Mapping[str, Num]
) -> dict[str, float | None]:
    checks: list[bool] = []

    roa = _ratio(curr.get("net_income"), curr.get("total_assets"))
    roa_p = _ratio(prev.get("net_income"), prev.get("total_assets"))
    cfo = curr.get("operating_cash_flow")
    ni = curr.get("net_income")

    if _ok(roa):
        checks.append(roa > 0)                     # 1 profitable
    if _ok(cfo):
        checks.append(cfo > 0)                     # 2 positive CFO
    if _ok(roa) and _ok(roa_p):
        checks.append(roa > roa_p)                 # 3 rising ROA
    if _ok(cfo) and _ok(ni):
        checks.append(cfo > ni)                    # 4 accrual quality

    ltd = _ratio(curr.get("long_term_debt"), curr.get("total_assets"))
    ltd_p = _ratio(prev.get("long_term_debt"), prev.get("total_assets"))
    if _ok(ltd) and _ok(ltd_p):
        checks.append(ltd < ltd_p)                 # 5 lower leverage

    cr = _ratio(curr.get("current_assets"), curr.get("current_liabilities"))
    cr_p = _ratio(prev.get("current_assets"), prev.get("current_liabilities"))
    if _ok(cr) and _ok(cr_p):
        checks.append(cr > cr_p)                   # 6 better liquidity

    so, so_p = curr.get("shares_outstanding"), prev.get("shares_outstanding")
    if _ok(so) and _ok(so_p):
        checks.append(so <= so_p)                  # 7 no dilution

    gm = _ratio(curr.get("gross_profit"), curr.get("revenue"))
    gm_p = _ratio(prev.get("gross_profit"), prev.get("revenue"))
    if _ok(gm) and _ok(gm_p):
        checks.append(gm > gm_p)                   # 8 rising gross margin

    at = _ratio(curr.get("revenue"), curr.get("total_assets"))
    at_p = _ratio(prev.get("revenue"), prev.get("total_assets"))
    if _ok(at) and _ok(at_p):
        checks.append(at > at_p)                   # 9 rising asset turnover

    f_score = (sum(checks) / len(checks)) if checks else None

    gross_prof = _ratio(curr.get("gross_profit"), curr.get("total_assets"))

    acc = None
    if _ok(ni) and _ok(cfo) and _ok(curr.get("total_assets")):
        acc = -((ni - cfo) / curr["total_assets"])

    return {"f_score": f_score, "gross_prof": gross_prof, "neg_accruals": acc}
