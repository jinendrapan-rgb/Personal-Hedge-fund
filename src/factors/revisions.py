"""Earnings Revisions.

3-month change in consensus FY1 EPS estimate plus revision breadth
(% analysts up minus % down). Pure over two estimate snapshots (now and
~3 months ago); the engine sources these from a provider (Polygon analyst
estimates on the live path). Both sub-signals: **higher = better**.
"""
from __future__ import annotations

import math
from typing import Mapping

Num = float | int | None


def _ok(x: Num) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def revisions_signals(
    now: Mapping[str, Num], prior: Mapping[str, Num]
) -> dict[str, float | None]:
    """``now``/``prior`` keys: ``fy1_eps``, ``n_up``, ``n_down``, ``n_total``."""
    e_now, e_old = now.get("fy1_eps"), prior.get("fy1_eps")
    eps_change: float | None = None
    if _ok(e_now) and _ok(e_old) and e_old != 0:
        eps_change = (e_now - e_old) / abs(e_old)

    breadth: float | None = None
    n_up, n_down = now.get("n_up"), now.get("n_down")
    n_total = now.get("n_total")
    if _ok(n_up) and _ok(n_down) and _ok(n_total) and n_total > 0:
        breadth = (n_up - n_down) / n_total

    return {"eps_change": eps_change, "breadth": breadth}
