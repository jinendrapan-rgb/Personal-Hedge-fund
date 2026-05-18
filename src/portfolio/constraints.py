"""Portfolio constraint set and the cvxpy constraint builder.

Note on the 0.5% minimum position: a hard lower bound on |w_i| is
non-convex (combinatorial). We enforce it as a post-optimisation cleanup
(drop sub-threshold names, re-solve) rather than a MIP — documented, and
consistent with how the spec's error table treats feasibility.
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Constraints:
    sector_net_tol: float = 0.02      # |net weight per sector| <= 2%
    beta_min: float = -0.10
    beta_max: float = 0.10
    pos_max: float = 0.04             # max |weight| per name
    pos_min: float = 0.005            # post-process floor (non-convex)
    max_adv_frac: float = 0.05        # size <= 5% of 20d ADV
    gross_max: float = 2.00           # 100% long + 100% short
    net_min: float = -0.10
    net_max: float = 0.10


def build_constraints(
    w: cp.Variable,
    *,
    tickers: list[str],
    sectors: pd.Series,
    betas: pd.Series,
    adv_cap: pd.Series | None,
    c: Constraints,
) -> list:
    """cvxpy constraints for weight vector ``w`` aligned to ``tickers``."""
    cons = [cp.sum(w) >= c.net_min, cp.sum(w) <= c.net_max,
            cp.norm1(w) <= c.gross_max]

    # per-name box, tightened by liquidity where ADV cap is provided
    ub = np.full(len(tickers), c.pos_max)
    if adv_cap is not None:
        liq = adv_cap.reindex(tickers).fillna(c.pos_max).clip(upper=c.pos_max)
        ub = np.minimum(ub, liq.values)
    cons += [w <= ub, w >= -ub]

    # sector net-neutral within tolerance
    sec = sectors.reindex(tickers).fillna("UNKNOWN")
    for s in sec.unique():
        mask = (sec.values == s).astype(float)
        cons += [mask @ w <= c.sector_net_tol,
                 mask @ w >= -c.sector_net_tol]

    # beta-neutral band
    b = betas.reindex(tickers).fillna(0.0).values
    cons += [b @ w >= c.beta_min, b @ w <= c.beta_max]
    return cons


def enforce_min_position(w: pd.Series, c: Constraints) -> pd.Series:
    """Zero out names below the 0.5% floor (post-process, non-convex part)."""
    return w.where(w.abs() >= c.pos_min, 0.0)
