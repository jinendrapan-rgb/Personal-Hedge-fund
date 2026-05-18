"""Mean-variance optimisation with a turnover penalty.

    maximise   alphaᵀw  -  risk_aversion · wᵀΣw  -  turnover_penalty · ‖w - w_prev‖₁
    subject to the Constraints set (gross/net/sector/beta/liquidity box)

The turnover penalty's L1 form makes round-trip cost explicit and keeps
the book from churning on noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from src.portfolio.constraints import Constraints, build_constraints


@dataclass(frozen=True)
class MVOConfig:
    risk_aversion: float = 8.0
    turnover_penalty: float = 0.0      # L1 coefficient on |w - w_prev|


class InfeasibleError(RuntimeError):
    """Raised when the solver cannot satisfy the constraints (spec: loosen
    sector ±2%→±5% or beta ±0.1→±0.2)."""


def optimise(
    alpha: pd.Series,
    cov: pd.DataFrame,
    sectors: pd.Series,
    betas: pd.Series,
    *,
    prev_w: pd.Series | None = None,
    adv_cap: pd.Series | None = None,
    constraints: Constraints | None = None,
    config: MVOConfig | None = None,
) -> pd.Series:
    c = constraints or Constraints()
    cfg = config or MVOConfig()

    tickers = list(alpha.dropna().index)
    a = alpha.reindex(tickers).values
    S = cov.reindex(index=tickers, columns=tickers).values
    S = (S + S.T) / 2.0

    w = cp.Variable(len(tickers))
    obj = a @ w - cfg.risk_aversion * cp.quad_form(w, cp.psd_wrap(S))
    if cfg.turnover_penalty > 0:
        pw = (prev_w.reindex(tickers).fillna(0.0).values
              if prev_w is not None else np.zeros(len(tickers)))
        obj = obj - cfg.turnover_penalty * cp.norm1(w - pw)

    cons = build_constraints(
        w, tickers=tickers, sectors=sectors, betas=betas,
        adv_cap=adv_cap, c=c,
    )
    prob = cp.Problem(cp.Maximize(obj), cons)
    # Prefer ECOS/CLARABEL if present; otherwise let cvxpy choose. This
    # keeps the solve working across environments.
    for solver in ("ECOS", "CLARABEL", "SCS"):
        if solver in cp.installed_solvers():
            prob.solve(solver=getattr(cp, solver))
            break
    else:
        prob.solve()
    if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        raise InfeasibleError(
            f"solver status={prob.status}; loosen constraints (sector "
            f"±{c.sector_net_tol:.0%} or beta band) and retry."
        )
    return pd.Series(np.asarray(w.value).ravel(), index=tickers, name="weight")
