"""Main portfolio interface: MVO (default) or Black-Litterman, with the
non-convex 0.5% floor handled by a drop-and-resolve pass.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.portfolio.black_litterman import black_litterman
from src.portfolio.constraints import Constraints, enforce_min_position
from src.portfolio.covariance import ledoit_wolf_cov
from src.portfolio.mvo import MVOConfig, optimise

log = logging.getLogger("greymatter.portfolio")


class Optimiser:
    def __init__(
        self,
        constraints: Constraints | None = None,
        config: MVOConfig | None = None,
    ) -> None:
        self.c = constraints or Constraints()
        self.cfg = config or MVOConfig()

    def build(
        self,
        alpha: pd.Series,
        returns_history: pd.DataFrame,
        sectors: pd.Series,
        betas: pd.Series,
        *,
        prev_w: pd.Series | None = None,
        adv_cap: pd.Series | None = None,
        mode: str = "mvo",
        market_weights: pd.Series | None = None,
    ) -> pd.Series:
        cov = ledoit_wolf_cov(returns_history)
        universe = [t for t in alpha.dropna().index if t in cov.index]
        alpha = alpha.reindex(universe)
        cov = cov.reindex(index=universe, columns=universe)

        if mode == "bl":
            mw = (market_weights.reindex(universe).fillna(0.0)
                  if market_weights is not None
                  else pd.Series(1.0 / len(universe), index=universe))
            # standardise scores into a modest return-view scale
            v = alpha - alpha.mean()
            sd = v.std(ddof=0)
            views = (v / sd * 0.02) if sd > 0 else v * 0.0
            exp_ret = black_litterman(cov, mw, views)
        elif mode == "mvo":
            exp_ret = alpha
        else:
            raise ValueError(f"unknown mode {mode!r} (use 'mvo' or 'bl')")

        w = optimise(exp_ret, cov, sectors, betas, prev_w=prev_w,
                     adv_cap=adv_cap, constraints=self.c, config=self.cfg)

        cleaned = enforce_min_position(w, self.c)
        survivors = list(cleaned[cleaned != 0].index)
        if survivors and len(survivors) < len(w):
            # Re-solve on the surviving support so constraints still hold.
            w = optimise(
                exp_ret.reindex(survivors), cov.reindex(survivors, axis=0)
                .reindex(survivors, axis=1),
                sectors, betas,
                prev_w=prev_w, adv_cap=adv_cap,
                constraints=self.c, config=self.cfg,
            )
            w = enforce_min_position(w, self.c)
        return w.reindex(alpha.index).fillna(0.0).rename("weight")
