"""Daily pipeline orchestrator — the connective tissue across all 8 layers.

Workflow (spec):
  Data -> Factor -> [AI overlay, OFF by default] -> Portfolio -> Risk
       -> Execution (paper unless live-confirmed) -> Dashboard artifacts

Every stage is guarded: a missing credential or data-plan limitation
degrades to a clear, actionable status rather than a stack trace. The
pipeline is dependency-injected so it runs live with real providers and
is integration-tested end-to-end with fakes.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from src.config import settings

log = logging.getLogger("greymatter.pipeline")


@dataclass
class Preflight:
    sec: bool
    polygon_key: bool
    ai_key: bool
    alpaca: bool
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"sec": self.sec, "polygon_key": self.polygon_key,
                "ai_key": self.ai_key, "alpaca": self.alpaca,
                "notes": self.notes}


def preflight() -> Preflight:
    pf = Preflight(
        sec=bool(settings.sec_user_agent),
        polygon_key=bool(settings.polygon_api_key),
        ai_key=bool(settings.openai_api_key or settings.anthropic_api_key),
        alpaca=bool(settings.alpaca_api_key and settings.alpaca_secret_key),
    )
    if not pf.polygon_key:
        pf.notes.append("No POLYGON_API_KEY — prices unavailable; factor "
                        "engine and backtest cannot run.")
    else:
        pf.notes.append("Polygon key present — factor history needs a "
                        "Starter+ plan (recent-only tiers fail with 403).")
    if not pf.ai_key:
        pf.notes.append("No AI key — forensic overlay unavailable (ok: "
                         "overlay is OFF by default).")
    if not pf.alpaca:
        pf.notes.append("No Alpaca keys — execution runs in dry-run "
                         "(targets logged, no orders).")
    return pf


@dataclass
class PipelineResult:
    as_of: str
    stages: dict[str, str] = field(default_factory=dict)
    target: pd.DataFrame = field(default_factory=pd.DataFrame)
    artifacts: list[str] = field(default_factory=list)

    def ok(self) -> bool:
        """Healthy unless a stage failed/blocked/empty. 'off (default)' and
        'dry-run' are intended states, not failures."""
        bad = ("failed", "blocked", "empty")
        return bool(self.stages) and not any(
            str(v).startswith(bad) for v in self.stages.values()
        )


class DailyPipeline:
    def __init__(
        self, dm, factor_engine, optimiser, *,
        ai_overlay=None, sectors=None, betas=None,
        executor=None, enable_ai_overlay: bool = False,
        min_factors: int = 5,
    ) -> None:
        self.dm = dm
        self.fe = factor_engine
        self.opt = optimiser
        self.ai = ai_overlay
        self.sectors = sectors
        self.betas = betas
        self.executor = executor
        self.enable_ai = enable_ai_overlay   # spec: OFF by default
        # Spec is a 5-factor system; require all 5 by default before a
        # portfolio is built. Lower only deliberately, eyes open.
        self.min_factors = min_factors

    def run(self, as_of: str | None = None,
            universe: list[str] | None = None) -> PipelineResult:
        as_of = as_of or date.today().isoformat()
        res = PipelineResult(as_of=as_of)
        out = settings.paths.data

        # 1. universe
        try:
            uni = universe or self.dm.get_universe(as_of)
            res.stages["universe"] = "ok" if uni else "empty"
        except Exception as e:
            res.stages["universe"] = f"failed: {e!r}"
            return res

        # 2. factor scores — with a coverage gate. The spec's whole thesis
        # is "never trade a silently broken signal": if factors are missing
        # because price history is plan-blocked, we must NOT build a book
        # on a crippled composite and call it success.
        try:
            from src.factors.factor_engine import FACTORS

            scores = self.fe.compute(as_of, universe=uni)
            if scores.empty:
                res.stages["factors"] = "empty (no price history available)"
                return res
            scores.to_parquet(out / "factor_scores.parquet")
            res.artifacts.append("factor_scores.parquet")

            covered = [f for f in FACTORS
                       if f in scores and scores[f].notna().any()]
            missing = [f for f in FACTORS if f not in covered]
            if len(covered) < self.min_factors:
                res.stages["factors"] = (
                    f"blocked: only {covered} have data; missing {missing} "
                    f"(need >= {self.min_factors}). Likely Polygon plan lacks "
                    f"price history — refusing to construct a portfolio on a "
                    f"degraded signal.")
                return res
            res.stages["factors"] = (
                "ok" if not missing
                else f"degraded: missing {missing} but {len(covered)} "
                     f">= {self.min_factors} covered — proceeding with caveat")
        except Exception as e:
            res.stages["factors"] = f"failed: {e!r}"
            return res

        # 3. AI overlay (OFF by default per spec)
        if self.enable_ai and self.ai is not None:
            try:
                scores = self.ai(scores)
                res.stages["ai_overlay"] = "ok (applied)"
            except Exception as e:
                res.stages["ai_overlay"] = f"failed: {e!r}"
        else:
            res.stages["ai_overlay"] = "off (default)"

        # 4. portfolio construction
        try:
            hist = self._returns_history(list(scores["ticker"]), as_of)
            alpha = scores.set_index("ticker")["composite"]
            w = self.opt.build(alpha, hist, self.sectors, self.betas,
                               mode="mvo")
            tgt = (w[w != 0].rename_axis("ticker")
                   .reset_index(name="weight"))
            tgt["date"] = as_of
            tgt.to_parquet(out / "target_portfolio.parquet")
            res.target = tgt
            res.artifacts.append("target_portfolio.parquet")
            res.stages["portfolio"] = "ok"
        except Exception as e:
            res.stages["portfolio"] = f"failed: {e!r}"
            return res

        # 5. risk gate
        try:
            from src.risk.pre_trade import pre_trade_check
            adv = pd.Series(1e12, index=tgt["ticker"])  # placeholder ADV
            chk = pre_trade_check(
                tgt.set_index("ticker")["weight"], self.sectors, self.betas,
                adv_usd=adv, book_usd=1e7)
            res.stages["risk"] = "ok" if chk.ok else \
                f"blocked: {chk.violations}"
            if not chk.ok:
                return res
        except Exception as e:
            res.stages["risk"] = f"failed: {e!r}"
            return res

        # 6. execution (dry-run unless an executor is wired)
        if self.executor is not None:
            try:
                self.executor(tgt)
                res.stages["execution"] = "ok"
            except Exception as e:
                res.stages["execution"] = f"failed: {e!r}"
        else:
            res.stages["execution"] = "dry-run (targets written, no orders)"

        (out / "pipeline_status.json").write_text(
            json.dumps({"as_of": as_of, "stages": res.stages}, indent=2))
        res.artifacts.append("pipeline_status.json")
        return res

    def _returns_history(self, tickers, as_of) -> pd.DataFrame:
        start = (pd.Timestamp(as_of) - pd.Timedelta(days=400)).date().isoformat()
        cols = {}
        for t in tickers:
            try:
                px = self.dm.get_prices(t, start, as_of)
                if not px.empty:
                    cols[t] = px["close"].pct_change()
            except Exception:
                continue
        return pd.DataFrame(cols).dropna(how="all")
