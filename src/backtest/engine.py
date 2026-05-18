"""Walk-forward backtest engine.

Design notes (and an honest interpretation of the spec):

* The signal is an equal-weight composite z-score — there are no fitted
  parameters, so "36m train / 3m test / 3m step" reduces in practice to:
  rebalance monthly, but only once at least ``min_history_months`` of
  history exists before the rebalance date (so momentum/quality have their
  lookbacks). Each month's target is held to the next rebalance.
* Point-in-time correctness is delegated to ``score_fn`` (Layer 2 over the
  Layer 1 PIT DataManager). The engine itself only ever uses prices AFTER
  a rebalance date to compute the *realised* P&L of positions already
  chosen — never to choose them. The lookahead test asserts that chosen
  positions are invariant to any future data.
* Survivorship-free: ``universe_fn(date)`` must return the membership as
  of that date, including names later delisted. Iterating that set is what
  keeps the backtest honest.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtest.costs import ZERO_COST, CostModel
from src.backtest.metrics import summary
from src.backtest.portfolio import build_long_short
from src.backtest.tearsheet import write_tearsheet
from src.config import settings

log = logging.getLogger("greymatter.backtest")


@dataclass
class BacktestResult:
    run_id: str
    returns: pd.DataFrame          # date x [long, short, gross, cost, total]
    positions: pd.DataFrame        # rebalance_date x ticker (target weights)
    trades: pd.DataFrame           # rebalance_date, ticker, traded, cost_*
    metrics: dict
    out_dir: str = ""


class WalkForwardBacktest:
    def __init__(
        self,
        price_provider,
        universe_fn,
        score_fn,
        cost_model: CostModel | None = None,
        benchmark: str = "SPY",
        min_history_months: int = 36,
        decile: float = 0.10,
        min_names: int = 20,
    ) -> None:
        self.px = price_provider
        self.universe_fn = universe_fn
        self.score_fn = score_fn
        self.cost = cost_model if cost_model is not None else CostModel()
        self.benchmark = benchmark
        self.min_names = min_names
        self.min_history_months = min_history_months
        self.decile = decile

    # ---- price panel ---------------------------------------------------
    def _panel(self, tickers, start, end):
        close, vol = {}, {}
        for t in sorted(set(tickers)):
            try:
                df = self.px.get_prices(t, start, end)
            except Exception as e:
                log.warning("price fetch failed %s: %r", t, e)
                continue
            if df is not None and not df.empty:
                close[t] = df["close"]
                if "volume" in df:
                    vol[t] = df["close"] * df["volume"]  # dollar volume
        c = pd.DataFrame(close).sort_index()
        v = pd.DataFrame(vol).sort_index() if vol else pd.DataFrame(index=c.index)
        return c, v

    def run(self, start: str, end: str, run_id: str | None = None) -> BacktestResult:
        run_id = run_id or uuid.uuid4().hex[:10]
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

        rebals = pd.date_range(start_ts, end_ts, freq="BME")
        # Collect every ticker that is ever a member across the run.
        universe_by_date = {d: list(self.universe_fn(d.date().isoformat()))
                            for d in rebals}
        all_tickers = sorted({t for v in universe_by_date.values() for t in v})
        buffer_start = (start_ts - pd.Timedelta(days=40 * 30)).date().isoformat()
        close, dvol = self._panel(all_tickers, buffer_start, end)
        if close.empty:
            raise RuntimeError("no price data for any ticker in the universe")

        rets = close.pct_change()
        adv = dvol.rolling(20, min_periods=5).mean() if not dvol.empty else None

        daily_rows: list[pd.Series] = []
        pos_rows: dict[pd.Timestamp, pd.Series] = {}
        trade_recs: list[dict] = []
        prev_w = pd.Series(dtype=float)

        for i, d in enumerate(rebals):
            hist = close.loc[close.index <= d]
            if len(hist) < self.min_history_months * 21:
                continue
            tickers = [t for t in universe_by_date[d] if t in close.columns
                       and pd.notna(close[t].reindex([hist.index[-1]]).iloc[0])]
            if len(tickers) < self.min_names:
                continue

            scores = pd.Series(self.score_fn(d.date().isoformat(), tickers)) \
                .reindex(tickers).astype(float)
            w = build_long_short(scores, self.decile)
            pos_rows[d] = w[w != 0]

            # turnover & rebalance cost on day d
            aligned = w.reindex(w.index.union(prev_w.index)).fillna(0.0)
            prev_al = prev_w.reindex(aligned.index).fillna(0.0)
            traded = aligned - prev_al
            adv_d = (adv.loc[d].reindex(aligned.index)
                     if adv is not None and d in adv.index
                     else pd.Series(np.inf, index=aligned.index))
            sp = self.cost.spread_cost(traded, adv_d)
            im = self.cost.impact_cost(traded, book_usd=1.0, adv_usd=adv_d)
            for tk, tv in traded[traded != 0].items():
                trade_recs.append({"rebalance": d, "ticker": tk,
                                    "traded_weight": float(tv)})

            # holding window [d, next_d)
            nxt = rebals[i + 1] if i + 1 < len(rebals) else end_ts
            window = rets.loc[(rets.index > d) & (rets.index <= nxt)]
            longs = w[w > 0]
            shorts = w[w < 0]
            short_abs = float(shorts.abs().sum())
            for j, (dt_, row) in enumerate(window.iterrows()):
                lr = float((longs * row.reindex(longs.index)).sum())
                sr = float((shorts * row.reindex(shorts.index)).sum())
                cost = self.cost.borrow_cost_daily(short_abs)
                if j == 0:
                    cost += sp + im
                daily_rows.append(pd.Series(
                    {"long": lr, "short": sr, "gross": lr + sr,
                     "cost": cost, "total": lr + sr - cost}, name=dt_))
            prev_w = w[w != 0]

        if not daily_rows:
            raise RuntimeError("no rebalances had sufficient history")
        returns = pd.DataFrame(daily_rows)
        returns.index.name = "date"
        positions = pd.DataFrame(pos_rows).T.sort_index()
        positions.index.name = "rebalance"
        trades = pd.DataFrame(trade_recs)

        bench = self._benchmark_returns(returns.index, buffer_start, end)
        m = summary(returns["total"], returns["long"], returns["short"],
                    bench, positions)
        m["n_rebalances"] = int(len(positions))
        m["start"], m["end"] = str(returns.index[0].date()), str(returns.index[-1].date())

        result = BacktestResult(run_id, returns, positions, trades, m)
        self._persist(result, bench)
        return result

    def _benchmark_returns(self, idx, start, end) -> pd.Series:
        try:
            b = self.px.get_prices(self.benchmark, start, end)
            br = b["close"].pct_change()
            return br.reindex(idx).fillna(0.0)
        except Exception:
            return pd.Series(0.0, index=idx)

    def _persist(self, res: BacktestResult, bench: pd.Series) -> None:
        out = settings.paths.root / "backtests" / res.run_id
        out.mkdir(parents=True, exist_ok=True)
        res.returns.to_parquet(out / "returns.parquet")
        res.positions.to_parquet(out / "positions.parquet")
        if not res.trades.empty:
            res.trades.to_parquet(out / "trades.parquet")
        (out / "metrics.json").write_text(json.dumps(res.metrics, indent=2))
        write_tearsheet(out / "tearsheet.html", res.returns, bench, res.metrics)
        res.out_dir = str(out)
        log.info("backtest %s -> %s", res.run_id, out)
