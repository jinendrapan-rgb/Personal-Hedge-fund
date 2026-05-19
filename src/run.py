"""GreyMatter CLI.

    uv run python -m src.run preflight
    uv run python -m src.run daily   [--as-of YYYY-MM-DD] [--ai] [--tickers A,B]
    uv run python -m src.run backtest --start YYYY-MM-DD --end YYYY-MM-DD

`daily` wires the real providers; it degrades with clear status when a
data plan or credential is missing (it will not crash).
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from src.pipeline import DailyPipeline, preflight


def _cmd_preflight(_args) -> int:
    print(json.dumps(preflight().as_dict(), indent=2))
    return 0


def _cmd_daily(args) -> int:
    from src.data.data_manager import DataManager
    from src.factors.factor_engine import FactorEngine
    from src.factors.sectors import SectorProvider
    from src.portfolio.optimiser import Optimiser

    dm = DataManager()
    tickers = args.tickers.split(",") if args.tickers else None
    sectors = SectorProvider().get(tickers) if tickers else pd.Series(dtype=str)
    betas = pd.Series(1.0, index=tickers) if tickers else pd.Series(dtype=float)

    pipe = DailyPipeline(
        dm, FactorEngine(dm), Optimiser(),
        sectors=sectors, betas=betas, enable_ai_overlay=args.ai,
    )
    res = pipe.run(as_of=args.as_of, universe=tickers)
    print(json.dumps({"as_of": res.as_of, "stages": res.stages,
                      "artifacts": res.artifacts, "ok": res.ok()}, indent=2))
    return 0 if res.ok() else 1


def _cmd_backtest(args) -> int:
    from src.backtest.engine import WalkForwardBacktest
    from src.data.data_manager import DataManager
    from src.factors.factor_engine import FactorEngine

    dm = DataManager()
    fe = FactorEngine(dm)

    def score_fn(as_of, _tickers):
        df = fe.compute(as_of)
        return df.set_index("ticker")["composite"] if not df.empty else \
            pd.Series(dtype=float)

    bt = WalkForwardBacktest(dm, dm.get_universe, score_fn)
    try:
        res = bt.run(args.start, args.end)
        print(json.dumps(res.metrics, indent=2))
        print("artifacts:", res.out_dir)
        return 0
    except Exception as e:
        print(f"backtest could not run: {e!r}")
        print("Most likely cause: Polygon plan lacks the required history "
              "depth (Starter+ needed).")
        return 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="greymatter")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight").set_defaults(fn=_cmd_preflight)

    d = sub.add_parser("daily")
    d.add_argument("--as-of", default=None)
    d.add_argument("--tickers", default=None, help="comma-separated override")
    d.add_argument("--ai", action="store_true", help="enable AI overlay")
    d.set_defaults(fn=_cmd_daily)

    b = sub.add_parser("backtest")
    b.add_argument("--start", required=True)
    b.add_argument("--end", required=True)
    b.set_defaults(fn=_cmd_backtest)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
