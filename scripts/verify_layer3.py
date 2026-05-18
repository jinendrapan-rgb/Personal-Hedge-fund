"""Layer-3 verification: a full walk-forward backtest, end to end.

Universe/prices are deterministic SYNTHETIC (Polygon key needed for real
data) with a mild planted signal so the L/S produces a modest, believable
Sharpe — not a leaky 2.0+. Prints the tear-sheet metrics and the on-disk
artifact paths so the run is independently inspectable.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.backtest.costs import CostModel
from src.backtest.engine import WalkForwardBacktest

IDX = pd.bdate_range("2015-01-01", "2023-12-31")
N = 60
TICKERS = [f"T{i:02d}" for i in range(N)] + ["SPY"]


class SyntheticPx:
    """Each name has a hidden 'quality' q; price drift correlates weakly
    with q so a momentum-style score has real but noisy predictive power."""

    def __init__(self):
        self.q = {f"T{i:02d}": (i / N - 0.5) for i in range(N)}

    def get_prices(self, t, start, end):
        rng = np.random.default_rng(abs(hash(t)) % 2**32)
        if t == "SPY":
            drift = 0.0003
        else:
            drift = 0.0002 + 0.0006 * self.q[t]      # weak signal
        shock = rng.normal(drift, 0.013, len(IDX))
        px = 100 * np.cumprod(1 + shock)
        s = pd.Series(px, index=IDX)
        s = s.loc[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
        return pd.DataFrame({"close": s, "volume": 3_000_000.0})


def universe_fn(date):
    return [f"T{i:02d}" for i in range(N)]


def score_fn(as_of, tickers):
    px = SyntheticPx()
    out = {}
    for t in tickers:
        c = px.get_prices(t, "2015-01-01", as_of)["close"]
        if len(c) > 130:
            out[t] = c.iloc[-21] / c.iloc[-252] - 1.0   # 12-1 momentum, as-of
    return pd.Series(out)


def main() -> int:
    bt = WalkForwardBacktest(
        SyntheticPx(), universe_fn, score_fn,
        cost_model=CostModel(), benchmark="SPY",
        min_history_months=36, decile=0.10, min_names=30,
    )
    print("Running walk-forward 2018-01-01 → 2023-12-31 (SYNTHETIC data)…\n")
    res = bt.run("2018-01-01", "2023-12-31", run_id="verify")

    print(json.dumps(res.metrics, indent=2))
    print(f"\nartifacts written to: {res.out_dir}")
    print("  returns.parquet  positions.parquet  trades.parquet")
    print("  metrics.json     tearsheet.html")

    sh = res.metrics["sharpe"]
    note = ("plausible for a noisy synthetic L/S"
            if -0.5 <= sh <= 2.0 else
            "OUT OF SANE RANGE — on real data this would mean a leak")
    print(f"\nSharpe={sh} -> {note}")
    print("On REAL S&P data the spec's expectation is Sharpe 0.6–1.0; "
          ">2.0 means investigate for lookahead before trusting it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
