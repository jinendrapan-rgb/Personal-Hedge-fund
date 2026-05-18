"""Layer-2 verification.

Runs the factor engine on 8 real S&P names. Fundamentals are fetched LIVE
from SEC EDGAR (no API key) so quality/value are computed from real
filings; prices are deterministic synthetic series because Polygon needs
the paid key (clearly labelled). Proves the engine wires into the real
Layer-1 DataManager and produces a sane sector-neutral ranking.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.data_manager import DataManager
from src.factors.factor_engine import FACTORS, FactorEngine
from src.factors.sectors import SectorProvider

TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "CVX", "PG", "KO"]
SECTORS = {
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "JPM": "Financials",
    "XOM": "Energy", "CVX": "Energy", "PG": "Staples", "KO": "Staples",
}
ASOF = "2023-06-30"


class RealFundSyntheticPxDM:
    """Real SEC fundamentals via DataManager; synthetic deterministic prices."""

    def __init__(self) -> None:
        self._dm = DataManager()

    def get_universe(self, date):
        return TICKERS

    def get_prices(self, t, start, end):
        rng = np.random.default_rng(abs(hash(t)) % 2**32)
        idx = pd.bdate_range("2021-06-01", periods=560)
        drift = 0.0003 + (abs(hash(t)) % 7) * 0.0002
        px = 100 * np.cumprod(1 + rng.normal(drift, 0.012, len(idx)))
        df = pd.DataFrame({"close": px}, index=idx)
        df.index.name = "date"
        return df.loc[df.index <= pd.Timestamp(end)]

    def get_fundamentals(self, t, when):
        return self._dm.get_fundamentals(t, when)


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        sec_csv = Path(d) / "sectors.csv"
        sec_csv.write_text(
            "ticker,sector\n" + "\n".join(f"{k},{v}" for k, v in SECTORS.items())
        )
        engine = FactorEngine(
            RealFundSyntheticPxDM(), sectors=SectorProvider(file=sec_csv)
        )
        print("Computing factors as-of", ASOF,
              "(SEC fundamentals = REAL, prices = synthetic)\n")
        df = engine.compute(ASOF)

    pd.set_option("display.float_format", lambda x: f"{x:+.2f}")
    cols = ["ticker", *FACTORS, "composite"]
    print(df[cols].to_string(index=False))

    print("\nSector-neutrality check (mean per sector, should be ~0):")
    sec = pd.Series(SECTORS)
    for c in ["quality", "value"]:
        m = df.set_index("ticker")[c].groupby(sec).mean()
        print(f"  {c:9s}: " + "  ".join(f"{k}={v:+.2e}" for k, v in m.items()))

    ok = list(df.columns) == ["date", "ticker", *FACTORS, "composite"]
    print(f"\nSchema OK: {ok} | rows={len(df)} | "
          f"composite==mean(5): "
          f"{np.allclose(df[FACTORS].mean(axis=1), df['composite'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
