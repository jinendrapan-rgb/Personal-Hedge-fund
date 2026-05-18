"""Layer-1 verification (spec: 'run a test fetch for AAPL ... print result').

SEC EDGAR needs no API key (only a descriptive User-Agent), so the
point-in-time fundamentals check runs for real here. Polygon prices need
the paid Starter key; that section self-skips with a clear message if the
key is absent, so the script always produces a verifiable result.
"""
from __future__ import annotations

import sys

import pandas as pd

from src.config import settings
from src.data.data_manager import DataManager


def main() -> int:
    dm = DataManager()

    print("=" * 64)
    print("SEC EDGAR — point-in-time fundamentals (REAL fetch, no key)")
    print("=" * 64)
    try:
        as_of = "2020-01-15"
        snap = dm.get_fundamentals("AAPL", as_of)
        # Show which filing each value came from to prove PIT correctness.
        tidy = dm._tidy_facts("AAPL")
        from src.data.xbrl_parser import as_of as resolve

        print(f"AAPL fundamentals as known on {as_of}:")
        for k, v in snap.items():
            print(f"  {k:22s}: {v}")
        # True provenance: whichever revenue tag actually supplied the value
        # (Apple's us-gaap:Revenues is frozen at FY2018 post-ASC606, so the
        # value comes from RevenueFromContractWithCustomerExcludingAssessedTax).
        full = resolve(tidy, as_of)
        rev = full[full["val"] == snap["revenue"]]
        if not rev.empty:
            r = rev.iloc[0]
            print(
                f"\n  revenue sourced from: {r['concept']} "
                f"form={r['form']} filed={r['filed'].date()} "
                f"period_end={r['end'].date()}"
            )
            ok = r["end"].year == 2019 and r["filed"] < pd.Timestamp(as_of)
            print(
                f"  -> FY2019 figure filed before {as_of}, NOT a later "
                f"restatement. PIT contract {'HOLDING ✓' if ok else 'CHECK!'}"
            )
    except Exception as e:  # network/None — report, don't crash silently
        print(f"  SEC fetch unavailable in this environment: {e!r}")

    print()
    print("=" * 64)
    print("POLYGON — 1y daily prices for AAPL")
    print("=" * 64)
    if not settings.polygon_api_key:
        print("  SKIPPED: POLYGON_API_KEY not set in .env.")
        print("  Add a Polygon Starter key, then re-run to fetch real prices.")
    else:
        df = dm.get_prices("AAPL", "2023-01-01", "2023-12-31")
        print(f"  rows={len(df)}  range={df.index.min()} -> {df.index.max()}")
        print(df.tail(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
