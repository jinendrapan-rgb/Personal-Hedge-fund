# GreyMatter — AI Hedge Fund

Institutional long/short equity on the S&P 500: 5 academic factors, a Claude
forensic-accounting overlay, sector-neutral construction with real transaction
costs, and a walk-forward backtest *before* any AI or live trading.

Built layer by layer. Each layer is verified before the next is started.

## Status

| # | Layer | State |
|---|-------|-------|
| 1 | Data Infrastructure | ✅ built + tested (15 tests, live SEC PIT verified) |
| 2 | Factor Engine | ✅ built + tested (31 tests total, real SEC fundamentals verified) |
| 3 | Walk-Forward Backtest | ⬜ next |
| 4 | AI Forensic Analysis | ⬜ |
| 5 | Portfolio Construction | ⬜ |
| 6 | Risk Management | ⬜ |
| 7 | Execution (Alpaca) | ⬜ |
| 8 | Reporting + Dashboard | ⬜ |

`greymatter.html` is a standalone interactive map of the full architecture
(open in a browser). It is a diagram, not the running system.

## Setup

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
cp .env.example .env        # then fill in keys
```

Keys: **SEC needs none** (just the User-Agent in `.env`). Polygon Starter is
required for prices; Anthropic for Layer 4; Alpaca (paper) for Layer 7.

## Layer 1 — what it does

`src/data/`

- `polygon_client.py` — daily adjusted OHLCV, paginated, rate-limited, retrying.
- `sec_edgar.py` — EDGAR via the structured `companyfacts`/`submissions` JSON
  APIs (parsed XBRL, **never** HTML regex). Includes delisted names.
- `xbrl_parser.py` — the point-in-time core. `as_of(tidy, date)` returns a
  fundamental **as it was known on that date**, resolving restatements on the
  correct side of their filing date and keeping periods annual-consistent.
- `universe.py` — PIT S&P 500 membership from an explicit add/remove CSV
  (current-snapshot bootstrap from Wikipedia is provided but flagged
  non-PIT — supply a real history file for a leakage-free backtest).
- `data_manager.py` — single OpenBB-style interface with incremental parquet
  caching (`get_prices`, `get_fundamentals`, `get_universe`).

### The non-negotiable, verified

```bash
uv run pytest                       # 15 passing, offline, no keys
uv run python scripts/verify_layer1.py
```

Live SEC fetch confirms `get_fundamentals("AAPL", "2020-01-15")` returns the
FY2019 10-K (filed 2019-10-31), not a later restatement.

## Next

Layer 2 (Factor Engine) depends only on Layer 1's `DataManager`. It can be
built and unit-tested offline with fixtures; full factor runs need a Polygon
key for prices. Don't skip ahead — the spec's whole thesis is that the
walk-forward backtest (Layer 3) only means something on top of correct PIT data.
