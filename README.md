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
| 3 | Walk-Forward Backtest | ✅ built + tested (38 tests total; lookahead/survivorship/cost tests pass) |
| 4 | AI Forensic Analysis | ✅ built + tested (48 tests total; live OpenAI fraud/clean discrimination verified) |
| 5 | Portfolio Construction | ✅ built + tested (55 tests total; MVO+BL constraints verified) |
| 6 | Risk Management | ✅ built + tested (61 tests total; full risk suite verified) |
| 7 | Execution (Alpaca) | ✅ built + tested (68 tests total; deterministic clock verified) |
| 8 | Reporting + Dashboard | ✅ built + tested (74 tests total; dashboard launches, 8 tabs) |

**All 8 layers built, tested, and verified.** Run the dashboard:

```bash
uv run streamlit run dashboard/app.py
```

Tabs: Overview · Research · Portfolio · Risk · Performance · Backtest ·
Execution · JARVIS. Every layer has a `scripts/verify_layerN.py` that
proves it on real data where no paid key is required (SEC, OpenAI) and on
deterministic synthetic data otherwise.

## Runbook

```bash
uv run python -m src.run preflight                     # what's runnable now
uv run python -m src.run daily --tickers AAPL,MSFT,... # daily pipeline
uv run python -m src.run backtest --start 2018-01-01 --end 2024-12-31
uv run streamlit run dashboard/app.py                  # dashboard
```

The daily pipeline (`src/pipeline.py`) chains all 8 layers and **refuses
to build a portfolio on a degraded signal**: if price history is missing
(e.g. Polygon plan lacks depth), the factor-coverage gate blocks at the
factor stage with `ok: False` rather than trading a silently broken
signal — the core discipline this whole system exists to enforce.

### Current readiness

- **Working on real data, no paid key:** SEC point-in-time fundamentals
  (Layers 1–2: quality, value), OpenAI forensic layer (Layer 4),
  optimiser/risk/execution/dashboard logic (Layers 5–8, tested).
- **Blocked on data plan:** momentum, low-vol, revisions and the
  walk-forward backtest need Polygon **Starter+** (current key is
  recent-data-only; deep history returns 403). The pipeline reports this
  precisely instead of failing silently.
- **To go fully live:** upgrade the Polygon plan, add Alpaca keys, then
  `preflight` → `daily`/`backtest` produce real results with no code
  changes.

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
