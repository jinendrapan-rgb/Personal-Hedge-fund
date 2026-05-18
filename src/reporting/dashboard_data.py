"""Artifact loaders for the dashboard.

All loaders degrade gracefully: if the pipeline hasn't produced data yet
(e.g. no Polygon key, no backtest run), they return well-typed empties so
the UI renders a friendly "no data" state instead of crashing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import settings

_BT = settings.paths.root / "backtests"


def list_backtests() -> list[str]:
    if not _BT.exists():
        return []
    return sorted(p.name for p in _BT.iterdir()
                  if p.is_dir() and (p / "metrics.json").exists())


def load_backtest(run_id: str) -> dict:
    d = _BT / run_id
    out: dict = {"run_id": run_id, "metrics": {}, "returns": pd.DataFrame(),
                 "positions": pd.DataFrame(), "trades": pd.DataFrame()}
    if not d.exists():
        return out
    mj = d / "metrics.json"
    if mj.exists():
        out["metrics"] = json.loads(mj.read_text())
    for key in ("returns", "positions", "trades"):
        f = d / f"{key}.parquet"
        if f.exists():
            out[key] = pd.read_parquet(f)
    return out


def equity_curve(returns: pd.DataFrame) -> pd.Series:
    if returns.empty or "total" not in returns:
        return pd.Series(dtype=float)
    return (1 + returns["total"].fillna(0)).cumprod()


def portfolio_snapshot(positions: pd.DataFrame) -> dict:
    if positions.empty:
        return {"n_positions": 0, "gross": 0.0, "net": 0.0,
                "n_long": 0, "n_short": 0}
    latest = positions.iloc[-1].dropna()
    return {
        "n_positions": int((latest != 0).sum()),
        "gross": float(latest.abs().sum()),
        "net": float(latest.sum()),
        "n_long": int((latest > 0).sum()),
        "n_short": int((latest < 0).sum()),
    }


def load_factor_scores() -> pd.DataFrame:
    """Latest factor_scores parquet if Layer 2 has written one."""
    f = settings.paths.data / "factor_scores.parquet"
    return pd.read_parquet(f) if f.exists() else pd.DataFrame()
