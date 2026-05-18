"""Disk cache for analyzer output, keyed (analyzer, ticker, filing_date).

Re-running an analyzer on the same filing returns the cached result — the
spec calls for aggressive caching because Opus-class calls are expensive
and filings don't change after they're filed.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.config import settings

_DIR = settings.paths.cache / "ai"


def _key(analyzer: str, ticker: str, filing_date: str) -> Path:
    safe = f"{analyzer}__{ticker.upper()}__{filing_date}".replace("/", "-")
    return _DIR / f"{safe}.json"


def load(analyzer: str, ticker: str, filing_date: str) -> dict | None:
    p = _key(analyzer, ticker, filing_date)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None
    return None


def store(analyzer: str, ticker: str, filing_date: str, payload: dict) -> None:
    _DIR.mkdir(parents=True, exist_ok=True)
    _key(analyzer, ticker, filing_date).write_text(json.dumps(payload, indent=2))
