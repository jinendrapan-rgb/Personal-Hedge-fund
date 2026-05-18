"""InsiderAnalyzer — classifies Form 4 transaction patterns over ~90 days."""
from __future__ import annotations

import json

from src.ai.base import Analyzer
from src.ai.models import AnalysisReport


class InsiderAnalyzer(Analyzer):
    name = "insider"
    prompt_file = "insider.md"

    def analyze(
        self,
        ticker: str,
        as_of_date: str,
        transactions: list[dict],
        *,
        use_cache: bool = True,
    ) -> AnalysisReport:
        """``transactions``: list of dicts with keys like insider, role,
        date, type (buy/sell), shares, price, plan (10b5-1/open)."""
        content = (
            f"Ticker {ticker}, trailing 90 days as of {as_of_date}.\n"
            f"Form 4 records:\n{json.dumps(transactions, indent=2, default=str)}"
        )
        return self._run(ticker, as_of_date, content, use_cache=use_cache)
