"""EarningsCallAnalyzer — tone/topic shift vs the prior call transcript."""
from __future__ import annotations

from src.ai.base import Analyzer
from src.ai.models import AnalysisReport


class EarningsCallAnalyzer(Analyzer):
    name = "earnings_call"
    prompt_file = "earnings_call.md"

    def analyze(
        self,
        ticker: str,
        call_date: str,
        transcript: str,
        prior_transcript: str | None = None,
        *,
        use_cache: bool = True,
    ) -> AnalysisReport:
        content = f"=== CURRENT CALL ({ticker} {call_date}) ===\n{transcript}"
        if prior_transcript:
            content += f"\n\n=== PRIOR CALL ===\n{prior_transcript}"
        else:
            content += "\n\n(No prior transcript supplied — judge tone in isolation, conservatively.)"
        return self._run(ticker, call_date, content, use_cache=use_cache)
