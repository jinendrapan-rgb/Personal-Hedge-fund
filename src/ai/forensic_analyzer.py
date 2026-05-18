"""ForensicAnalyzer — reads the 10-K plus recent 10-Qs for accounting
manipulation patterns."""
from __future__ import annotations

from src.ai.base import Analyzer
from src.ai.models import AnalysisReport


class ForensicAnalyzer(Analyzer):
    name = "forensic"
    prompt_file = "forensic.md"

    def analyze(
        self,
        ticker: str,
        filing_date: str,
        annual_text: str,
        quarterly_texts: list[str] | None = None,
        *,
        use_cache: bool = True,
    ) -> AnalysisReport:
        parts = [f"=== 10-K ({ticker} filed {filing_date}) ===\n{annual_text}"]
        for i, q in enumerate(quarterly_texts or [], 1):
            parts.append(f"\n=== 10-Q #{i} ===\n{q}")
        return self._run(
            ticker, filing_date, "\n".join(parts), use_cache=use_cache
        )
