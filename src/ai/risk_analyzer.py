"""RiskAnalyzer — flags materially new/escalated Risk Factors vs prior year."""
from __future__ import annotations

from src.ai.base import Analyzer
from src.ai.models import AnalysisReport


class RiskAnalyzer(Analyzer):
    name = "risk"
    prompt_file = "risk.md"

    def analyze(
        self,
        ticker: str,
        filing_date: str,
        risk_factors: str,
        prior_risk_factors: str | None = None,
        *,
        use_cache: bool = True,
    ) -> AnalysisReport:
        content = f"=== CURRENT RISK FACTORS ({ticker} {filing_date}) ===\n{risk_factors}"
        if prior_risk_factors:
            content += f"\n\n=== PRIOR-YEAR RISK FACTORS ===\n{prior_risk_factors}"
        else:
            content += "\n\n(No prior-year section supplied — judge novelty conservatively.)"
        return self._run(ticker, filing_date, content, use_cache=use_cache)
