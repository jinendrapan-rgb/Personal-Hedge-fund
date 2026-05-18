"""Shared analyzer machinery: prompt loading, caching, LLM call, parsing."""
from __future__ import annotations

import logging
from pathlib import Path

from src.ai import cache
from src.ai.llm import LLMClient, get_llm
from src.ai.models import AnalysisReport

log = logging.getLogger("greymatter.ai")

_PROMPT_DIR = Path(__file__).parent / "prompts"
# Cap user-content size: filings run to hundreds of KB; keep token cost
# bounded and predictable. Forensic signal lives in MD&A / notes, which
# fit comfortably; callers pass the relevant sections.
MAX_CHARS = 60_000


class Analyzer:
    name: str = "base"
    prompt_file: str = ""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm
        self._system = (_PROMPT_DIR / self.prompt_file).read_text()

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:        # lazily created so offline tests/imports
            self._llm = get_llm()    # don't require a key
        return self._llm

    def _run(
        self, ticker: str, filing_date: str, user_content: str, *, use_cache=True
    ) -> AnalysisReport:
        if use_cache:
            cached = cache.load(self.name, ticker, filing_date)
            if cached is not None:
                return AnalysisReport.parse(
                    cached, analyzer=self.name, ticker=ticker,
                    filing_date=filing_date,
                )
        raw = self.llm.complete_json(self._system, user_content[:MAX_CHARS])
        if use_cache:
            cache.store(self.name, ticker, filing_date, raw)
        return AnalysisReport.parse(
            raw, analyzer=self.name, ticker=ticker, filing_date=filing_date
        )
