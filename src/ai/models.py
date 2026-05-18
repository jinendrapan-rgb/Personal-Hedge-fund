"""Structured output types for the forensic AI layer.

Every analyzer must return this shape so downstream code never parses free
text. Construction validates and clamps so a misbehaving model can't inject
an out-of-range score or an unknown severity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SEVERITIES = ("low", "med", "high")
CONFIDENCES = ("low", "med", "high")


@dataclass(frozen=True)
class Flag:
    concern: str
    severity: str          # low | med | high
    evidence: str          # quoted text the conclusion rests on

    @staticmethod
    def parse(d: dict[str, Any]) -> "Flag":
        sev = str(d.get("severity", "low")).lower().strip()
        if sev not in SEVERITIES:
            sev = "low"
        return Flag(
            concern=str(d.get("concern", "")).strip(),
            severity=sev,
            evidence=str(d.get("evidence", "")).strip(),
        )


@dataclass(frozen=True)
class AnalysisReport:
    analyzer: str
    ticker: str
    filing_date: str
    flags: list[Flag] = field(default_factory=list)
    score_adjustment: float = 0.0      # clamped to [-2, +2]
    summary: str = ""
    confidence: str = "low"            # low | med | high

    @staticmethod
    def parse(
        raw: dict[str, Any], *, analyzer: str, ticker: str, filing_date: str
    ) -> "AnalysisReport":
        flags = [
            Flag.parse(f)
            for f in (raw.get("flags") or [])
            if isinstance(f, dict) and str(f.get("concern", "")).strip()
        ]
        try:
            score = float(raw.get("score_adjustment", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(-2.0, min(2.0, score))
        # No evidence => no concern. Don't let a model assert a score with
        # zero flags (the prompts forbid it; enforce it here too).
        if not flags:
            score = 0.0
        conf = str(raw.get("confidence", "low")).lower().strip()
        if conf not in CONFIDENCES:
            conf = "low"
        return AnalysisReport(
            analyzer=analyzer,
            ticker=ticker.upper(),
            filing_date=filing_date,
            flags=flags,
            score_adjustment=score,
            summary=str(raw.get("summary", "")).strip(),
            confidence=conf,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "analyzer": self.analyzer,
            "ticker": self.ticker,
            "filing_date": self.filing_date,
            "flags": [vars(f) for f in self.flags],
            "score_adjustment": self.score_adjustment,
            "summary": self.summary,
            "confidence": self.confidence,
        }
