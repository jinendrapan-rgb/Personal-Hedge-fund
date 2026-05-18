"""JARVIS — the dashboard's Claude/LLM assistant.

System prompt is loaded with live portfolio state (positions, risk
metrics, recent factor scores) so answers are grounded in the actual
book. The LLM is injected so this is testable offline; in production it
uses the Layer-4 provider abstraction (Claude Opus 4.7 by spec, OpenAI
when that's the key set).
"""
from __future__ import annotations

import json
from typing import Any

_SYSTEM = """You are JARVIS, the analyst-assistant for the GreyMatter
long/short equity system. Answer concisely and quantitatively. Ground
every claim in the STATE block below — if the state doesn't contain the
answer, say so rather than guessing. Never invent positions or numbers.

=== CURRENT STATE ===
{state}
=== END STATE ==="""


class Jarvis:
    def __init__(self, llm=None) -> None:
        self._llm = llm

    @property
    def llm(self):
        if self._llm is None:
            from src.ai.llm import get_llm
            self._llm = get_llm()
        return self._llm

    @staticmethod
    def build_system(state: dict[str, Any]) -> str:
        return _SYSTEM.format(state=json.dumps(state, indent=2, default=str))

    def ask(self, question: str, state: dict[str, Any]) -> str:
        return self.llm.complete_text(self.build_system(state), question).strip()
