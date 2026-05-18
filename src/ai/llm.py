"""Provider-abstracted LLM client returning strict JSON.

The spec mandates Claude Opus 4.7; this build also supports OpenAI so it
can run on whichever key is present. ``get_llm()`` prefers Anthropic when
its key is set (spec default), otherwise OpenAI. All calls are
temperature 0 for reproducible forensic output.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

log = logging.getLogger("greymatter.ai.llm")


class LLMClient(Protocol):
    name: str

    def complete_json(self, system: str, user: str) -> dict[str, Any]: ...


def _coerce_json(text: str) -> dict[str, Any]:
    """Best-effort: strip code fences / prose around a JSON object."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        t = t[4:] if t.lower().startswith("json") else t
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        i, j = t.find("{"), t.rfind("}")
        if i != -1 and j != -1 and j > i:
            return json.loads(t[i : j + 1])
        raise


class OpenAIClient:
    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI

        settings.require("openai_api_key")
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = model or settings.openai_model

    @retry(wait=wait_exponential(multiplier=1, min=2, max=30),
           stop=stop_after_attempt(4), reraise=True)
    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return _coerce_json(resp.choices[0].message.content or "{}")


class AnthropicClient:
    """Spec-default path (Claude Opus 4.7). Used when ANTHROPIC_API_KEY is
    set and the ``anthropic`` package is installed."""

    name = "anthropic"

    def __init__(self, model: str = "claude-opus-4-7") -> None:
        import anthropic  # optional dependency

        settings.require("anthropic_api_key")
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = model

    @retry(wait=wait_exponential(multiplier=1, min=2, max=30),
           stop=stop_after_attempt(4), reraise=True)
    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=2000,
            temperature=0,
            system=system + "\n\nRespond with a single valid JSON object only.",
            messages=[{"role": "user", "content": user}],
        )
        return _coerce_json(msg.content[0].text)


def get_llm() -> LLMClient:
    if settings.anthropic_api_key:
        return AnthropicClient()           # spec default
    if settings.openai_api_key:
        return OpenAIClient()
    raise RuntimeError(
        "No AI key set. Add ANTHROPIC_API_KEY (spec default, Claude Opus 4.7) "
        "or OPENAI_API_KEY to .env."
    )
