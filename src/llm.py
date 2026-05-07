"""Thin wrapper around the OpenAI Chat Completions API.

Kept tiny on purpose so the agent loop in ``agent.py`` stays the
star of the show. Swap this module out for another provider without
touching anything else.
"""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI, OpenAIError


class LLMError(RuntimeError):
    """Raised when the underlying provider call fails."""


class LLMClient:
    """Wraps any OpenAI-compatible chat-completions endpoint.

    Works out of the box with OpenAI, Groq, Together, OpenRouter, etc.
    Just set ``OPENAI_BASE_URL`` to the provider's endpoint and pick a
    model the provider supports.
    """

    def __init__(self,
                 api_key: str | None = None,
                 model: str | None = None,
                 base_url: str | None = None) -> None:
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env "
                "and fill it in."
            )
        base_url = base_url or os.getenv("OPENAI_BASE_URL") or None
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def chat(self,
             messages: list[dict[str, Any]],
             tools: list[dict[str, Any]] | None = None,
             temperature: float = 0.2) -> Any:
        """Call the model once and return the raw OpenAI message object."""
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=temperature,
            )
        except OpenAIError as exc:
            raise LLMError(f"OpenAI API call failed: {exc}") from exc

        if not response.choices:
            raise LLMError("OpenAI returned no choices.")
        return response.choices[0].message
