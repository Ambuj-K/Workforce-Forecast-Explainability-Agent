"""LLM access behind a small interface, so the provider can change without touching the agent.

* ``GeminiLLM``: Google Gemini via the ``google-genai`` SDK (key from ``GOOGLE_API_KEY``).
* ``ScriptedLLM``: replays prepared responses, for deterministic tests.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "gemini-3.1-flash-lite"


class LLM(Protocol):
    def structured(self, system: str, user: str, schema: type[T]) -> T: ...

    def text(self, system: str, user: str) -> str: ...


class GeminiLLM:
    """Gemini with temperature 0, JSON-schema output for structured calls, and retry on rate limits."""

    def __init__(self, model: str | None = None, api_key: str | None = None, max_retries: int = 4) -> None:
        from google import genai  # imported lazily so tests never need the SDK configured

        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is not set (put it in .env, never in code)")
        self._client = genai.Client(api_key=key)
        self._model = model or os.environ.get("WFX_LLM_MODEL", DEFAULT_MODEL)
        self._max_retries = max_retries

    def _call(self, fn: Callable[[], object]) -> object:
        from google.genai import errors

        delay = 2.0
        for attempt in range(self._max_retries + 1):
            try:
                return fn()
            except errors.APIError as err:
                if getattr(err, "code", None) not in (429, 500, 503) or attempt == self._max_retries:
                    raise
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            response_mime_type="application/json",
            response_schema=schema,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response = self._call(lambda: self._client.models.generate_content(model=self._model, contents=user, config=config))
        return schema.model_validate_json(response.text)

    def text(self, system: str, user: str) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response = self._call(lambda: self._client.models.generate_content(model=self._model, contents=user, config=config))
        return response.text or ""


class ScriptedLLM:
    """Returns prepared responses in order; records every prompt it received."""

    def __init__(self, structured: list[BaseModel] | None = None, texts: list[str] | None = None) -> None:
        self._structured = list(structured or [])
        self._texts = list(texts or [])
        self.prompts: list[tuple[str, str]] = []

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        self.prompts.append((system, user))
        response = self._structured.pop(0)
        assert isinstance(response, schema), f"scripted {type(response).__name__}, expected {schema.__name__}"
        return response

    def text(self, system: str, user: str) -> str:
        self.prompts.append((system, user))
        return self._texts.pop(0)
