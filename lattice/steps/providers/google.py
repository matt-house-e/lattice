"""Google Gemini provider adapter — optional extra: pip install lattice[google]."""

from __future__ import annotations

import os
from typing import Any, Optional

from ...schemas.base import UsageInfo
from .base import LLMAPIError, LLMResponse


class GoogleClient:
    """Adapter for Google's Gemini models.

    Requires: pip install lattice[google]
    """

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
            except ImportError:
                raise ImportError(
                    "google-genai package required: pip install lattice[google]"
                )
            key = self._api_key or os.environ.get("GOOGLE_API_KEY")
            self._client = genai.Client(api_key=key)
        return self._client

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        client = self._get_client()

        # Convert messages to Gemini format
        system_instruction = ""
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
            elif msg["role"] == "assistant":
                contents.append({"role": "model", "parts": [{"text": msg["content"]}]})

        try:
            from google.genai import types

            config: dict[str, Any] = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if system_instruction:
                config["system_instruction"] = system_instruction
            if response_format and response_format.get("type") == "json_object":
                config["response_mime_type"] = "application/json"

            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**config),
            )
        except Exception as exc:
            # Google SDK doesn't have typed error classes as clean as OpenAI/Anthropic
            exc_str = str(exc).lower()
            if "429" in exc_str or "rate" in exc_str:
                raise LLMAPIError(
                    str(exc), status_code=429, is_rate_limit=True
                ) from exc
            if "timeout" in exc_str:
                raise LLMAPIError(str(exc), status_code=408) from exc
            raise LLMAPIError(str(exc)) from exc

        content = response.text or ""
        usage = None
        if response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count or 0
            completion_tokens = response.usage_metadata.candidates_token_count or 0
            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                model=model,
            )

        return LLMResponse(content=content, usage=usage)
