"""OpenAI provider adapter — default, covers all OpenAI-compatible APIs."""

from __future__ import annotations

import os
from typing import Any, Optional

from ...schemas.base import UsageInfo
from .base import LLMAPIError, LLMResponse


class OpenAIClient:
    """Adapter for OpenAI and OpenAI-compatible providers.

    Covers: OpenAI, DeepSeek, Groq, Together, Fireworks, Ollama,
    vLLM, Mistral, LM Studio — anything with a base_url.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            key = self._api_key or os.environ.get("OPENAI_API_KEY")
            kwargs: dict[str, Any] = {"api_key": key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        from openai import APIError, APITimeoutError, RateLimitError

        client = self._get_client()
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response_format:
            kwargs["response_format"] = response_format

        try:
            response = await client.chat.completions.create(**kwargs)
        except RateLimitError as exc:
            retry_after = None
            if hasattr(exc, "response") and exc.response is not None:
                retry_after_header = exc.response.headers.get("retry-after")
                if retry_after_header:
                    try:
                        retry_after = float(retry_after_header)
                    except (ValueError, TypeError):
                        pass
            raise LLMAPIError(
                f"OpenAI rate limit for model '{model}': {exc}",
                status_code=429,
                retry_after=retry_after,
                is_rate_limit=True,
            ) from exc
        except APITimeoutError as exc:
            raise LLMAPIError(
                f"OpenAI timeout for model '{model}': {exc}",
                status_code=408,
            ) from exc
        except APIError as exc:
            raise LLMAPIError(
                f"OpenAI API error for model '{model}': {exc}",
                status_code=getattr(exc, "status_code", None),
            ) from exc

        content = response.choices[0].message.content or ""
        usage = None
        if response.usage:
            usage = UsageInfo(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                model=model,
            )

        return LLMResponse(content=content, usage=usage)
