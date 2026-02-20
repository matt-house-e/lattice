"""Anthropic provider adapter — optional extra: pip install lattice[anthropic]."""

from __future__ import annotations

import os
from typing import Any, Optional

from ...schemas.base import UsageInfo
from .base import LLMAPIError, LLMResponse


class AnthropicClient:
    """Adapter for Anthropic's Claude models.

    Requires: pip install lattice[anthropic]
    """

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required: pip install lattice[anthropic]"
                )
            key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
            self._client = AsyncAnthropic(api_key=key)
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

        # Separate system message from conversation messages
        system_content = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                chat_messages.append(msg)

        kwargs: dict[str, Any] = dict(
            model=model,
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if system_content:
            kwargs["system"] = system_content

        # Structured outputs: Anthropic uses output_config.format (GA)
        # json_schema → constrained decoding; json_object → no equivalent, skip
        if response_format and response_format.get("type") == "json_schema":
            inner = response_format.get("json_schema", {})
            schema = inner.get("schema", {})
            if schema:
                kwargs["output_config"] = {
                    "format": {
                        "type": "json_schema",
                        "schema": schema,
                    }
                }

        try:
            from anthropic import APIError, APITimeoutError, RateLimitError

            response = await client.messages.create(**kwargs)
        except RateLimitError as exc:
            raise LLMAPIError(
                str(exc), status_code=429, is_rate_limit=True
            ) from exc
        except APITimeoutError as exc:
            raise LLMAPIError(str(exc), status_code=408) from exc
        except APIError as exc:
            raise LLMAPIError(
                str(exc), status_code=getattr(exc, "status_code", None)
            ) from exc

        content = response.content[0].text if response.content else ""
        usage = None
        if response.usage:
            usage = UsageInfo(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                model=model,
            )

        return LLMResponse(content=content, usage=usage)
