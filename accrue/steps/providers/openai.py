"""OpenAI provider adapter — default, covers all OpenAI-compatible APIs.

Uses the Responses API (``client.responses.create``).  For OpenAI-compatible
third-party providers that only expose the Chat Completions endpoint (Ollama,
Groq, etc.), the adapter falls back to ``client.chat.completions.create``
when a ``base_url`` is configured.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from ...schemas.base import UsageInfo
from ...schemas.grounding import Citation
from .base import LLMAPIError, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Adapter for OpenAI and OpenAI-compatible providers.

    Native OpenAI (no ``base_url``) uses the Responses API which supports
    web search tools, structured output via ``text.format``, and inline
    citations.

    Third-party providers with a ``base_url`` (Ollama, Groq, DeepSeek,
    Together, Fireworks, vLLM, Mistral, LM Studio) use the Chat Completions
    API for maximum compatibility.
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
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        if self._base_url:
            return await self._complete_chat(
                messages, model, temperature, max_tokens, response_format,
            )
        return await self._complete_responses(
            messages, model, temperature, max_tokens, response_format, tools,
        )

    # -- Responses API (native OpenAI) ------------------------------------

    async def _complete_responses(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        from openai import APIError, APITimeoutError, RateLimitError

        client = self._get_client()

        # Separate system/instructions from conversation input
        instructions: str | None = None
        input_items: list[dict[str, Any]] = []
        for msg in messages:
            if msg["role"] == "system":
                instructions = msg["content"]
            else:
                input_items.append({"role": msg["role"], "content": msg["content"]})

        kwargs: dict[str, Any] = dict(
            model=model,
            input=input_items,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if instructions:
            kwargs["instructions"] = instructions

        # Structured outputs: response_format → text.format
        if response_format:
            kwargs["text"] = {"format": _translate_response_format(response_format)}

        # Tools (e.g. web_search)
        if tools:
            kwargs["tools"] = _translate_tools(tools)

        try:
            response = await client.responses.create(**kwargs)
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

        # Extract content
        content = response.output_text or ""

        # Extract citations from url_citation annotations
        citations = _extract_citations(response)

        # Extract usage
        usage = None
        if response.usage:
            usage = UsageInfo(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.total_tokens,
                model=model,
            )

        return LLMResponse(content=content, usage=usage, citations=citations)

    # -- Chat Completions API (base_url providers) ------------------------

    async def _complete_chat(
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _translate_response_format(response_format: dict[str, Any]) -> dict[str, Any]:
    """Translate Chat Completions ``response_format`` to Responses API ``text.format``.

    Chat Completions nests the schema under ``json_schema.schema``;
    the Responses API flattens it into the format dict directly.
    """
    fmt_type = response_format.get("type")
    if fmt_type == "json_schema":
        inner = response_format.get("json_schema", {})
        return {
            "type": "json_schema",
            "name": inner.get("name", "response"),
            "strict": inner.get("strict", True),
            "schema": inner.get("schema", {}),
        }
    if fmt_type == "json_object":
        return {"type": "json_object"}
    return {"type": "text"}


def _translate_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate generic grounding tool dicts to OpenAI Responses API format.

    Maps cross-provider fields from :class:`GroundingConfig` into the native
    OpenAI ``web_search`` tool structure:

    - ``allowed_domains`` → ``filters.allowed_domains``
    - ``blocked_domains`` → *unsupported* (logged warning, dropped)
    - ``user_location``   → ``user_location`` with ``{"type": "approximate", ...}``
    - ``max_searches``    → *unsupported* (logged warning, dropped)
    - ``provider_kwargs`` → merged at top level (e.g. ``search_context_size``)
    """
    translated: list[dict[str, Any]] = []
    for tool in tools:
        out = dict(tool)

        # -- provider_kwargs: merge at top level -------------------------
        pk = out.pop("provider_kwargs", None)
        if pk:
            out.update(pk)

        # -- allowed_domains → filters.allowed_domains -------------------
        allowed = out.pop("allowed_domains", None)
        if allowed:
            out.setdefault("filters", {})["allowed_domains"] = allowed

        # -- blocked_domains: not supported by OpenAI --------------------
        blocked = out.pop("blocked_domains", None)
        if blocked:
            logger.warning(
                "OpenAI web_search does not support blocked_domains; "
                "this setting will be ignored."
            )

        # -- user_location → {"type": "approximate", ...} ---------------
        location = out.pop("user_location", None)
        if location:
            out["user_location"] = {"type": "approximate", **location}

        # -- max_searches: not supported by OpenAI -----------------------
        max_searches = out.pop("max_searches", None)
        if max_searches is not None:
            logger.warning(
                "OpenAI web_search does not support max_searches; "
                "this setting will be ignored."
            )

        translated.append(out)
    return translated


def _extract_citations(response: Any) -> list[Citation]:
    """Extract url_citation annotations from a Responses API response."""
    citations: list[Citation] = []
    seen_urls: set[str] = set()
    for item in response.output:
        if not hasattr(item, "content"):
            continue
        for part in item.content:
            if not hasattr(part, "annotations"):
                continue
            for annotation in part.annotations:
                if getattr(annotation, "type", None) == "url_citation":
                    url = getattr(annotation, "url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        citations.append(Citation(
                            url=url,
                            title=getattr(annotation, "title", ""),
                        ))
    return citations
