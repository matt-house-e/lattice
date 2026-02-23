"""Anthropic provider adapter — optional extra: pip install accrue[anthropic]."""

from __future__ import annotations

import os
from typing import Any, Optional

from ...schemas.base import UsageInfo
from ...schemas.grounding import Citation
from .base import LLMAPIError, LLMResponse


class AnthropicClient:
    """Adapter for Anthropic's Claude models.

    Requires: pip install accrue[anthropic]

    Supports the ``web_search_20250305`` server tool for grounded responses.
    When web search tools are active, structured outputs via
    ``output_config.format`` are disabled (incompatible with citations).
    """

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError("anthropic package required: pip install accrue[anthropic]")
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
        tools: list[dict[str, Any]] | None = None,
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

        # Translate tools (e.g. web_search → web_search_20250305 server tool)
        anthropic_tools = _translate_tools(tools) if tools else None
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        # Structured outputs: Anthropic uses output_config.format (GA)
        # json_schema → constrained decoding; json_object → no equivalent, skip
        # IMPORTANT: output_config.format is incompatible with web search citations
        if not anthropic_tools and response_format and response_format.get("type") == "json_schema":
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
                f"Anthropic rate limit for model '{model}': {exc}",
                status_code=429,
                is_rate_limit=True,
            ) from exc
        except APITimeoutError as exc:
            raise LLMAPIError(
                f"Anthropic timeout for model '{model}': {exc}",
                status_code=408,
            ) from exc
        except APIError as exc:
            raise LLMAPIError(
                f"Anthropic API error for model '{model}': {exc}",
                status_code=getattr(exc, "status_code", None),
            ) from exc

        # Extract text from potentially multi-block response
        content = _extract_text(response)

        # Extract citations from web_search_result_location blocks
        citations = _extract_citations(response) if anthropic_tools else []

        usage = None
        if response.usage:
            usage = UsageInfo(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                model=model,
            )

        return LLMResponse(content=content, usage=usage, citations=citations)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _translate_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate generic Accrue tool dicts to Anthropic server tool format."""
    anthropic_tools: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") == "web_search":
            server_tool: dict[str, Any] = {
                "type": "web_search_20250305",
                "name": "web_search",
            }
            # Map config fields
            if "allowed_domains" in tool:
                server_tool["allowed_domains"] = tool["allowed_domains"]
            if "blocked_domains" in tool:
                server_tool["blocked_domains"] = tool["blocked_domains"]
            if "user_location" in tool:
                loc = tool["user_location"]
                server_tool["user_location"] = {"type": "approximate", **loc}
            if "max_searches" in tool:
                server_tool["max_uses"] = tool["max_searches"]
            # Merge provider-specific kwargs (pass-through)
            if "provider_kwargs" in tool:
                server_tool.update(tool["provider_kwargs"])
            anthropic_tools.append(server_tool)
    return anthropic_tools


def _extract_text(response: Any) -> str:
    """Extract all text content from an Anthropic response (may have multiple blocks)."""
    parts: list[str] = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


def _extract_citations(response: Any) -> list[Citation]:
    """Extract web_search_result_location citations from an Anthropic response."""
    citations: list[Citation] = []
    seen_urls: set[str] = set()
    for block in response.content:
        if getattr(block, "type", None) != "text":
            continue
        block_citations = getattr(block, "citations", None)
        if not block_citations:
            continue
        for cite in block_citations:
            if getattr(cite, "type", None) == "web_search_result_location":
                url = getattr(cite, "url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    citations.append(Citation(
                        url=url,
                        title=getattr(cite, "title", ""),
                        snippet=getattr(cite, "cited_text", ""),
                    ))
    return citations
