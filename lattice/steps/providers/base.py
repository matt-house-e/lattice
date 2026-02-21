"""LLMClient protocol and LLMResponse — provider-agnostic LLM interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from ...schemas.base import UsageInfo
from ...schemas.grounding import Citation


@dataclass
class LLMResponse:
    """Response from an LLM provider.

    Attributes:
        content: The text content of the response.
        usage: Token usage information.
        citations: Normalised source citations when grounding tools were used.
    """

    content: str
    usage: Optional[UsageInfo] = None
    citations: list[Citation] = field(default_factory=list)


class LLMAPIError(Exception):
    """Provider-agnostic API error for retry logic.

    Wraps provider-specific errors (openai.RateLimitError, anthropic.RateLimitError, etc.)
    so LLMStep retry logic doesn't need to know about specific SDKs.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
        is_rate_limit: bool = False,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.is_rate_limit = is_rate_limit


@runtime_checkable
class LLMClient(Protocol):
    """Protocol all LLM provider adapters must satisfy."""

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse: ...
