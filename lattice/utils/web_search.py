"""Web search utility — factory wrapping OpenAI Responses API web search.

Returns an async callable compatible with ``FunctionStep``.  Reduces the
common two-step research-then-analyze pattern from ~30 lines to 3.

Example::

    from lattice import Pipeline, FunctionStep, LLMStep, web_search

    Pipeline([
        FunctionStep("research",
            fn=web_search("Research {company}: market position"),
            fields=["__web_context", "sources"],
        ),
        LLMStep("analyze", fields={...}, depends_on=["research"]),
    ])
"""

from __future__ import annotations

import os
from typing import Any, Awaitable, Callable

from ..steps.base import StepContext

_VALID_CONTEXT_SIZES = frozenset({"low", "medium", "high"})


def web_search(
    query: str,
    *,
    model: str = "gpt-4.1-mini",
    search_context_size: str = "medium",
    api_key: str | None = None,
    include_sources: bool = True,
) -> Callable[[StepContext], Awaitable[dict[str, Any]]]:
    """Factory returning an async callable for ``FunctionStep``.

    Args:
        query: Template string with ``{field}`` placeholders, formatted with
            ``ctx.row`` and ``ctx.prior_results``.
        model: OpenAI model for the search call. Must support web search
            (gpt-4.1-mini or gpt-4.1).
        search_context_size: ``"low"`` | ``"medium"`` | ``"high"`` — amount of
            context from search results.
        api_key: OpenAI API key.  Falls back to ``OPENAI_API_KEY`` env var.
        include_sources: If True, extract URL citations from the response.

    Returns:
        ``{"__web_context": str, "sources": list[str]}``
    """
    # Eager validation
    if search_context_size not in _VALID_CONTEXT_SIZES:
        raise ValueError(
            f"search_context_size must be one of {sorted(_VALID_CONTEXT_SIZES)}, "
            f"got {search_context_size!r}"
        )

    async def _search(ctx: StepContext) -> dict[str, Any]:
        from openai import APIError, APITimeoutError, RateLimitError
        from openai import AsyncOpenAI

        # Format the query template
        template_vars = dict(ctx.row)
        if ctx.prior_results:
            template_vars.update(ctx.prior_results)

        try:
            formatted_query = query.format(**template_vars)
        except KeyError as exc:
            raise ValueError(
                f"web_search query template references missing field: {exc}"
            ) from exc

        key = api_key or os.environ.get("OPENAI_API_KEY")
        client = AsyncOpenAI(api_key=key)

        try:
            response = await client.responses.create(
                model=model,
                tools=[{"type": "web_search_preview", "search_context_size": search_context_size}],
                input=formatted_query,
            )

            # Extract text content
            web_context = ""
            sources: list[str] = []

            for item in response.output:
                if hasattr(item, "content"):
                    # Message output item — extract text
                    for part in item.content:
                        if hasattr(part, "text"):
                            web_context = part.text
                        # Extract citations from annotations
                        if include_sources and hasattr(part, "annotations"):
                            for annotation in part.annotations:
                                if hasattr(annotation, "url"):
                                    sources.append(annotation.url)

            if not include_sources:
                sources = []

            return {"__web_context": web_context, "sources": sources}

        except (APIError, RateLimitError, APITimeoutError):
            # Graceful degradation — downstream LLMStep works with row data
            return {"__web_context": "", "sources": []}

    return _search
