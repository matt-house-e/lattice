"""Tests for the web_search utility factory."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.steps.base import StepContext
from lattice.utils.web_search import web_search


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(**overrides: Any) -> StepContext:
    defaults: dict = dict(
        row={"company": "Acme", "industry": "Tech"},
        fields={},
        prior_results={},
    )
    defaults.update(overrides)
    return StepContext(**defaults)


def _mock_response(text: str = "Search result text", urls: list[str] | None = None):
    """Build a mock OpenAI Responses API response."""
    urls = urls or []

    annotations = []
    for url in urls:
        ann = SimpleNamespace(type="url_citation", url=url)
        annotations.append(ann)

    text_part = SimpleNamespace(type="output_text", text=text, annotations=annotations)
    message_item = SimpleNamespace(type="message", content=[text_part])

    return SimpleNamespace(output=[message_item])


# ---------------------------------------------------------------------------
# Factory validation
# ---------------------------------------------------------------------------


class TestWebSearchFactory:
    def test_invalid_search_context_size_raises(self):
        with pytest.raises(ValueError, match="search_context_size"):
            web_search("test query", search_context_size="huge")

    def test_valid_context_sizes(self):
        for size in ("low", "medium", "high"):
            fn = web_search("test", search_context_size=size)
            assert asyncio.iscoroutinefunction(fn)

    def test_returns_async_callable(self):
        fn = web_search("Search {company}")
        assert asyncio.iscoroutinefunction(fn)


# ---------------------------------------------------------------------------
# Query template formatting
# ---------------------------------------------------------------------------


class TestQueryFormatting:
    @pytest.mark.asyncio
    async def test_formats_from_row_data(self):
        fn = web_search("Research {company} in {industry}")
        ctx = _make_ctx()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(return_value=_mock_response())
            mock_cls.return_value = mock_client

            await fn(ctx)

            call_kwargs = mock_client.responses.create.call_args.kwargs
            assert call_kwargs["input"] == "Research Acme in Tech"

    @pytest.mark.asyncio
    async def test_formats_from_prior_results(self):
        fn = web_search("Search {company} context: {__web_context}")
        ctx = _make_ctx(prior_results={"__web_context": "prior data"})

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(return_value=_mock_response())
            mock_cls.return_value = mock_client

            await fn(ctx)

            call_kwargs = mock_client.responses.create.call_args.kwargs
            assert "prior data" in call_kwargs["input"]

    @pytest.mark.asyncio
    async def test_missing_template_field_raises(self):
        fn = web_search("Research {nonexistent_field}")
        ctx = _make_ctx()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client

            with pytest.raises(ValueError, match="missing field"):
                await fn(ctx)


# ---------------------------------------------------------------------------
# API call verification
# ---------------------------------------------------------------------------


class TestAPICall:
    @pytest.mark.asyncio
    async def test_passes_correct_params(self):
        fn = web_search("test query", model="gpt-4.1", search_context_size="high")
        ctx = _make_ctx()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(return_value=_mock_response())
            mock_cls.return_value = mock_client

            await fn(ctx)

            call_kwargs = mock_client.responses.create.call_args.kwargs
            assert call_kwargs["model"] == "gpt-4.1"
            assert call_kwargs["tools"] == [{"type": "web_search_preview", "search_context_size": "high"}]

    @pytest.mark.asyncio
    async def test_returns_web_context_and_sources(self):
        fn = web_search("Search {company}")
        ctx = _make_ctx()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(
                return_value=_mock_response("Result text", ["https://example.com"])
            )
            mock_cls.return_value = mock_client

            result = await fn(ctx)

        assert result["__web_context"] == "Result text"
        assert result["sources"] == ["https://example.com"]

    @pytest.mark.asyncio
    async def test_include_sources_false(self):
        fn = web_search("Search {company}", include_sources=False)
        ctx = _make_ctx()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(
                return_value=_mock_response("text", ["https://example.com"])
            )
            mock_cls.return_value = mock_client

            result = await fn(ctx)

        assert result["sources"] == []


# ---------------------------------------------------------------------------
# Error handling (graceful degradation)
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_api_error_returns_empty(self):
        from openai import APIError

        fn = web_search("Search {company}")
        ctx = _make_ctx()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            # Create a proper APIError
            mock_client.responses.create = AsyncMock(
                side_effect=APIError(message="test error", request=MagicMock(), body=None)
            )
            mock_cls.return_value = mock_client

            result = await fn(ctx)

        assert result == {"__web_context": "", "sources": []}

    @pytest.mark.asyncio
    async def test_rate_limit_returns_empty(self):
        from openai import RateLimitError

        fn = web_search("Search {company}")
        ctx = _make_ctx()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.headers = {}
            mock_client.responses.create = AsyncMock(
                side_effect=RateLimitError(
                    message="rate limited",
                    response=mock_response,
                    body=None,
                )
            )
            mock_cls.return_value = mock_client

            result = await fn(ctx)

        assert result == {"__web_context": "", "sources": []}

    @pytest.mark.asyncio
    async def test_timeout_returns_empty(self):
        from openai import APITimeoutError

        fn = web_search("Search {company}")
        ctx = _make_ctx()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(
                side_effect=APITimeoutError(request=MagicMock())
            )
            mock_cls.return_value = mock_client

            result = await fn(ctx)

        assert result == {"__web_context": "", "sources": []}


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------


class TestCitationExtraction:
    @pytest.mark.asyncio
    async def test_multiple_citations(self):
        fn = web_search("Search {company}")
        ctx = _make_ctx()

        with patch("openai.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_client.responses.create = AsyncMock(
                return_value=_mock_response(
                    "text",
                    ["https://a.com", "https://b.com", "https://c.com"],
                )
            )
            mock_cls.return_value = mock_client

            result = await fn(ctx)

        assert len(result["sources"]) == 3
        assert "https://a.com" in result["sources"]


# ---------------------------------------------------------------------------
# FunctionStep compatibility
# ---------------------------------------------------------------------------


class TestFunctionStepCompat:
    def test_is_async_callable(self):
        fn = web_search("test")
        assert asyncio.iscoroutinefunction(fn)
