"""Tests for LLMClient protocol and provider adapters."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.schemas.base import UsageInfo
from lattice.steps.providers.base import LLMAPIError, LLMClient, LLMResponse, LLMAPIError
from lattice.steps.providers.openai import OpenAIClient


# -- LLMResponse ---------------------------------------------------------


class TestLLMResponse:
    def test_creation_minimal(self):
        r = LLMResponse(content="hello")
        assert r.content == "hello"
        assert r.usage is None

    def test_creation_with_usage(self):
        usage = UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15, model="test")
        r = LLMResponse(content="hello", usage=usage)
        assert r.usage.total_tokens == 15


# -- LLMAPIError ---------------------------------------------------------


class TestLLMAPIError:
    def test_basic(self):
        e = LLMAPIError("rate limited", status_code=429, is_rate_limit=True)
        assert e.status_code == 429
        assert e.is_rate_limit is True
        assert e.retry_after is None

    def test_with_retry_after(self):
        e = LLMAPIError("rate limited", retry_after=2.5, is_rate_limit=True)
        assert e.retry_after == 2.5


# -- LLMClient protocol check -------------------------------------------


class TestLLMClientProtocol:
    def test_openai_client_satisfies_protocol(self):
        client = OpenAIClient(api_key="test")
        assert isinstance(client, LLMClient)


# -- OpenAIClient --------------------------------------------------------


class TestOpenAIClient:
    @patch("openai.AsyncOpenAI")
    def test_lazy_client_creation(self, mock_cls):
        client = OpenAIClient(api_key="key123")
        assert client._client is None
        inner = client._get_client()
        mock_cls.assert_called_once_with(api_key="key123")
        assert inner is mock_cls.return_value

    @patch("openai.AsyncOpenAI")
    def test_base_url_passed_to_sdk(self, mock_cls):
        client = OpenAIClient(api_key="key", base_url="http://localhost:11434/v1")
        client._get_client()
        mock_cls.assert_called_once_with(api_key="key", base_url="http://localhost:11434/v1")

    @pytest.mark.asyncio
    async def test_complete_success(self):
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"f": 1}'))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        mock_inner = MagicMock()
        mock_inner.chat.completions.create = AsyncMock(return_value=mock_response)

        client = OpenAIClient(api_key="test")
        client._client = mock_inner

        result = await client.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4.1-nano",
            temperature=0.2,
            max_tokens=100,
        )

        assert isinstance(result, LLMResponse)
        assert result.content == '{"f": 1}'
        assert result.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_complete_with_response_format(self):
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
            usage=None,
        )
        mock_inner = MagicMock()
        mock_inner.chat.completions.create = AsyncMock(return_value=mock_response)

        client = OpenAIClient(api_key="test")
        client._client = mock_inner

        result = await client.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="test",
            temperature=0.0,
            max_tokens=10,
            response_format={"type": "json_object"},
        )

        call_kwargs = mock_inner.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert result.usage is None

    @pytest.mark.asyncio
    async def test_rate_limit_raises_llm_api_error(self):
        from openai import RateLimitError

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "2.5"}

        exc = RateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body=None,
        )

        mock_inner = MagicMock()
        mock_inner.chat.completions.create = AsyncMock(side_effect=exc)

        client = OpenAIClient(api_key="test")
        client._client = mock_inner

        with pytest.raises(LLMAPIError) as exc_info:
            await client.complete(
                messages=[], model="test", temperature=0.0, max_tokens=10,
            )

        assert exc_info.value.is_rate_limit is True
        assert exc_info.value.status_code == 429
        assert exc_info.value.retry_after == 2.5

    @pytest.mark.asyncio
    async def test_timeout_raises_llm_api_error(self):
        from openai import APITimeoutError

        exc = APITimeoutError(request=MagicMock())

        mock_inner = MagicMock()
        mock_inner.chat.completions.create = AsyncMock(side_effect=exc)

        client = OpenAIClient(api_key="test")
        client._client = mock_inner

        with pytest.raises(LLMAPIError) as exc_info:
            await client.complete(
                messages=[], model="test", temperature=0.0, max_tokens=10,
            )

        assert exc_info.value.status_code == 408
