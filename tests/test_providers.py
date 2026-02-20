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


# -- AnthropicClient --------------------------------------------------------


class TestAnthropicClient:
    """Test Anthropic adapter — structured outputs via output_config.format.

    The ``anthropic`` package is an optional extra, so we inject a fake module
    into ``sys.modules`` to satisfy the deferred ``from anthropic import ...``
    inside ``complete()``.
    """

    @staticmethod
    def _install_mock_anthropic():
        """Install a minimal mock ``anthropic`` package into sys.modules."""
        import sys

        mock_mod = MagicMock()
        # Provide the exception classes the adapter imports
        mock_mod.APIError = type("APIError", (Exception,), {})
        mock_mod.APITimeoutError = type("APITimeoutError", (Exception,), {})
        mock_mod.RateLimitError = type("RateLimitError", (Exception,), {})
        mock_mod.AsyncAnthropic = MagicMock()
        sys.modules.setdefault("anthropic", mock_mod)
        return mock_mod

    @pytest.mark.asyncio
    async def test_json_schema_translated_to_output_config(self):
        """json_schema response_format → Anthropic output_config.format."""
        self._install_mock_anthropic()
        from lattice.steps.providers.anthropic import AnthropicClient

        schema = {
            "type": "object",
            "properties": {"f1": {"type": "string"}},
            "required": ["f1"],
            "additionalProperties": False,
        }
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "enrichment_result",
                "schema": schema,
                "strict": True,
            },
        }

        mock_response = SimpleNamespace(
            content=[SimpleNamespace(text='{"f1": "val"}')],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
        mock_inner = MagicMock()
        mock_inner.messages.create = AsyncMock(return_value=mock_response)

        client = AnthropicClient(api_key="test")
        client._client = mock_inner

        result = await client.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-sonnet-4-5-20250929",
            temperature=0.2,
            max_tokens=1000,
            response_format=response_format,
        )

        call_kwargs = mock_inner.messages.create.call_args.kwargs
        assert "output_config" in call_kwargs
        assert call_kwargs["output_config"]["format"]["type"] == "json_schema"
        assert call_kwargs["output_config"]["format"]["schema"] == schema
        assert result.content == '{"f1": "val"}'

    @pytest.mark.asyncio
    async def test_json_object_ignored(self):
        """json_object has no Anthropic equivalent — no output_config set."""
        self._install_mock_anthropic()
        from lattice.steps.providers.anthropic import AnthropicClient

        mock_response = SimpleNamespace(
            content=[SimpleNamespace(text='{"f1": "val"}')],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
        mock_inner = MagicMock()
        mock_inner.messages.create = AsyncMock(return_value=mock_response)

        client = AnthropicClient(api_key="test")
        client._client = mock_inner

        await client.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-sonnet-4-5-20250929",
            temperature=0.2,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )

        call_kwargs = mock_inner.messages.create.call_args.kwargs
        assert "output_config" not in call_kwargs

    @pytest.mark.asyncio
    async def test_none_response_format_no_output_config(self):
        """No response_format → no output_config."""
        self._install_mock_anthropic()
        from lattice.steps.providers.anthropic import AnthropicClient

        mock_response = SimpleNamespace(
            content=[SimpleNamespace(text="hello")],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
        mock_inner = MagicMock()
        mock_inner.messages.create = AsyncMock(return_value=mock_response)

        client = AnthropicClient(api_key="test")
        client._client = mock_inner

        await client.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-sonnet-4-5-20250929",
            temperature=0.2,
            max_tokens=1000,
        )

        call_kwargs = mock_inner.messages.create.call_args.kwargs
        assert "output_config" not in call_kwargs


# -- GoogleClient -----------------------------------------------------------


class TestGoogleClient:
    """Test Google Gemini adapter — structured outputs via response_json_schema.

    The ``google-genai`` package is an optional extra, so we inject a fake module
    into ``sys.modules`` to satisfy the deferred ``from google.genai import types``
    inside ``complete()``.
    """

    @staticmethod
    def _install_mock_google():
        """Install minimal mock ``google.genai`` package into sys.modules."""
        import sys

        mock_types = MagicMock()
        mock_types.GenerateContentConfig = lambda **kw: kw

        mock_genai = MagicMock()
        mock_genai.types = mock_types
        mock_genai.Client = MagicMock()

        # google is a namespace package — install all levels
        sys.modules.setdefault("google", MagicMock())
        sys.modules.setdefault("google.genai", mock_genai)
        sys.modules.setdefault("google.genai.types", mock_types)
        return mock_types

    @pytest.mark.asyncio
    async def test_json_schema_translated_to_response_json_schema(self):
        """json_schema → response_mime_type + response_json_schema."""
        self._install_mock_google()
        from lattice.steps.providers.google import GoogleClient

        schema = {
            "type": "object",
            "properties": {"f1": {"type": "string"}},
            "required": ["f1"],
            "additionalProperties": False,
        }
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "enrichment_result",
                "schema": schema,
                "strict": True,
            },
        }

        mock_response = SimpleNamespace(
            text='{"f1": "val"}',
            usage_metadata=SimpleNamespace(
                prompt_token_count=10, candidates_token_count=5,
            ),
        )
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = GoogleClient(api_key="test")
        client._client = mock_client

        result = await client.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-2.5-flash",
            temperature=0.2,
            max_tokens=1000,
            response_format=response_format,
        )

        call_args = mock_client.aio.models.generate_content.call_args
        config = call_args.kwargs["config"]
        assert config["response_mime_type"] == "application/json"
        assert config["response_json_schema"] == schema
        assert result.content == '{"f1": "val"}'

    @pytest.mark.asyncio
    async def test_json_object_sets_mime_type_only(self):
        """json_object → response_mime_type only, no schema."""
        self._install_mock_google()
        from lattice.steps.providers.google import GoogleClient

        mock_response = SimpleNamespace(
            text='{"f1": "val"}',
            usage_metadata=SimpleNamespace(
                prompt_token_count=10, candidates_token_count=5,
            ),
        )
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = GoogleClient(api_key="test")
        client._client = mock_client

        await client.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-2.5-flash",
            temperature=0.2,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )

        call_args = mock_client.aio.models.generate_content.call_args
        config = call_args.kwargs["config"]
        assert config["response_mime_type"] == "application/json"
        assert "response_json_schema" not in config
