"""Tests for LLMStep (LLM calls are mocked via LLMClient protocol)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from lattice.core.exceptions import StepError
from lattice.schemas.base import UsageInfo
from lattice.steps.base import Step, StepContext, StepResult
from lattice.steps.llm import DEFAULT_SYSTEM_PROMPT, LLMStep
from lattice.steps.providers.base import LLMAPIError, LLMResponse


# -- helpers -------------------------------------------------------------


def _make_ctx(**overrides: Any) -> StepContext:
    defaults: dict[str, Any] = dict(
        row={"company": "Acme", "industry": "Tech"},
        fields={"market_size": {"prompt": "Estimate market size", "data_type": "String"}},
        prior_results={},
    )
    defaults.update(overrides)
    return StepContext(**defaults)


def _mock_llm_response(content: str, prompt_tokens: int = 10, completion_tokens: int = 5) -> LLMResponse:
    """Build a fake LLMResponse."""
    return LLMResponse(
        content=content,
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model="gpt-4.1-nano",
        ),
    )


def _make_mock_client(responses: list[LLMResponse] | LLMResponse) -> AsyncMock:
    """Create a mock LLMClient with preset responses."""
    mock = AsyncMock()
    if isinstance(responses, list):
        mock.complete = AsyncMock(side_effect=responses)
    else:
        mock.complete = AsyncMock(return_value=responses)
    return mock


# -- construction --------------------------------------------------------


class TestLLMStepConstruction:
    def test_satisfies_protocol(self):
        step = LLMStep(name="llm", fields=["market_size"])
        assert isinstance(step, Step)

    def test_defaults(self):
        step = LLMStep(name="llm", fields=["f1"])
        assert step.depends_on == []
        assert step.model == "gpt-4.1-nano"
        assert step.temperature is None
        assert step.max_tokens is None
        assert step.system_prompt == DEFAULT_SYSTEM_PROMPT
        assert step.max_retries == 2

    def test_custom_params(self):
        step = LLMStep(
            name="custom",
            fields=["a"],
            depends_on=["dep"],
            model="gpt-4o",
            temperature=0.2,
            max_tokens=2000,
            system_prompt="You are a bot.",
            max_retries=5,
        )
        assert step.model == "gpt-4o"
        assert step.temperature == 0.2
        assert step.system_prompt == "You are a bot."


# -- client resolution ---------------------------------------------------


class TestLLMStepClient:
    def test_client_not_created_at_init(self):
        step = LLMStep(name="llm", fields=["f1"], api_key="test-key")
        assert step._client is None

    @patch("openai.AsyncOpenAI")
    def test_default_creates_openai_client(self, mock_cls):
        step = LLMStep(name="llm", fields=["f1"], api_key="test-key")
        client = step._resolve_client()
        # Should have created an OpenAIClient internally
        assert client is not None

    def test_injected_client_used_directly(self):
        mock_client = AsyncMock()
        step = LLMStep(name="llm", fields=["f1"], client=mock_client)
        assert step._resolve_client() is mock_client

    @patch("openai.AsyncOpenAI")
    def test_base_url_creates_openai_client(self, mock_cls):
        step = LLMStep(
            name="llm", fields=["f1"],
            base_url="http://localhost:11434/v1", api_key="test"
        )
        client = step._resolve_client()
        assert client is not None
        # Trigger lazy creation of the inner OpenAI SDK client
        client._get_client()
        mock_cls.assert_called_once_with(api_key="test", base_url="http://localhost:11434/v1")

    def test_client_reused(self):
        mock_client = AsyncMock()
        step = LLMStep(name="llm", fields=["f1"], client=mock_client)
        c1 = step._resolve_client()
        c2 = step._resolve_client()
        assert c1 is c2


# -- system message building --------------------------------------------


class TestLLMStepMessageBuilding:
    def test_includes_row_and_fields(self):
        step = LLMStep(name="llm", fields=["market_size"])
        ctx = _make_ctx()
        msg = step._build_system_message(ctx)

        assert "Acme" in msg
        assert "market_size" in msg
        assert "--- DATA ---" in msg

    def test_includes_prior_results(self):
        step = LLMStep(name="llm", fields=["f1"])
        ctx = _make_ctx(prior_results={"search_ctx": "relevant info"})
        msg = step._build_system_message(ctx)

        assert "Prior Step Results" in msg
        assert "relevant info" in msg

    def test_omits_prior_results_when_empty(self):
        step = LLMStep(name="llm", fields=["f1"])
        ctx = _make_ctx(prior_results={})
        msg = step._build_system_message(ctx)

        assert "Prior Step Results" not in msg

    def test_custom_system_prompt(self):
        step = LLMStep(name="llm", fields=["f1"], system_prompt="Custom prompt.")
        ctx = _make_ctx()
        msg = step._build_system_message(ctx)

        assert msg.startswith("Custom prompt.")
        assert DEFAULT_SYSTEM_PROMPT not in msg


# -- successful run ------------------------------------------------------


class TestLLMStepRun:
    @pytest.mark.asyncio
    async def test_successful_run(self):
        resp = _mock_llm_response(json.dumps({"market_size": "Large"}))
        mock_client = _make_mock_client(resp)
        step = LLMStep(name="llm", fields=["market_size"], client=mock_client)

        result = await step.run(_make_ctx())

        assert isinstance(result, StepResult)
        assert result.values == {"market_size": "Large"}
        assert result.usage is not None
        assert result.usage.total_tokens == 15
        assert result.usage.model == "gpt-4.1-nano"
        assert result.metadata["attempts"] == 1

    @pytest.mark.asyncio
    async def test_filters_to_declared_fields(self):
        resp = _mock_llm_response(json.dumps({"market_size": "Large", "extra": "junk"}))
        mock_client = _make_mock_client(resp)
        step = LLMStep(name="llm", fields=["market_size"], client=mock_client)

        result = await step.run(_make_ctx())
        assert "extra" not in result.values
        assert result.values == {"market_size": "Large"}

    @pytest.mark.asyncio
    async def test_temperature_from_config(self):
        resp = _mock_llm_response(json.dumps({"f1": "val"}))
        mock_client = _make_mock_client(resp)
        step = LLMStep(name="llm", fields=["f1"], client=mock_client)

        config = SimpleNamespace(temperature=0.1, max_tokens=500, max_workers=3)
        ctx = _make_ctx(config=config)
        await step.run(ctx)

        call_kwargs = mock_client.complete.call_args
        assert call_kwargs.kwargs["temperature"] == 0.1
        assert call_kwargs.kwargs["max_tokens"] == 500


# -- retries on validation/parse error -----------------------------------


class TestLLMStepRetries:
    @pytest.mark.asyncio
    async def test_retries_on_json_error_then_succeeds(self):
        bad_resp = _mock_llm_response("not json at all")
        good_resp = _mock_llm_response(json.dumps({"f1": "ok"}))
        mock_client = _make_mock_client([bad_resp, good_resp])
        step = LLMStep(name="llm", fields=["f1"], client=mock_client, max_retries=2)

        result = await step.run(_make_ctx())
        assert result.values == {"f1": "ok"}
        assert result.metadata["attempts"] == 2

    @pytest.mark.asyncio
    async def test_retries_on_validation_error_then_succeeds(self):
        class Strict(BaseModel):
            f1: int  # must be int, not str

        bad_resp = _mock_llm_response(json.dumps({"f1": "not_an_int"}))
        good_resp = _mock_llm_response(json.dumps({"f1": 42}))
        mock_client = _make_mock_client([bad_resp, good_resp])
        step = LLMStep(
            name="llm", fields=["f1"], client=mock_client, schema=Strict, max_retries=2
        )

        result = await step.run(_make_ctx())
        assert result.values == {"f1": 42}

    @pytest.mark.asyncio
    async def test_raises_after_max_retries_exhausted(self):
        bad_resp = _mock_llm_response("bad json")
        mock_client = _make_mock_client([bad_resp, bad_resp])
        step = LLMStep(name="llm", fields=["f1"], client=mock_client, max_retries=1)

        with pytest.raises(StepError, match="failed after 2 parse attempts"):
            await step.run(_make_ctx())

        # 1 initial + 1 retry = 2 calls
        assert mock_client.complete.call_count == 2


# -- API error retries ---------------------------------------------------


class TestLLMStepAPIRetries:
    @pytest.mark.asyncio
    async def test_retries_on_rate_limit_then_succeeds(self):
        good_resp = _mock_llm_response(json.dumps({"f1": "ok"}))
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(
            side_effect=[
                LLMAPIError("rate limited", status_code=429, is_rate_limit=True),
                good_resp,
            ]
        )
        config = SimpleNamespace(
            temperature=0.2, max_tokens=100, max_retries=2, retry_base_delay=0.01,
        )
        step = LLMStep(name="llm", fields=["f1"], client=mock_client)

        result = await step.run(_make_ctx(config=config))
        assert result.values == {"f1": "ok"}
        assert result.metadata["api_retries"] == 1

    @pytest.mark.asyncio
    async def test_retries_on_timeout_then_succeeds(self):
        good_resp = _mock_llm_response(json.dumps({"f1": "ok"}))
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(
            side_effect=[
                LLMAPIError("timeout", status_code=408),
                good_resp,
            ]
        )
        config = SimpleNamespace(
            temperature=0.2, max_tokens=100, max_retries=2, retry_base_delay=0.01,
        )
        step = LLMStep(name="llm", fields=["f1"], client=mock_client)

        result = await step.run(_make_ctx(config=config))
        assert result.values == {"f1": "ok"}

    @pytest.mark.asyncio
    async def test_api_retries_exhausted_raises(self):
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(
            side_effect=LLMAPIError("rate limited", status_code=429, is_rate_limit=True)
        )
        config = SimpleNamespace(
            temperature=0.2, max_tokens=100, max_retries=1, retry_base_delay=0.01,
        )
        step = LLMStep(name="llm", fields=["f1"], client=mock_client)

        with pytest.raises(StepError, match="API error after 2 retries"):
            await step.run(_make_ctx(config=config))

    @pytest.mark.asyncio
    async def test_respects_retry_after_header(self):
        good_resp = _mock_llm_response(json.dumps({"f1": "ok"}))
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(
            side_effect=[
                LLMAPIError("rate limited", status_code=429, retry_after=0.01, is_rate_limit=True),
                good_resp,
            ]
        )
        config = SimpleNamespace(
            temperature=0.2, max_tokens=100, max_retries=2, retry_base_delay=0.001,
        )
        step = LLMStep(name="llm", fields=["f1"], client=mock_client)

        result = await step.run(_make_ctx(config=config))
        assert result.values == {"f1": "ok"}

    @pytest.mark.asyncio
    async def test_parse_error_inside_api_retry(self):
        """Parse error on first API attempt, API error on second, success on third."""
        bad_parse_resp = _mock_llm_response("not json")
        good_resp = _mock_llm_response(json.dumps({"f1": "ok"}))
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(
            side_effect=[
                bad_parse_resp,  # parse attempt 1 fails
                bad_parse_resp,  # parse attempt 2 fails → parse retries exhausted → StepError
                # But StepError is not LLMAPIError, so it won't trigger API retry
            ]
        )
        config = SimpleNamespace(
            temperature=0.2, max_tokens=100, max_retries=2, retry_base_delay=0.01,
        )
        step = LLMStep(name="llm", fields=["f1"], client=mock_client, max_retries=1)

        # Parse retries (1) exhausted, raises StepError (not caught by API retry)
        with pytest.raises(StepError, match="parse attempts"):
            await step.run(_make_ctx(config=config))


# -- custom schema -------------------------------------------------------


class TestLLMStepCustomSchema:
    @pytest.mark.asyncio
    async def test_custom_schema_validation(self):
        class MySchema(BaseModel):
            score: float
            label: str

        resp = _mock_llm_response(json.dumps({"score": 0.95, "label": "positive"}))
        mock_client = _make_mock_client(resp)
        step = LLMStep(
            name="llm", fields=["score", "label"], client=mock_client, schema=MySchema,
        )

        result = await step.run(_make_ctx())
        assert result.values == {"score": 0.95, "label": "positive"}
