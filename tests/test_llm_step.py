"""Tests for LLMStep (OpenAI calls are mocked)."""

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


# -- helpers -------------------------------------------------------------


def _make_ctx(**overrides: Any) -> StepContext:
    defaults: dict[str, Any] = dict(
        row={"company": "Acme", "industry": "Tech"},
        fields={"market_size": {"prompt": "Estimate market size", "data_type": "String"}},
        prior_results={},
    )
    defaults.update(overrides)
    return StepContext(**defaults)


def _mock_response(content: str, prompt_tokens: int = 10, completion_tokens: int = 5) -> Any:
    """Build a fake OpenAI ChatCompletion response."""
    choice = SimpleNamespace(message=SimpleNamespace(content=content))
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


# -- construction --------------------------------------------------------


class TestLLMStepConstruction:
    def test_satisfies_protocol(self):
        step = LLMStep(name="llm", fields=["market_size"])
        assert isinstance(step, Step)

    def test_defaults(self):
        step = LLMStep(name="llm", fields=["f1"])
        assert step.depends_on == []
        assert step.model == "gpt-4o-mini"
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


# -- lazy client ---------------------------------------------------------


class TestLLMStepClient:
    def test_client_not_created_at_init(self):
        step = LLMStep(name="llm", fields=["f1"], api_key="test-key")
        assert step._client is None

    @patch("openai.AsyncOpenAI")
    def test_client_created_on_first_call(self, mock_cls):
        step = LLMStep(name="llm", fields=["f1"], api_key="test-key")
        client = step._get_client()
        mock_cls.assert_called_once_with(api_key="test-key")
        assert client is mock_cls.return_value

    @patch("openai.AsyncOpenAI")
    def test_client_reused(self, mock_cls):
        step = LLMStep(name="llm", fields=["f1"], api_key="test-key")
        c1 = step._get_client()
        c2 = step._get_client()
        assert c1 is c2
        assert mock_cls.call_count == 1


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
        step = LLMStep(name="llm", fields=["market_size"], api_key="fake")
        resp = _mock_response(json.dumps({"market_size": "Large"}))

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=resp)
        step._client = mock_client

        result = await step.run(_make_ctx())

        assert isinstance(result, StepResult)
        assert result.values == {"market_size": "Large"}
        assert result.usage is not None
        assert result.usage.total_tokens == 15
        assert result.usage.model == "gpt-4o-mini"
        assert result.metadata["attempts"] == 1

    @pytest.mark.asyncio
    async def test_filters_to_declared_fields(self):
        step = LLMStep(name="llm", fields=["market_size"], api_key="fake")
        resp = _mock_response(json.dumps({"market_size": "Large", "extra": "junk"}))

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=resp)
        step._client = mock_client

        result = await step.run(_make_ctx())
        assert "extra" not in result.values
        assert result.values == {"market_size": "Large"}

    @pytest.mark.asyncio
    async def test_temperature_from_config(self):
        step = LLMStep(name="llm", fields=["f1"], api_key="fake")
        resp = _mock_response(json.dumps({"f1": "val"}))

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=resp)
        step._client = mock_client

        config = SimpleNamespace(temperature=0.1, max_tokens=500, max_workers=3)
        ctx = _make_ctx(config=config)
        await step.run(ctx)

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.1
        assert call_kwargs.kwargs["max_tokens"] == 500


# -- retries on validation/parse error -----------------------------------


class TestLLMStepRetries:
    @pytest.mark.asyncio
    async def test_retries_on_json_error_then_succeeds(self):
        step = LLMStep(name="llm", fields=["f1"], api_key="fake", max_retries=2)

        bad_resp = _mock_response("not json at all")
        good_resp = _mock_response(json.dumps({"f1": "ok"}))

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[bad_resp, good_resp])
        step._client = mock_client

        result = await step.run(_make_ctx())
        assert result.values == {"f1": "ok"}
        assert result.metadata["attempts"] == 2

    @pytest.mark.asyncio
    async def test_retries_on_validation_error_then_succeeds(self):
        class Strict(BaseModel):
            f1: int  # must be int, not str

        step = LLMStep(name="llm", fields=["f1"], api_key="fake", schema=Strict, max_retries=2)

        bad_resp = _mock_response(json.dumps({"f1": "not_an_int"}))
        good_resp = _mock_response(json.dumps({"f1": 42}))

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[bad_resp, good_resp])
        step._client = mock_client

        result = await step.run(_make_ctx())
        assert result.values == {"f1": 42}

    @pytest.mark.asyncio
    async def test_raises_after_max_retries_exhausted(self):
        step = LLMStep(name="llm", fields=["f1"], api_key="fake", max_retries=1)

        bad_resp = _mock_response("bad json")

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=bad_resp)
        step._client = mock_client

        with pytest.raises(StepError, match="failed after 2 attempts"):
            await step.run(_make_ctx())

        # 1 initial + 1 retry = 2 calls
        assert mock_client.chat.completions.create.call_count == 2


# -- custom schema -------------------------------------------------------


class TestLLMStepCustomSchema:
    @pytest.mark.asyncio
    async def test_custom_schema_validation(self):
        class MySchema(BaseModel):
            score: float
            label: str

        step = LLMStep(
            name="llm",
            fields=["score", "label"],
            api_key="fake",
            schema=MySchema,
        )
        resp = _mock_response(json.dumps({"score": 0.95, "label": "positive"}))

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=resp)
        step._client = mock_client

        result = await step.run(_make_ctx())
        assert result.values == {"score": 0.95, "label": "positive"}
