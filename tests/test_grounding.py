"""Tests for provider-level grounding on LLMStep (Issue #48)."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from lattice import GroundingConfig, LLMStep, Pipeline
from lattice.core.cache import _compute_step_cache_key
from lattice.core.exceptions import PipelineError, StepError
from lattice.schemas.grounding import Citation, GroundingConfig
from lattice.steps.base import StepContext, StepResult
from lattice.steps.providers.base import LLMResponse
from lattice.steps.providers.openai import OpenAIClient


# ============================================================================
# GroundingConfig
# ============================================================================


class TestGroundingConfig:
    def test_default(self):
        cfg = GroundingConfig()
        assert cfg.allowed_domains is None
        assert cfg.blocked_domains is None
        assert cfg.user_location is None
        assert cfg.max_searches is None

    def test_full_config(self):
        cfg = GroundingConfig(
            allowed_domains=["crunchbase.com", "sec.gov"],
            blocked_domains=["reddit.com"],
            user_location={"country": "US", "city": "NYC"},
            max_searches=3,
        )
        assert cfg.allowed_domains == ["crunchbase.com", "sec.gov"]
        assert cfg.blocked_domains == ["reddit.com"]
        assert cfg.user_location == {"country": "US", "city": "NYC"}
        assert cfg.max_searches == 3

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError, match="extra"):
            GroundingConfig(unknown_field="bad")

    def test_from_dict(self):
        cfg = GroundingConfig.model_validate({"allowed_domains": ["example.com"]})
        assert cfg.allowed_domains == ["example.com"]


class TestCitation:
    def test_defaults(self):
        c = Citation(url="https://example.com")
        assert c.url == "https://example.com"
        assert c.title == ""
        assert c.snippet == ""

    def test_full(self):
        c = Citation(url="https://example.com", title="Test", snippet="A snippet")
        assert c.title == "Test"
        assert c.snippet == "A snippet"


# ============================================================================
# LLMStep grounding normalization
# ============================================================================


class TestLLMStepGroundingInit:
    def test_grounding_none(self):
        step = LLMStep("s", fields=["f1"])
        assert step._grounding_config is None

    def test_grounding_false(self):
        step = LLMStep("s", fields=["f1"], grounding=False)
        assert step._grounding_config is None

    def test_grounding_true(self):
        step = LLMStep("s", fields=["f1"], grounding=True)
        assert isinstance(step._grounding_config, GroundingConfig)
        assert step._grounding_config.allowed_domains is None

    def test_grounding_dict(self):
        step = LLMStep("s", fields=["f1"], grounding={
            "allowed_domains": ["example.com"],
            "max_searches": 5,
        })
        assert step._grounding_config.allowed_domains == ["example.com"]
        assert step._grounding_config.max_searches == 5

    def test_grounding_config_instance(self):
        cfg = GroundingConfig(blocked_domains=["bad.com"])
        step = LLMStep("s", fields=["f1"], grounding=cfg)
        assert step._grounding_config is cfg

    def test_grounding_invalid_dict(self):
        with pytest.raises(ValidationError, match="extra"):
            LLMStep("s", fields=["f1"], grounding={"bad_key": True})

    def test_grounding_invalid_type(self):
        with pytest.raises(PipelineError, match="Invalid grounding value"):
            LLMStep("s", fields=["f1"], grounding=42)


# ============================================================================
# LLMStep._build_tools_config()
# ============================================================================


class TestBuildToolsConfig:
    def test_no_grounding_returns_none(self):
        step = LLMStep("s", fields=["f1"])
        assert step._build_tools_config() is None

    def test_grounding_true_returns_web_search(self):
        step = LLMStep("s", fields=["f1"], grounding=True)
        tools = step._build_tools_config()
        assert tools == [{"type": "web_search"}]

    def test_grounding_with_config(self):
        step = LLMStep("s", fields=["f1"], grounding={
            "allowed_domains": ["a.com"],
            "blocked_domains": ["b.com"],
            "user_location": {"country": "GB"},
            "max_searches": 2,
        })
        tools = step._build_tools_config()
        assert len(tools) == 1
        tool = tools[0]
        assert tool["type"] == "web_search"
        assert tool["allowed_domains"] == ["a.com"]
        assert tool["blocked_domains"] == ["b.com"]
        assert tool["user_location"] == {"country": "GB"}
        assert tool["max_searches"] == 2


# ============================================================================
# LLMStep.run() — grounding integration (mocked)
# ============================================================================


class TestLLMStepRunWithGrounding:
    """Test that run() passes tools to client and injects __sources."""

    @pytest.mark.asyncio
    async def test_tools_passed_and_sources_injected(self):
        """When grounding is enabled, tools are passed to client and __sources injected."""
        citations = [
            Citation(url="https://example.com", title="Example", snippet="The answer"),
        ]
        mock_response = LLMResponse(
            content='{"summary": "AI is growing"}',
            citations=citations,
        )
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(return_value=mock_response)

        step = LLMStep(
            "s",
            fields={"summary": "Summarize the topic"},
            grounding=True,
            client=mock_client,
        )

        ctx = StepContext(
            row={"topic": "AI"},
            fields={"summary": {"prompt": "Summarize the topic"}},
            prior_results={},
        )
        result = await step.run(ctx)

        # Tools were passed
        call_kwargs = mock_client.complete.call_args.kwargs
        assert call_kwargs["tools"] == [{"type": "web_search"}]

        # __sources injected
        assert "__sources" in result.values
        assert len(result.values["__sources"]) == 1
        assert result.values["__sources"][0]["url"] == "https://example.com"

        # Regular field present
        assert result.values["summary"] == "AI is growing"

    @pytest.mark.asyncio
    async def test_no_citations_no_sources(self):
        """When grounding returns no citations, __sources is NOT injected."""
        mock_response = LLMResponse(
            content='{"summary": "AI is growing"}',
            citations=[],
        )
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(return_value=mock_response)

        step = LLMStep(
            "s",
            fields={"summary": "Summarize"},
            grounding=True,
            client=mock_client,
        )

        ctx = StepContext(
            row={"topic": "AI"},
            fields={"summary": {"prompt": "Summarize"}},
            prior_results={},
        )
        result = await step.run(ctx)

        assert "__sources" not in result.values

    @pytest.mark.asyncio
    async def test_no_grounding_no_tools(self):
        """When grounding is disabled, tools=None is passed."""
        mock_response = LLMResponse(content='{"f1": "val"}')
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(return_value=mock_response)

        step = LLMStep("s", fields={"f1": "prompt"}, client=mock_client)

        ctx = StepContext(
            row={"x": 1},
            fields={"f1": {"prompt": "prompt"}},
            prior_results={},
        )
        await step.run(ctx)

        call_kwargs = mock_client.complete.call_args.kwargs
        assert call_kwargs["tools"] is None

    @pytest.mark.asyncio
    async def test_type_error_caught_for_custom_client(self):
        """TypeError from client.complete() gives a clear error when grounding enabled."""

        class OldClient:
            async def complete(self, messages, model, temperature, max_tokens,
                               response_format=None):
                return LLMResponse(content="{}")

        step = LLMStep("s", fields=["f1"], grounding=True, client=OldClient())

        ctx = StepContext(
            row={"x": 1},
            fields={},
            prior_results={},
        )

        with pytest.raises(StepError, match="does not support the 'tools' parameter"):
            await step.run(ctx)


# ============================================================================
# Cache key includes grounding hash
# ============================================================================


class TestCacheKeyGrounding:
    def _make_step(self, grounding=None):
        return LLMStep("test_step", fields=["f1"], model="gpt-4.1-mini", grounding=grounding)

    def test_no_grounding_backward_compat(self):
        """Cache key without grounding includes empty grounding_hash."""
        step = self._make_step()
        key = _compute_step_cache_key(step, {"a": 1}, {}, {"f1": {}})
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex

    def test_grounding_changes_cache_key(self):
        """Different grounding configs produce different cache keys."""
        step_none = self._make_step(grounding=None)
        step_true = self._make_step(grounding=True)
        step_cfg = self._make_step(grounding={"allowed_domains": ["x.com"]})

        row = {"a": 1}
        prior = {}
        fields = {"f1": {}}

        key_none = _compute_step_cache_key(step_none, row, prior, fields)
        key_true = _compute_step_cache_key(step_true, row, prior, fields)
        key_cfg = _compute_step_cache_key(step_cfg, row, prior, fields)

        assert key_none != key_true
        assert key_true != key_cfg
        assert key_none != key_cfg

    def test_same_grounding_same_key(self):
        """Same grounding config produces the same cache key."""
        step1 = self._make_step(grounding={"max_searches": 3})
        step2 = self._make_step(grounding={"max_searches": 3})

        row = {"a": 1}
        prior = {}
        fields = {"f1": {}}

        assert _compute_step_cache_key(step1, row, prior, fields) == \
               _compute_step_cache_key(step2, row, prior, fields)


# ============================================================================
# Export check
# ============================================================================


class TestExports:
    def test_grounding_config_importable(self):
        from lattice import GroundingConfig
        assert GroundingConfig is not None

    def test_grounding_config_in_all(self):
        import lattice
        assert "GroundingConfig" in lattice.__all__
