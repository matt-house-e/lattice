"""LLMStep — calls an LLM provider for structured enrichment."""

from __future__ import annotations

import asyncio
import json
import random
from typing import Any, Optional, Type

from pydantic import BaseModel, ValidationError

from ..core.exceptions import StepError
from ..schemas.base import UsageInfo
from ..schemas.enrichment import EnrichmentResult
from ..schemas.field_spec import FieldSpec
from ..utils.logger import get_logger
from .base import StepContext, StepResult
from .prompt_builder import build_system_message
from .providers.base import LLMAPIError, LLMClient, LLMResponse

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Refusal detection for default enforcement
# ---------------------------------------------------------------------------
REFUSAL_PATTERNS = frozenset({
    "unable to determine",
    "n/a",
    "not available",
    "not specified",
    "insufficient data",
    "unknown",
    "not enough information",
    "cannot determine",
    "no data",
    "no information",
    "not applicable",
    "data not available",
    "information not available",
})


class LLMStep:
    """Calls an LLM provider to produce enrichment values.

    Features:
      - Provider-agnostic via LLMClient protocol (OpenAI default).
      - Lazy client initialisation (no import-time API key check).
      - 7-key field spec validation via :class:`FieldSpec` on construction.
      - Dynamic system prompt (markdown headers + XML data boundaries).
      - Default enforcement: replaces LLM refusals with field ``default`` values.
      - Uses ``response_format={"type": "json_object"}``.
      - Validates response with Pydantic ``model_validate()``.
      - On validation/parse error: appends error to conversation and retries.
    """

    def __init__(
        self,
        name: str,
        fields: list[str] | dict[str, str | dict],
        depends_on: list[str] | None = None,
        model: str = "gpt-4.1-mini",
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        system_prompt_header: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        client: LLMClient | None = None,
        schema: Type[BaseModel] = EnrichmentResult,
        max_retries: int = 2,
        cache: bool = True,
        structured_outputs: bool | None = None,
    ):
        self.name = name
        self.depends_on = depends_on or []
        self.model = model
        self.temperature = temperature
        self.cache = cache
        self.max_tokens = max_tokens
        self._custom_system_prompt = system_prompt
        self._system_prompt_header = system_prompt_header
        self.api_key = api_key
        self.base_url = base_url
        self.schema = schema
        self.max_retries = max_retries
        self._client: LLMClient | None = client
        self._structured_outputs_param = structured_outputs

        # Normalize fields: dict → inline FieldSpec objects + field names list
        if isinstance(fields, dict):
            self._field_specs = self._normalize_field_specs(fields)
            self.fields = list(fields.keys())
        else:
            self._field_specs: dict[str, FieldSpec] = {}
            self.fields = fields

        # Build and cache structured outputs format (field specs are immutable)
        self._response_format = self._build_response_format()
        self._use_structured_outputs = self._response_format.get("type") == "json_schema"

    @staticmethod
    def _normalize_field_specs(fields: dict[str, str | dict]) -> dict[str, FieldSpec]:
        """Convert shorthand field specs to validated FieldSpec objects.

        ``{"market_size": "Estimate TAM"}`` → ``{"market_size": FieldSpec(prompt="Estimate TAM")}``
        ``{"market_size": {"prompt": "...", "type": "String"}}`` → validated FieldSpec
        """
        result: dict[str, FieldSpec] = {}
        for name, spec in fields.items():
            if isinstance(spec, str):
                result[name] = FieldSpec(prompt=spec)
            else:
                result[name] = FieldSpec.model_validate(spec)
        return result

    # -- client ----------------------------------------------------------

    def _resolve_client(self) -> LLMClient:
        """Lazily create or return the LLMClient."""
        if self._client is None:
            from .providers.openai import OpenAIClient

            self._client = OpenAIClient(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    # -- response format -------------------------------------------------

    def _build_response_format(self) -> dict:
        """Determine the response_format based on auto-detection or explicit override.

        Auto-detect logic:
          - Custom schema → json_object (user manages validation)
          - No field specs (list fields) → json_object
          - Non-OpenAI client → json_object
          - OpenAI with base_url and structured_outputs is None → json_object
          - OpenAI native (no base_url) → json_schema
          - structured_outputs=True → force json_schema
          - structured_outputs=False → force json_object
        """
        from .providers.openai import OpenAIClient

        # Explicit override
        if self._structured_outputs_param is False:
            return {"type": "json_object"}

        if self._structured_outputs_param is True:
            # Force on — requires field specs
            if self._field_specs:
                from .schema_builder import build_json_schema
                return build_json_schema(self._field_specs)
            return {"type": "json_object"}

        # Auto-detect (structured_outputs is None)

        # Custom schema → json_object
        if self.schema is not EnrichmentResult:
            return {"type": "json_object"}

        # No field specs (list[str] fields) → json_object
        if not self._field_specs:
            return {"type": "json_object"}

        # Known providers that support structured outputs (json_schema)
        if self._client is not None and not isinstance(self._client, OpenAIClient):
            # Check for Anthropic/Google — both support json_schema now
            _supports_structured = False
            try:
                from .providers.anthropic import AnthropicClient
                if isinstance(self._client, AnthropicClient):
                    _supports_structured = True
            except ImportError:
                pass
            try:
                from .providers.google import GoogleClient
                if isinstance(self._client, GoogleClient):
                    _supports_structured = True
            except ImportError:
                pass

            if _supports_structured:
                from .schema_builder import build_json_schema
                return build_json_schema(self._field_specs)

            # Unknown custom client → json_object (safe fallback)
            return {"type": "json_object"}

        # OpenAI with base_url → json_object (third-party compatibility)
        if self.base_url is not None:
            return {"type": "json_object"}

        # Native OpenAI → json_schema
        from .schema_builder import build_json_schema
        return build_json_schema(self._field_specs)

    # -- message building ------------------------------------------------

    def _build_system_message(self, ctx: StepContext) -> str:
        """Build the full system message using the dynamic prompt builder."""
        return build_system_message(
            field_specs=self._field_specs,
            row=ctx.row,
            prior_results=ctx.prior_results or None,
            custom_system_prompt=self._custom_system_prompt,
            system_prompt_header=self._system_prompt_header,
        )

    # -- default enforcement ---------------------------------------------

    def _apply_defaults(self, values: dict[str, Any]) -> dict[str, Any]:
        """Replace refusal values with field defaults where configured."""
        for field_name, spec in self._field_specs.items():
            if field_name not in values:
                continue
            if "default" not in spec.model_fields_set:
                continue
            if _is_refusal(values[field_name]):
                values[field_name] = spec.default
        return values

    # -- run -------------------------------------------------------------

    async def run(self, ctx: StepContext) -> StepResult:
        """Execute the LLM call with two-layer retry.

        Outer loop: API errors (429, 500, timeouts) with exponential backoff.
            Uses config.max_retries and config.retry_base_delay.
        Inner loop: Parse/validation errors fed back to the LLM.
            Uses self.max_retries (step-level).
        """
        client = self._resolve_client()

        temperature = self.temperature
        if temperature is None and ctx.config is not None:
            temperature = getattr(ctx.config, "temperature", None)
        if temperature is None:
            temperature = 0.2

        max_tokens = self.max_tokens
        if max_tokens is None and ctx.config is not None:
            max_tokens = getattr(ctx.config, "max_tokens", None)
        if max_tokens is None:
            max_tokens = 4000

        # API retry config from EnrichmentConfig
        api_max_retries = 3
        retry_base_delay = 1.0
        if ctx.config is not None:
            api_max_retries = getattr(ctx.config, "max_retries", api_max_retries)
            retry_base_delay = getattr(ctx.config, "retry_base_delay", retry_base_delay)

        system_content = self._build_system_message(ctx)

        last_api_error: BaseException | None = None
        total_attempts = 0

        for api_attempt in range(api_max_retries + 1):
            # Reset messages for each API retry (parse retries accumulate within)
            messages: list[dict[str, str]] = [
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": "Analyze the provided data and return the requested fields as JSON.",
                },
            ]

            last_parse_error: BaseException | None = None
            content: str = ""

            try:
                for parse_attempt in range(self.max_retries + 1):
                    total_attempts += 1
                    try:
                        response: LLMResponse = await client.complete(
                            messages=messages,
                            model=self.model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            response_format=self._response_format,
                        )

                        content = response.content
                        parsed = json.loads(content)

                        # Validate: use dynamic model for structured outputs,
                        # otherwise fall back to self.schema
                        if self._use_structured_outputs:
                            from .schema_builder import build_response_model
                            dynamic_model = build_response_model(self._field_specs)
                            validated = dynamic_model.model_validate(parsed)
                        else:
                            validated = self.schema.model_validate(parsed)
                        all_values = validated.model_dump()

                        # Filter to declared fields
                        values = {k: v for k, v in all_values.items() if k in self.fields}

                        # Apply default enforcement
                        values = self._apply_defaults(values)

                        return StepResult(
                            values=values,
                            usage=response.usage,
                            metadata={
                                "raw_response": content,
                                "attempts": total_attempts,
                                "api_retries": api_attempt,
                                "structured_outputs": self._use_structured_outputs,
                            },
                        )

                    except (json.JSONDecodeError, ValidationError) as exc:
                        last_parse_error = exc
                        logger.warning(
                            "LLMStep '%s' parse attempt %d failed: %s",
                            self.name, parse_attempt + 1, exc,
                        )
                        # Feed the error back so the LLM can self-correct
                        messages.append({"role": "assistant", "content": content})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"Your response was invalid: {exc}. "
                                    "Please return valid JSON matching the required schema."
                                ),
                            },
                        )

                # All parse retries exhausted
                raise StepError(
                    f"LLMStep '{self.name}' failed after {self.max_retries + 1} "
                    f"parse attempts: {last_parse_error}",
                    step_name=self.name,
                )

            except LLMAPIError as exc:
                last_api_error = exc
                if api_attempt < api_max_retries:
                    # Exponential backoff with jitter
                    delay = retry_base_delay * (2 ** api_attempt)
                    # Respect Retry-After header if available
                    if exc.retry_after is not None:
                        delay = max(delay, exc.retry_after)
                    # Add jitter (0-25% of delay)
                    delay += random.uniform(0, delay * 0.25)
                    logger.warning(
                        "LLMStep '%s' API error (attempt %d/%d), retrying in %.1fs: %s",
                        self.name, api_attempt + 1, api_max_retries + 1, delay, exc,
                    )
                    await asyncio.sleep(delay)

        raise StepError(
            f"LLMStep '{self.name}' API error after {api_max_retries + 1} retries: {last_api_error}",
            step_name=self.name,
        )


def _is_refusal(value: Any) -> bool:
    """Check if a value looks like an LLM refusal."""
    if value is None:
        return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized == "" or normalized in REFUSAL_PATTERNS
    return False
