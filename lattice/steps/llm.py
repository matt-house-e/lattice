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
from ..utils.logger import get_logger
from .base import StepContext, StepResult
from .providers.base import LLMAPIError, LLMClient, LLMResponse

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default system prompt — ported from v0.2 LLMChain._create_default_prompt()
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = """\
You are a structured data enrichment engine operating over tabular rows.
Given one input row, a set of field specifications, and optional context \
from prior processing steps, produce a JSON object with EXACTLY the \
requested fields as keys and values that satisfy each field's constraints.

Each field specification contains at least:
  - prompt: the concrete instruction for this field
  - instructions: format/refinement constraints
  - data_type: expected type (e.g., String, Number, Boolean, Date, JSON, List[String])
  - examples: optional examples of ideal outputs

OUTPUT CONTRACT
- Return ONLY a single valid JSON object. No prose, no code fences, no explanations.
- Top-level keys MUST be exactly the field names present in Field Specifications.
- Values MUST comply with each field's data_type and instructions.
- Keep outputs concise and information-dense. Avoid filler language.
- If the row and context are insufficient to answer a field, return \
"Unable to determine" (String), null (for non-String types), or an empty \
list (for list types). Never fabricate sources or numbers.
- When context includes citations or sources and the field's instructions \
ask for sources, include terse source references inline \
(e.g., "… (Reuters, 2024)"). Do not add URLs unless clearly present in context.
- Do not include any keys not requested. Do not include reasoning if not \
explicitly requested.

DECISION GUIDELINES
- If sources contradict, prefer the most specific and recent context; \
otherwise, prefer Row Data.
- Follow examples to style the answer when provided, but never copy them verbatim.
- Normalize simple formatting:
  - Numbers: plain numerals; include units only if requested.
  - Dates: ISO-8601 (YYYY-MM-DD) unless instructions specify another format.
  - Lists: small, ordered by relevance; 4 items max unless otherwise stated.

EXECUTION
For each field:
  1) Read prompt and instructions.
  2) Check Row Data; then consult prior step results if helpful.
  3) Produce an accurate value that satisfies data_type and instructions.
  4) If insufficient evidence, use the fallback policy above without guessing.

Return ONLY the final JSON object with the requested fields as keys."""


class LLMStep:
    """Calls an LLM provider to produce enrichment values.

    Features:
      - Provider-agnostic via LLMClient protocol (OpenAI default).
      - Lazy client initialisation (no import-time API key check).
      - Builds system message with: system prompt + row data + field specs + prior results.
      - Uses ``response_format={"type": "json_object"}``.
      - Validates response with Pydantic ``model_validate()``.
      - On validation/parse error: appends error to conversation and retries.
    """

    def __init__(
        self,
        name: str,
        fields: list[str] | dict[str, str | dict],
        depends_on: list[str] | None = None,
        model: str = "gpt-4.1-nano",
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        client: LLMClient | None = None,
        schema: Type[BaseModel] = EnrichmentResult,
        max_retries: int = 2,
    ):
        self.name = name
        self.depends_on = depends_on or []
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.api_key = api_key
        self.base_url = base_url
        self.schema = schema
        self.max_retries = max_retries
        self._client: LLMClient | None = client

        # Normalize fields: dict → inline specs + field names list
        if isinstance(fields, dict):
            self._field_specs = self._normalize_field_specs(fields)
            self.fields = list(fields.keys())
        else:
            self._field_specs: dict[str, dict[str, Any]] = {}
            self.fields = fields

    @staticmethod
    def _normalize_field_specs(fields: dict[str, str | dict]) -> dict[str, dict[str, Any]]:
        """Convert shorthand field specs to full dicts.

        ``{"market_size": "Estimate TAM"}`` → ``{"market_size": {"prompt": "Estimate TAM"}}``
        ``{"market_size": {"prompt": "...", "type": "String"}}`` → passed through
        """
        result: dict[str, dict[str, Any]] = {}
        for name, spec in fields.items():
            if isinstance(spec, str):
                result[name] = {"prompt": spec}
            else:
                result[name] = dict(spec)
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

    # -- message building ------------------------------------------------

    def _build_system_message(self, ctx: StepContext) -> str:
        parts = [self.system_prompt]
        parts.append("\n--- DATA ---")
        parts.append(f"Row Data: {json.dumps(ctx.row, default=str)}")
        # Use inline field specs if available, otherwise fall back to context
        field_specs = self._field_specs if self._field_specs else ctx.fields
        parts.append(f"Field Specifications: {json.dumps(field_specs, default=str)}")
        if ctx.prior_results:
            parts.append(f"Prior Step Results: {json.dumps(ctx.prior_results, default=str)}")
        return "\n".join(parts)

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
            temperature = 0.5

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
                            response_format={"type": "json_object"},
                        )

                        content = response.content
                        parsed = json.loads(content)
                        validated = self.schema.model_validate(parsed)
                        all_values = validated.model_dump()

                        # Filter to declared fields
                        values = {k: v for k, v in all_values.items() if k in self.fields}

                        return StepResult(
                            values=values,
                            usage=response.usage,
                            metadata={
                                "raw_response": content,
                                "attempts": total_attempts,
                                "api_retries": api_attempt,
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
