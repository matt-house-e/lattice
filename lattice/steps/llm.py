"""LLMStep — calls an OpenAI-compatible chat model for enrichment."""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Type

from pydantic import BaseModel, ValidationError

from ..core.exceptions import StepError
from ..schemas.base import UsageInfo
from ..schemas.enrichment import EnrichmentResult
from ..utils.logger import get_logger
from .base import StepContext, StepResult

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
    """Calls an OpenAI-compatible chat model to produce enrichment values.

    Features:
      - Lazy client initialisation (no import-time API key check).
      - Builds system message with: system prompt + row data + field specs + prior results.
      - Uses ``response_format={"type": "json_object"}``.
      - Validates response with Pydantic ``model_validate()``.
      - On validation/parse error: appends error to conversation and retries.
    """

    def __init__(
        self,
        name: str,
        fields: list[str],
        depends_on: list[str] | None = None,
        model: str = "gpt-4.1-nano",
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        api_key: str | None = None,
        schema: Type[BaseModel] = EnrichmentResult,
        max_retries: int = 2,
    ):
        self.name = name
        self.fields = fields
        self.depends_on = depends_on or []
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.api_key = api_key
        self.schema = schema
        self.max_retries = max_retries
        self._client: Any = None  # openai.AsyncOpenAI, lazily created

    # -- client ----------------------------------------------------------

    def _get_client(self) -> Any:
        """Lazily create the AsyncOpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI

            key = self.api_key or os.environ.get("OPENAI_API_KEY")
            self._client = AsyncOpenAI(api_key=key)
        return self._client

    # -- message building ------------------------------------------------

    def _build_system_message(self, ctx: StepContext) -> str:
        parts = [self.system_prompt]
        parts.append("\n--- DATA ---")
        parts.append(f"Row Data: {json.dumps(ctx.row, default=str)}")
        parts.append(f"Field Specifications: {json.dumps(ctx.fields, default=str)}")
        if ctx.prior_results:
            parts.append(f"Prior Step Results: {json.dumps(ctx.prior_results, default=str)}")
        return "\n".join(parts)

    # -- run -------------------------------------------------------------

    async def run(self, ctx: StepContext) -> StepResult:
        client = self._get_client()

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

        system_content = self._build_system_message(ctx)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": "Analyze the provided data and return the requested fields as JSON.",
            },
        ]

        last_error: BaseException | None = None
        content: str = ""

        for attempt in range(self.max_retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )

                content = response.choices[0].message.content or ""
                parsed = json.loads(content)
                validated = self.schema.model_validate(parsed)
                all_values = validated.model_dump()

                # Filter to declared fields
                values = {k: v for k, v in all_values.items() if k in self.fields}

                usage = None
                if response.usage:
                    usage = UsageInfo(
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                        model=self.model,
                    )

                return StepResult(
                    values=values,
                    usage=usage,
                    metadata={"raw_response": content, "attempts": attempt + 1},
                )

            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                logger.warning(
                    "LLMStep '%s' attempt %d failed: %s", self.name, attempt + 1, exc
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

        raise StepError(
            f"LLMStep '{self.name}' failed after {self.max_retries + 1} attempts: {last_error}",
            step_name=self.name,
        )
