"""Type-safe wrappers for structured LLM responses.

Provides generic wrapper types that combine validated Pydantic outputs
with metadata like token usage and the full prompt sent to the LLM.
"""

from typing import TypeVar, Generic
from pydantic import BaseModel, ConfigDict, Field


T = TypeVar('T')


class UsageInfo(BaseModel):
    """Token usage information for LLM calls.

    Tracks prompt and completion tokens to enable cost monitoring
    and optimization of LLM usage patterns.
    """

    prompt_tokens: int = Field(ge=0, description="Number of tokens in the prompt")
    completion_tokens: int = Field(ge=0, description="Number of tokens in the completion")

    @property
    def total_tokens(self) -> int:
        """Total tokens used (prompt + completion)."""
        return self.prompt_tokens + self.completion_tokens


class StructuredResult(BaseModel, Generic[T]):
    """Type-safe wrapper for validated LLM responses.

    Combines the validated Pydantic model result with metadata about
    the LLM call, enabling type-safe access to results with full
    IDE autocomplete support.

    Usage:
        result: StructuredResult[EnrichmentResult] = await chain.acomplete_structured(...)
        enrichment = result.data  # Type: EnrichmentResult with autocomplete
        tokens = result.usage.total_tokens
        debug_prompt = result.full_prompt

    Attributes:
        data: The validated Pydantic model instance
        usage: Token usage statistics
        full_prompt: The exact prompt sent to the LLM (for debugging/logging)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: T = Field(description="The validated Pydantic model result")
    usage: UsageInfo = Field(description="Token usage information")
    full_prompt: str = Field(default="", description="Exact prompt sent to LLM API")

    @property
    def prompt_tokens(self) -> int:
        """Quick access to prompt tokens."""
        return self.usage.prompt_tokens

    @property
    def completion_tokens(self) -> int:
        """Quick access to completion tokens."""
        return self.usage.completion_tokens

    @property
    def total_tokens(self) -> int:
        """Quick access to total tokens."""
        return self.usage.total_tokens
