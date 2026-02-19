"""Base schema types for the Lattice pipeline."""

from pydantic import BaseModel


class UsageInfo(BaseModel):
    """Token usage information from an LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
