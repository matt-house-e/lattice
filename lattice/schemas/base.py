"""Base schema types for the Lattice pipeline."""

from pydantic import BaseModel


class UsageInfo(BaseModel):
    """Token usage information from an LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""


class StepUsage(BaseModel):
    """Aggregated token usage for a single pipeline step."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    rows_processed: int = 0
    model: str = ""
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class CostSummary(BaseModel):
    """Aggregated cost/usage across all pipeline steps."""

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    steps: dict[str, StepUsage] = {}
