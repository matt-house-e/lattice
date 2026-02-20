"""Step protocol and data classes for the Lattice pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

from ..schemas.base import UsageInfo

if TYPE_CHECKING:
    from ..core.config import EnrichmentConfig


@dataclass(frozen=True)
class StepContext:
    """Immutable context passed to each step.

    Attributes:
        row: Original row data (converted from pd.Series at the Enricher boundary).
        fields: Field specs for THIS step only (sliced by Pipeline).
        prior_results: Merged outputs from dependency steps.
        config: Optional EnrichmentConfig.
    """

    row: dict[str, Any]
    fields: dict[str, dict[str, Any]]
    prior_results: dict[str, Any]
    config: EnrichmentConfig | None = None


@dataclass
class StepResult:
    """Output from a single step execution.

    Attributes:
        values: Field name -> produced value.
        usage: Token usage (LLM steps only).
        metadata: Arbitrary metadata for logging/debugging.
    """

    values: dict[str, Any]
    usage: Optional[UsageInfo] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Step(Protocol):
    """Protocol all steps must satisfy.

    Async-only to eliminate the sync/async duplication from v0.2.
    Implementations can be plain classes — no inheritance required.
    """

    name: str
    fields: list[str]
    depends_on: list[str]

    async def run(self, ctx: StepContext) -> StepResult: ...
