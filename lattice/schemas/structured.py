"""Generic structured result wrapper.

Kept in schemas/ for standalone use but NOT used in the core pipeline flow.
LLMStep.run() returns StepResult directly.
"""

from __future__ import annotations

from typing import Generic, Optional, TypeVar

from pydantic import BaseModel

from .base import UsageInfo

T = TypeVar("T", bound=BaseModel)


class StructuredResult(BaseModel, Generic[T]):
    """Wrapper for structured LLM results with metadata."""

    data: T
    usage: Optional[UsageInfo] = None
    model: str = ""
    raw_response: str = ""
