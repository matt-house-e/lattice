"""Lattice — Enrichment Pipeline Engine.

A programmatic enrichment engine built on composable, column-oriented
pipeline steps with Pydantic validation, checkpointing, and async concurrency.
"""

# Set up default logging when package is imported
from .utils.logger import setup_logging
setup_logging(level="WARNING", format_type="console", include_timestamp=False)

# Public API
from .core import Enricher, EnrichmentConfig, EnrichmentError, FieldValidationError, StepError, PipelineError, RowError
from .core.hooks import (
    EnrichmentHooks,
    PipelineStartEvent,
    PipelineEndEvent,
    StepStartEvent,
    StepEndEvent,
    RowCompleteEvent,
)
from .pipeline import Pipeline, PipelineResult
from .steps import Step, StepContext, StepResult, FunctionStep, LLMStep
from .schemas.base import CostSummary
from .schemas.field_spec import FieldSpec
from .utils.web_search import web_search

__version__ = "0.5.0"
__author__ = "Lattice Team"

__all__ = [
    # Primary API
    "Pipeline",
    "PipelineResult",
    "LLMStep",
    "FunctionStep",
    "EnrichmentConfig",
    # Hooks
    "EnrichmentHooks",
    "PipelineStartEvent",
    "PipelineEndEvent",
    "StepStartEvent",
    "StepEndEvent",
    "RowCompleteEvent",
    # Step protocol
    "Step",
    "StepContext",
    "StepResult",
    # Schemas
    "FieldSpec",
    # Utilities
    "web_search",
    # Results & errors
    "CostSummary",
    "RowError",
    "EnrichmentError",
    "FieldValidationError",
    "StepError",
    "PipelineError",
    # Internal runner (power users)
    "Enricher",
]
