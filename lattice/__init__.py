"""Lattice — Enrichment Pipeline Engine.

A programmatic enrichment engine built on composable, column-oriented
pipeline steps with Pydantic validation, checkpointing, and async concurrency.
"""

# Set up default logging when package is imported
from .utils.logger import setup_logging
setup_logging(level="WARNING", format_type="console", include_timestamp=False)

# Public API
from .core import Enricher, EnrichmentConfig, EnrichmentError, FieldValidationError
from .pipeline import Pipeline
from .steps import Step, StepContext, StepResult, FunctionStep, LLMStep
from .data import FieldManager

__version__ = "0.3.0"
__author__ = "Lattice Team"

__all__ = [
    "Enricher",
    "Pipeline",
    "Step",
    "StepContext",
    "StepResult",
    "FunctionStep",
    "LLMStep",
    "FieldManager",
    "EnrichmentConfig",
    "EnrichmentError",
    "FieldValidationError",
]
