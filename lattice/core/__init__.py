"""Core functionality for the Lattice enrichment tool."""

from .enricher import Enricher
from .checkpoint import CheckpointManager, CheckpointData
from .config import EnrichmentConfig
from .exceptions import (
    EnrichmentError,
    FieldValidationError,
    StepError,
    PipelineError,
    PartialEnrichmentResult,
)

__all__ = [
    "Enricher",
    "CheckpointManager",
    "CheckpointData",
    "EnrichmentConfig",
    "EnrichmentError",
    "FieldValidationError",
    "StepError",
    "PipelineError",
    "PartialEnrichmentResult",
]
