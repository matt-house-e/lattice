"""
Core functionality for the Lattice enrichment tool.
"""

from .enricher import TableEnricher
from .processors import RowProcessor
from .config import EnrichmentConfig
from .exceptions import EnrichmentError, FieldValidationError, PartialEnrichmentResult

__all__ = [
    'TableEnricher',
    'RowProcessor', 
    'EnrichmentConfig',
    'EnrichmentError',
    'FieldValidationError',
    'PartialEnrichmentResult'
]