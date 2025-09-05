"""
Lattice - CSV Enrichment Tool

A powerful tool for enriching CSV data using Large Language Models (LLM) 
on a row-by-row basis with clean, production-ready architecture.
"""

# Import main classes for clean public API using new structure
from .core import TableEnricher, EnrichmentConfig, EnrichmentError, FieldValidationError
from .chains import LLMChain, VectorStoreLLMChain
from .data import FieldManager

__version__ = "0.2.0"
__author__ = "Lattice Team"

__all__ = [
    'TableEnricher', 
    'LLMChain', 
    'VectorStoreLLMChain',
    'FieldManager', 
    'EnrichmentConfig',
    'EnrichmentError',
    'FieldValidationError'
]