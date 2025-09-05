"""
Unified configuration for the Lattice enrichment tool.

Consolidates all configuration options into a single, well-documented
configuration class with sensible defaults.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any
from pathlib import Path


@dataclass
class EnrichmentConfig:
    """
    Unified configuration for enrichment processes.
    
    Consolidates all configuration options that were previously
    scattered across multiple config classes.
    """
    
    # === LLM Configuration ===
    max_tokens: int = 4000
    """Maximum tokens for LLM output"""
    
    temperature: float = 0.5
    """LLM temperature (0.0-1.0)"""
    
    context_window: int = 20000
    """Size of context window for LLM"""
    
    # === Processing Configuration ===
    row_delay: float = 1.0
    """Delay between processing rows in seconds (rate limiting)"""
    
    batch_size: int = 10
    """Number of rows to process in each batch"""
    
    max_workers: int = 3
    """Maximum number of concurrent workers for async processing"""
    
    overwrite_fields: bool = False
    """Whether to overwrite existing field values"""
    
    # === Performance & Reliability ===
    enable_retries: bool = True
    """Enable automatic retries for failed requests"""
    
    max_retries: int = 3
    """Maximum number of retries for failed requests"""
    
    retry_delay: float = 2.0
    """Delay between retries in seconds"""
    
    # === Async & FastAPI Support ===
    enable_async: bool = False
    """Enable asynchronous processing"""
    
    progress_callback: Optional[Callable[[int, int], None]] = None
    """Optional callback for progress updates (current, total)"""
    
    # === Caching (Future) ===
    enable_caching: bool = False
    """Enable caching of enrichment results"""
    
    cache_ttl: int = 3600
    """Cache time-to-live in seconds"""
    
    # === Vector Store Configuration ===
    vector_store_path: Optional[str] = None
    """Path to vector store directory"""
    
    similarity_threshold: float = 0.8
    """Minimum similarity score for vector store matches"""
    
    # === Logging Configuration ===
    log_level: str = "INFO"
    """Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"""
    
    log_dir: Optional[str] = None
    """Directory for log files (None = no file logging)"""
    
    enable_progress_bar: bool = True
    """Enable progress bar display"""
    
    # === Validation ===
    def __post_init__(self):
        """Validate configuration values after initialization."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {self.temperature}")
            
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
            
        if self.row_delay < 0:
            raise ValueError(f"row_delay must be non-negative, got {self.row_delay}")
            
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
            
        if self.max_workers <= 0:
            raise ValueError(f"max_workers must be positive, got {self.max_workers}")
            
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(f"similarity_threshold must be between 0.0 and 1.0, got {self.similarity_threshold}")
    
    @classmethod
    def for_development(cls) -> 'EnrichmentConfig':
        """Create configuration optimized for development."""
        return cls(
            row_delay=0.5,  # Faster for testing
            batch_size=5,   # Smaller batches
            max_workers=2,  # Less resource usage
            enable_progress_bar=True,
            log_level="DEBUG"
        )
    
    @classmethod
    def for_production(cls) -> 'EnrichmentConfig':
        """Create configuration optimized for production."""
        return cls(
            row_delay=2.0,      # Respect rate limits
            batch_size=20,      # Larger batches
            max_workers=5,      # More concurrency
            enable_async=True,  # Async processing
            enable_caching=True,# Cache results
            enable_retries=True,# Retry on failure
            log_level="INFO"
        )
    
    @classmethod
    def for_fast_api(cls) -> 'EnrichmentConfig':
        """Create configuration optimized for FastAPI services."""
        return cls(
            enable_async=True,      # Must be async for FastAPI
            batch_size=50,          # Large batches for efficiency
            max_workers=10,         # High concurrency
            enable_caching=True,    # Cache for duplicate requests
            enable_retries=True,    # Resilience
            enable_progress_bar=False,  # No console output
            log_level="WARNING"     # Less verbose logging
        )