"""Convenience re-export of LLM provider adapters.

Usage:
    from lattice.providers import OpenAIClient
    from lattice.providers import AnthropicClient  # requires: pip install lattice[anthropic]
    from lattice.providers import GoogleClient      # requires: pip install lattice[google]
"""

from .steps.providers.base import LLMAPIError, LLMClient, LLMResponse
from .steps.providers.openai import OpenAIClient

# Optional providers — import errors deferred to instantiation
from .steps.providers.anthropic import AnthropicClient
from .steps.providers.google import GoogleClient

__all__ = [
    "LLMClient",
    "LLMResponse",
    "LLMAPIError",
    "OpenAIClient",
    "AnthropicClient",
    "GoogleClient",
]
