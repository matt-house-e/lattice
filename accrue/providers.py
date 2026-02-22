"""Convenience re-export of LLM provider adapters.

Usage:
    from accrue.providers import OpenAIClient
    from accrue.providers import AnthropicClient  # requires: pip install accrue[anthropic]
    from accrue.providers import GoogleClient      # requires: pip install accrue[google]
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
