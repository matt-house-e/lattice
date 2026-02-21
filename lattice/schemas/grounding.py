"""Grounding types — config and citation models for provider-level web search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class GroundingConfig(BaseModel):
    """Configuration for provider-level web search grounding.

    Normalised in ``LLMStep.__init__``: ``True`` becomes ``GroundingConfig()``,
    a ``dict`` is validated, ``None`` / ``False`` disables grounding.

    Provider mapping:
      - ``allowed_domains``: OpenAI ``filters.allowed_domains``,
        Anthropic ``allowed_domains``.  Google does not support allow-listing
        (a warning is logged if set).
      - ``blocked_domains``: Anthropic ``blocked_domains``,
        Google ``exclude_domains``.  OpenAI does not support block-listing
        (a warning is logged if set).
      - ``user_location``: All three providers support approximate location.
      - ``max_searches``: Anthropic ``max_uses``.  Other providers ignore.

    Attributes:
        allowed_domains: Only include search results from these domains.
        blocked_domains: Never include search results from these domains.
        user_location: Approximate user location for localised results.
            Keys: ``country`` (ISO 3166-1 alpha-2), ``region``, ``city``,
            ``timezone`` (IANA).
        max_searches: Maximum number of searches per LLM call.
    """

    model_config = ConfigDict(extra="forbid")

    allowed_domains: Optional[List[str]] = None
    blocked_domains: Optional[List[str]] = None
    user_location: Optional[Dict[str, str]] = None
    max_searches: Optional[int] = None


@dataclass
class Citation:
    """A normalised source citation from a grounded LLM response.

    Provider adapters translate their native citation formats into this
    common representation.  Lattice injects these as ``__sources`` on the
    step result (filtered from final output, available in ``prior_results``).

    Attributes:
        url: Source URL.
        title: Page or document title (may be empty).
        snippet: Short extract from the source (may be empty).
    """

    url: str
    title: str = ""
    snippet: str = ""
