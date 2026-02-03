"""Base classes for Pydantic LLM schemas.

Provides common configuration and base classes for all LLM output schemas.
Following patterns from Pydantic best practices for LLM structured outputs.
"""

from pydantic import BaseModel, ConfigDict


class BaseLLMSchema(BaseModel):
    """Base class for all LLM output schemas.

    Provides strict validation configuration to ensure LLM outputs
    conform to expected structure. Key settings:

    - validate_default: Validate even default values
    - use_enum_values: Serialize enums as their values (strings)
    - populate_by_name: Accept both field names and aliases
    - extra="forbid": Reject unknown fields from LLM (catches hallucinations)
    """

    model_config = ConfigDict(
        validate_default=True,
        use_enum_values=True,
        populate_by_name=True,
        extra="forbid"
    )
