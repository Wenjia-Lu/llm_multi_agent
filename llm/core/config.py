"""LLM Configuration Module"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    """Unified configuration for both API and local models."""

    # Required fields (no defaults)
    model_name: str
    model_type: str  # "api" or "local"
    temperature: float
    max_tokens: int
    context_length: int
    parameter_count: int  # Number of parameters (for FLOP calculation)

    # Optional fields (with defaults)
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    model_path: Optional[str] = None
    device: Optional[str] = None
    quantization: Optional[str] = None

    @property
    def is_api(self) -> bool:
        """Check if this is an API-based model."""
        return self.model_type == "api"

    @property
    def is_local(self) -> bool:
        """Check if this is a local model."""
        return self.model_type == "local"
