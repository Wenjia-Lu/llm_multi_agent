"""LLM Response Module"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class LLMResponse:
    """Unified response containing text, metadata, and extended data."""

    # Core output (required)
    text: str
    finish_reason: str
    confidence_score: float

    # Token usage (optional with defaults)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Extended data (optional)
    flops_used: Optional[float] = None

    # Raw data for advanced users
    logits: Optional[object] = None
    token_probs: Optional[list] = None

    # Metadata
    model_name: str = ""
    generation_time: float = 0.0
    error: Optional[str] = None
