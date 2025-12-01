"""LLM Framework Package"""

from .core.config import LLMConfig
from .core.response import LLMResponse
from .core.base import LLM
from .factory import LLMFactory
from .implementations.api_llm import APILLM

try:
    from .implementations.local_llm import LocalLLM
except ImportError:
    # LocalLLM may not be implemented yet
    LocalLLM = None

__all__ = [
    "LLMConfig",
    "LLMResponse",
    "LLM",
    "LLMFactory",
    "APILLM",
    "LocalLLM"
]
