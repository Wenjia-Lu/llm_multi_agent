"""LLM Abstract Base Class Module"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

from .config import LLMConfig
from .response import LLMResponse


class LLM(ABC):
    """Simplified LLM interface with core functionality."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._setup()

    @abstractmethod
    def _setup(self):
        """Setup model/tokenizer/client."""
        raise NotImplementedError("Subclasses must implement _setup")

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate text from prompt.

        Core method - automatically includes confidence intervals in response.
        """
        raise NotImplementedError("Subclasses must implement generate")

    @abstractmethod
    def _calculate_confidence_score(self, raw_data) -> float:
        """
        Calculate confidence score using the paper's formula:
        c₁ = (1 / |r₁|) * Σ(i=1 to |r₁|) P(ti)

        Args:
            raw_data: Raw model outputs (logits for local, response for API)

        Returns:
            Float confidence score between 0 and 1
        """
        raise NotImplementedError("Subclasses must implement _calculate_confidence_score")

    @abstractmethod
    def _calculate_flops(self, response: LLMResponse) -> float:
        """Calculate FLOPs for this response."""
        raise NotImplementedError("Subclasses must implement _calculate_flops")

    def generate_with_flops(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate with automatic FLOP calculation."""
        response = self.generate(prompt, **kwargs)
        response.flops_used = self._calculate_flops(response)
        return response

    def get_config(self) -> LLMConfig:
        """Get the configuration object for this LLM instance."""
        return self.config

    def _print_response(self, response: LLMResponse):
        """Print the LLM response to terminal."""
        print(f"\n=== LLM Response ===")
        print(f"Model: {response.model_name}")
        print(f"Text: {response.text}")
        print(f"Finish Reason: {response.finish_reason}")
        print(f"Confidence Score: {response.confidence_score:.4f}")
        print(f"Input Tokens: {response.input_tokens}")
        print(f"Output Tokens: {response.output_tokens}")
        print(f"Total Tokens: {response.total_tokens}")
        print(f"Generation Time: {response.generation_time:.2f}s")
        if response.flops_used is not None:
            print(f"FLOPs Used: {response.flops_used}")
        if response.error:
            print(f"Error: {response.error}")
        print("=" * 50)
