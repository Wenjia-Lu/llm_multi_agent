"""API LLM Implementation Module"""

import time
from typing import Dict, Tuple, Any
import logging

try:
    from openai import OpenAI
    import tiktoken
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    tiktoken = None
    OPENAI_AVAILABLE = False

from ..core.base import LLM
from ..core.config import LLMConfig
from ..core.response import LLMResponse

logger = logging.getLogger(__name__)


class APILLM(LLM):
    """API-based LLM implementation using OpenAI."""

    def __init__(self, config: LLMConfig):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available. Install with: pip install openai tiktoken")

        self.client = None
        self.tokenizer = None
        super().__init__(config)

    def _setup(self):
        """Setup OpenAI client and tokenizer."""
        if not self.config.api_key:
            raise ValueError("API key required for API models")

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_url if self.config.api_url else None
        )

        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.config.model_name)
        except KeyError:
            # Fallback to cl100k_base (GPT-3.5/4 tokenizer)
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        logger.info(f"Initialized APILLM with model: {self.config.model_name}")

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate via OpenAI API with automatic confidence interval calculation."""
        start_time = time.time()

        try:
            # Prepare API call parameters
            api_kwargs = {
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "logprobs": True,  # Enable log probabilities for confidence calculation
                "top_logprobs": 5,  # Get top 5 probabilities for better confidence estimation
            }

            # Override with any additional kwargs
            api_kwargs.update(kwargs)

            # Make API call with retry logic
            response = self._make_api_call_with_retry(api_kwargs)

            # Extract response data
            choice = response.choices[0]
            generated_text = choice.message.content
            finish_reason = choice.finish_reason

            # Calculate token usage
            input_tokens = len(self.tokenizer.encode(prompt))
            output_tokens = len(self.tokenizer.encode(generated_text))
            total_tokens = input_tokens + output_tokens

            # Calculate confidence score using the paper's formula
            confidence_score = self._calculate_confidence_score(choice)

            # Create response object
            llm_response = LLMResponse(
                text=generated_text,
                finish_reason=finish_reason,
                confidence_score=confidence_score,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                model_name=self.config.model_name,
                generation_time=time.time() - start_time
            )

            # Print response to terminal
            self._print_response(llm_response)

            return llm_response

        except Exception as e:
            logger.error(f"Error in API generation: {e}")
            error_response = LLMResponse(
                text="",
                finish_reason="error",
                confidence_score=0.0,
                error=str(e),
                model_name=self.config.model_name,
                generation_time=time.time() - start_time
            )
            # Print error response to terminal
            self._print_response(error_response)
            return error_response

    def _make_api_call_with_retry(self, api_kwargs: Dict[str, Any], max_retries: int = 3):
        """Make API call with exponential backoff retry logic."""
        import time

        for attempt in range(max_retries):
            try:
                return self.client.chat.completions.create(**api_kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e

                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

        raise RuntimeError("Max retries exceeded")

    def _calculate_confidence_score(self, choice) -> float:
        """
        Calculate confidence score using the paper's formula:
        c₁ = (1 / |r₁|) * Σ(i=1 to |r₁|) P(ti)

        For API models, we use OpenAI's logprobs which are already probabilities.
        """
        try:
            logprobs = choice.logprobs
            if not logprobs or not hasattr(logprobs, 'content'):
                return 0.5

            # Extract token probabilities from OpenAI logprobs
            token_probs = []
            for token_logprob in logprobs.content:
                if token_logprob.logprob is not None:
                    # Convert from log2 probability to linear probability
                    prob = 2 ** token_logprob.logprob
                    token_probs.append(prob)

            if not token_probs:
                return 0.5

            # Calculate confidence score: average of token probabilities
            # c₁ = (1 / |r₁|) * Σ(i=1 to |r₁|) P(ti)
            confidence_score = sum(token_probs) / len(token_probs)

            return confidence_score

        except Exception as e:
            logger.warning(f"Failed to calculate confidence score: {e}")
            return 0.5

    def _calculate_flops(self, response: LLMResponse) -> float:
        """Estimate FLOPs using parameter count: 2 × params × tokens."""
        try:
            # Get parameter count from config
            param_count = self.config.parameter_count

            if param_count is None:
                logger.warning(f"No parameter count specified for model {self.config.model_name}")
                return 0.0

            # Estimate FLOPs: 2 * params * total_tokens
            # The factor of 2 accounts for forward and backward passes in attention mechanisms
            flops = 2 * param_count * response.total_tokens

            return flops

        except Exception as e:
            logger.warning(f"Failed to calculate FLOPs: {e}")
            return 0.0

