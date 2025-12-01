"""Local LLM Implementation Module"""

import time
import logging
from typing import Dict, Tuple, Optional

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import accelerate
    import huggingface_hub
    HF_AVAILABLE = True
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    BitsAndBytesConfig = None
    accelerate = None
    huggingface_hub = None
    HF_AVAILABLE = False

from ..core.base import LLM
from ..core.config import LLMConfig
from ..core.response import LLMResponse

logger = logging.getLogger(__name__)


class LocalLLM(LLM):
    """Local HuggingFace model implementation."""

    def __init__(self, config: LLMConfig):
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace transformers not available. Install with: pip install torch transformers accelerate")

        # Initialize attributes before calling super().__init__() which calls _setup()
        self.tokenizer = None
        self.model = None
        self.device = None

        super().__init__(config)

    def _setup(self):
        """Load model and tokenizer from HuggingFace."""
        try:
            # Determine device
            if self.config.device:
                self.device = torch.device(self.config.device)
            elif torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")

            logger.info(f"Using device: {self.device}")

            # Setup quantization config if specified
            quantization_config = None
            if self.config.quantization:
                if "4bit" in self.config.quantization.lower():
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                elif "8bit" in self.config.quantization.lower():
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_compute_dtype=torch.float16
                    )

            # Load tokenizer
            model_path = self.config.model_path or self.config.model_name

            # Check if model_path is a local directory
            import os
            is_local_path = os.path.isdir(model_path)

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=is_local_path
            )

            # Add pad token if missing (common for some models)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with device specification
            device_map = {"": self.device.type if self.device.type == "cuda" else "cpu"}

            if is_local_path:
                # Load from local directory
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    local_files_only=True,
                    device_map=device_map if self.device.type == "cuda" else None
                )
            else:
                # Load from HuggingFace Hub
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    device_map=device_map if self.device.type == "cuda" else None
                )

            # For CPU, move model after loading
            if self.device.type == "cpu":
                self.model.to(self.device)

            # Set model to evaluation mode
            self.model.eval()

            # Verify the model and tokenizer were loaded
            if self.tokenizer is None or self.model is None:
                raise RuntimeError("Model or tokenizer failed to load")

            logger.info(f"Successfully loaded model: {self.config.model_name}")

        except Exception as e:
            logger.error(f"Failed to load model {self.config.model_name}: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate with local model and automatic confidence interval calculation."""
        start_time = time.time()

        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            # Filter out token_type_ids if present (not used by causal language models)
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
            # Move inputs to the same device as the model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            input_tokens = inputs["input_ids"].shape[1]

            # Prepare generation parameters
            generation_kwargs = {
                "max_new_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "do_sample": self.config.temperature > 0,
                "return_dict_in_generate": True,
                "output_scores": True,  # Enable logits output for confidence calculation
            }

            # Only set pad_token_id if it exists
            if self.tokenizer.pad_token_id is not None:
                generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

            # Override with any additional kwargs
            generation_kwargs.update(kwargs)

            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)

            # Extract generated tokens (excluding input)
            generated_tokens = outputs.sequences[:, input_tokens:]
            generated_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

            # Calculate token counts
            output_tokens = generated_tokens.shape[1]
            total_tokens = input_tokens + output_tokens

            # Extract logits for confidence calculation
            logits = outputs.scores  # List of tensors, one per generated token

            # Calculate confidence score using the paper's formula
            confidence_score = self._calculate_confidence_score(logits)

            # Determine finish reason
            if output_tokens >= self.config.max_tokens:
                finish_reason = "length"
            else:
                finish_reason = "stop"

            # Create response object
            llm_response = LLMResponse(
                text=generated_text,
                finish_reason=finish_reason,
                confidence_score=confidence_score,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                model_name=self.config.model_name,
                generation_time=time.time() - start_time,
                logits=logits
            )

            # Print response to terminal
            self._print_response(llm_response)

            return llm_response

        except Exception as e:
            logger.error(f"Error in local generation: {e}")
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

    def _calculate_confidence_score(self, logits) -> float:
        """
        Calculate confidence score using the paper's formula:
        c₁ = (1 / |r₁|) * Σ(i=1 to |r₁|) P(ti)

        Where P(ti) is the softmax probability of the selected token at step i,
        and |r₁| is the number of tokens in the generated response.
        """
        try:
            if not logits or len(logits) == 0:
                return 0.5

            # Convert logits to probabilities and get selected token probabilities
            selected_probs = []
            for token_logits in logits:
                # Apply softmax to get probabilities
                probs = torch.softmax(token_logits, dim=-1)
                # Get the probability of the selected token (argmax)
                selected_prob = torch.max(probs).item()
                selected_probs.append(selected_prob)

            if not selected_probs:
                return 0.5

            # Calculate confidence score: average of selected token probabilities
            # c₁ = (1 / |r₁|) * Σ(i=1 to |r₁|) P(ti)
            confidence_score = sum(selected_probs) / len(selected_probs)

            return confidence_score

        except Exception as e:
            logger.warning(f"Failed to calculate confidence score from logits: {e}")
            return 0.5

    def _calculate_flops(self, response: LLMResponse) -> float:
        """Calculate FLOPs directly from model operations."""
        try:
            # Get parameter count from config
            param_count = self.config.parameter_count

            if param_count is None:
                logger.warning(f"No parameter count specified for model {self.config.model_name}")
                return 0.0

            # For transformer models, a rough estimate is:
            # FLOPs = 2 * num_params * num_tokens_generated
            # The factor of 2 accounts for forward and backward operations in attention
            # This is a simplification; actual FLOP count would require detailed model analysis
            flops = 2 * param_count * response.total_tokens

            return flops

        except Exception as e:
            logger.warning(f"Failed to calculate FLOPs: {e}")
            return 0.0
