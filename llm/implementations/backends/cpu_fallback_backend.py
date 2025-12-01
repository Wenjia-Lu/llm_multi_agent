"""CPU Fallback Backend Implementation using basic transformers"""

import time
import logging
import os
from pathlib import Path

from ...core.config import LLMConfig
from ...core.response import LLMResponse
from .model_path_manager import ModelPathManager

logger = logging.getLogger(__name__)


class CPUFallbackBackend:
    """CPU fallback using basic transformers."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self._load_model()

    def _get_models_dir(self) -> Path:
        """Get the local models directory."""
        return ModelPathManager._get_models_dir()

    def _load_model(self):
        """Load basic transformers model."""
        # Check if dependencies are available without importing them
        try:
            import subprocess
            import sys
            result = subprocess.run(
                [sys.executable, "-c", "import torch; import transformers"],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                raise ImportError("Dependencies not available")
        except Exception:
            logger.warning("PyTorch/transformers not available. Using mock backend.")
            self._use_mock_backend = True
            return

        # Dependencies are available, proceed with loading
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            # For CPU fallback, we can use transformers' built-in caching
            # but we'll ensure the model name is properly handled
            if self.config.model_path:
                model_path = self.config.model_path
            else:
                model_path = self.config.model_name

            # Set cache directory to our local models folder for transformers
            models_dir = self._get_models_dir()
            cache_dir = models_dir / "transformers_cache"
            cache_dir.mkdir(exist_ok=True)

            # Set environment variable for transformers cache
            os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
            os.environ['HF_HOME'] = str(cache_dir)

            # Handle quantization for transformers
            quantization_config = None
            if self.config.quantization:
                if self.config.quantization in ["4bit", "8bit"]:
                    try:
                        from transformers import BitsAndBytesConfig
                        if self.config.quantization == "4bit":
                            quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4"
                            )
                        elif self.config.quantization == "8bit":
                            quantization_config = BitsAndBytesConfig(
                                load_in_8bit=True,
                                bnb_8bit_compute_dtype=torch.float16
                            )
                    except ImportError:
                        logger.warning(f"BitsAndBytes not available for {self.config.quantization} quantization")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir=str(cache_dir),
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                cache_dir=str(cache_dir),
                trust_remote_code=True,
                quantization_config=quantization_config
            )

            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Move to CPU
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Loaded CPU fallback model: {model_path} (cached in {cache_dir})")

        except Exception as e:
            logger.error(f"Failed to load CPU fallback model: {e}")
            raise

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate with basic transformers or mock backend."""
        start_time = time.time()

        # Check if we're using mock backend
        if hasattr(self, '_use_mock_backend') and self._use_mock_backend:
            return self._generate_mock(prompt, start_time)

        try:
            import torch

            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_tokens = inputs["input_ids"].shape[1]

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=self.config.temperature > 0,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **kwargs
                )

            # Extract generated tokens
            generated_tokens = outputs.sequences[:, input_tokens:]
            response_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

            # Token counts
            output_tokens = generated_tokens.shape[1]
            total_tokens = input_tokens + output_tokens

            # Calculate confidence
            logits = outputs.scores
            if logits:
                selected_probs = []
                for token_logits in logits:
                    probs = torch.softmax(token_logits, dim=-1)
                    selected_prob = torch.max(probs).item()
                    selected_probs.append(selected_prob)
                confidence_score = sum(selected_probs) / len(selected_probs)
            else:
                confidence_score = 0.8

            response = LLMResponse(
                text=response_text,
                finish_reason="stop",
                confidence_score=confidence_score,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                model_name=self.config.model_name,
                generation_time=time.time() - start_time,
            )

            return response

        except Exception as e:
            error_response = LLMResponse(
                text="",
                finish_reason="error",
                confidence_score=0.0,
                error=str(e),
                model_name=self.config.model_name,
                generation_time=time.time() - start_time
            )
            return error_response

    def _generate_mock(self, prompt: str, start_time: float) -> LLMResponse:
        """Generate a mock response when ML dependencies are not available."""
        # Create a simple mock response based on the prompt
        mock_responses = [
            "I'm sorry, but I cannot generate responses right now because the required machine learning dependencies are not available in this environment.",
            "This is a mock response. The system detected that PyTorch and transformers are not properly installed.",
            "Mock backend active: Original prompt was: " + prompt[:50] + "..." if len(prompt) > 50 else prompt,
        ]

        import random
        response_text = random.choice(mock_responses)

        return LLMResponse(
            text=response_text,
            finish_reason="stop",
            confidence_score=0.5,
            input_tokens=len(prompt.split()),
            output_tokens=len(response_text.split()),
            total_tokens=len(prompt.split()) + len(response_text.split()),
            model_name=self.config.model_name,
            generation_time=time.time() - start_time,
        )
