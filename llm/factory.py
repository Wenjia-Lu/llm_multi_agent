"""LLM Factory Module"""

import json
from typing import Any, Dict

from .core.config import LLMConfig


class LLMFactory:
    """Factory for creating LLM instances from configuration files."""

    @staticmethod
    def from_config_file(config_file_path: str):
        """
        Create LLM instance directly from a JSON configuration file.

        Args:
            config_file_path: Path to JSON config file

        Returns:
            LLM instance (APILLM or LocalLLM)

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid or model type unknown
        """
        try:
            with open(config_file_path, 'r') as f:
                config_dict = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {config_file_path}: {e}")

        # Validate required fields are present
        required_fields = ['model_name', 'model_type', 'temperature', 'max_tokens', 'context_length', 'parameter_count']
        missing_fields = [field for field in required_fields if field not in config_dict]
        if missing_fields:
            raise ValueError(f"Missing required fields in config {config_file_path}: {missing_fields}")

        # Create config and validate model type
        try:
            config = LLMConfig(**config_dict)
        except TypeError as e:
            raise ValueError(f"Invalid config parameters in {config_file_path}: {e}")

        # Validate model type specific requirements
        if config.is_api and not config.api_key:
            raise ValueError(f"API models require 'api_key' in config {config_file_path}")
        # Local models can use model_path (for local files) or model_name (for HuggingFace download)
        # No validation needed here - the LocalLLM will handle it

        # Create and return LLM instance
        return LLMFactory.create(config)

    @staticmethod
    def create(config: LLMConfig):
        """Create LLM instance from config."""
        if config.is_api:
            from .implementations.api_llm import APILLM
            return APILLM(config)
        elif config.is_local:
            from .implementations.local_llm import LocalLLM
            return LocalLLM(config)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]):
        """Create LLM from dictionary config."""
        config = LLMConfig(**config_dict)
        return LLMFactory.create(config)

    @staticmethod
    def from_json(config_path: str):
        """Create LLM from JSON config file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return LLMFactory.from_dict(config_dict)
