"""Test Local LLM Implementation"""

import pytest

from llm import LLMConfig, LLMFactory


def test_factory_identifies_local_llm():
    """Test that factory correctly identifies local LLM configurations."""
    config = LLMConfig(
        model_name="distilgpt2",
        model_type="local",
        model_path=None,
        temperature=0.7,
        max_tokens=30,
        context_length=1024,
        parameter_count=82000000
    )

    # Factory should attempt to create LocalLLM
    # This will fail if dependencies are not installed
    try:
        llm = LLMFactory.create(config)
        # If we get here, the basic structure works
        assert llm is not None
        assert llm.config.model_type == "local"
        assert llm.config.model_name == "microsoft/DialoGPT-small"
    except ImportError as e:
        # Expected if dependencies aren't installed
        assert "transformers" in str(e) or "torch" in str(e)
