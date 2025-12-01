"""Test LLM Configuration"""

import pytest

from llm import LLMConfig


def test_llm_config_creation():
    """Test basic LLMConfig creation."""
    config = LLMConfig(
        model_name="gpt-4o-mini",
        model_type="api",
        api_key="test-key",
        temperature=0.7,
        max_tokens=4096,
        context_length=128000,
        parameter_count=8000000000
    )

    assert config.model_name == "gpt-4o-mini"
    assert config.model_type == "api"
    assert config.api_key == "test-key"
    assert config.temperature == 0.7
    assert config.max_tokens == 4096
    assert config.context_length == 128000
    assert config.parameter_count == 8000000000
    assert config.is_api is True
    assert config.is_local is False
