"""Test API LLM Implementation"""

import pytest
from unittest.mock import Mock, patch

from llm import LLMConfig, APILLM


@pytest.fixture
def api_config():
    """Create a test API config."""
    return LLMConfig(
        model_name="gpt-4o-mini",
        model_type="api",
        api_key="test-key-123",
        temperature=0.7,
        max_tokens=100,
        context_length=4096,
        parameter_count=8_000_000_000
    )


def test_api_llm_get_config(api_config):
    """Test that APILLM exposes its config via get_config()."""
    # Import the module to set up the conditional imports
    import llm.implementations.api_llm as api_llm_module

    with patch.object(api_llm_module, 'OPENAI_AVAILABLE', True):
        with patch.object(api_llm_module, 'OpenAI'):
            with patch.object(api_llm_module, 'tiktoken'):
                # Create APILLM instance
                api_llm = APILLM(api_config)

                # Test get_config method
                config = api_llm.get_config()
                assert config == api_config
                assert config.model_name == "gpt-4o-mini"
                assert config.model_type == "api"
                assert config.api_key == "test-key-123"


