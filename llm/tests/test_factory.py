"""Test LLM Factory"""

import pytest
from unittest.mock import patch

from llm import LLMConfig, LLMFactory


def test_factory_create_api():
    """Test factory creation of API LLM."""
    config = LLMConfig(
        model_name="gpt-4o-mini",
        model_type="api",
        api_key="test-key",
        temperature=0.7,
        max_tokens=2048,
        context_length=4096,
        parameter_count=8_000_000_000
    )

    # Import the module to set up the conditional imports
    import llm.implementations.api_llm as api_llm_module

    with patch.object(api_llm_module, 'OPENAI_AVAILABLE', True):
        with patch.object(api_llm_module, 'OpenAI'):
            with patch.object(api_llm_module, 'tiktoken'):
                llm = LLMFactory.create(config)

                from llm.implementations.api_llm import APILLM
                assert isinstance(llm, APILLM)
                assert llm.get_config() == config


def test_factory_create_local():
    """Test factory creation of local LLM."""
    config = LLMConfig(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        model_type="local",
        model_path="/path/to/model",
        temperature=0.7,
        max_tokens=4096,
        context_length=4096,
        parameter_count=7000000000
    )

    # This will fail since LocalLLM is not implemented yet
    with pytest.raises(NotImplementedError):
        LLMFactory.create(config)


def test_factory_invalid_type():
    """Test factory with invalid model type."""
    config = LLMConfig(
        model_name="test-model",
        model_type="invalid",
        temperature=0.7,
        max_tokens=4096,
        context_length=4096,
        parameter_count=1000000000
    )

    with pytest.raises(ValueError, match="Unknown model type: invalid"):
        LLMFactory.create(config)


def test_factory_from_config_file(tmp_path):
    """Test factory creation from config file."""
    import json

    config_dict = {
        "model_name": "gpt-4o-mini",
        "model_type": "api",
        "api_key": "test-key",
        "temperature": 0.7,
        "max_tokens": 4096,
        "context_length": 128000,
        "parameter_count": 8_000_000_000
    }

    # Create temporary config file
    config_file = tmp_path / "gpt4o-mini.json"
    with open(config_file, 'w') as f:
        json.dump(config_dict, f)

    # Import the module to set up the conditional imports
    import llm.implementations.api_llm as api_llm_module

    with patch.object(api_llm_module, 'OPENAI_AVAILABLE', True):
        with patch.object(api_llm_module, 'OpenAI'):
            with patch.object(api_llm_module, 'tiktoken'):
                llm = LLMFactory.from_config_file(str(config_file))

                from llm.implementations.api_llm import APILLM
                assert isinstance(llm, APILLM)
                assert llm.get_config().model_name == "gpt-4o-mini"


def test_factory_from_config_file_missing_required(tmp_path):
    """Test factory creation fails with missing required fields."""
    import json

    # Config missing required fields
    config_dict = {
        "model_name": "gpt-4o-mini",
        "model_type": "api"
        # Missing temperature, max_tokens, etc.
    }

    config_file = tmp_path / "incomplete.json"
    with open(config_file, 'w') as f:
        json.dump(config_dict, f)

    with pytest.raises(ValueError, match="Missing required fields"):
        LLMFactory.from_config_file(str(config_file))


def test_factory_from_config_file_invalid_json(tmp_path):
    """Test factory creation fails with invalid JSON."""
    config_file = tmp_path / "invalid.json"
    with open(config_file, 'w') as f:
        f.write("invalid json content")

    with pytest.raises(ValueError, match="Invalid JSON"):
        LLMFactory.from_config_file(str(config_file))


def test_factory_from_config_file_not_found():
    """Test factory creation fails with non-existent file."""
    with pytest.raises(FileNotFoundError):
        LLMFactory.from_config_file("non_existent_file.json")


def test_llm_get_config(tmp_path):
    """Test that LLM instances properly expose their config via get_config()."""
    import json

    # Create test config files
    api_config = {
        "model_name": "gpt-4o-mini",
        "model_type": "api",
        "api_key": "test-key",
        "temperature": 0.7,
        "max_tokens": 4096,
        "context_length": 128000,
        "parameter_count": 8000000000
    }

    local_config = {
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "model_type": "local",
        "model_path": "/path/to/model",
        "temperature": 0.8,
        "max_tokens": 2048,
        "context_length": 4096,
        "parameter_count": 7000000000
    }

    # Create temporary config files
    api_file = tmp_path / "api_config.json"
    local_file = tmp_path / "local_config.json"

    with open(api_file, 'w') as f:
        json.dump(api_config, f)

    with open(local_file, 'w') as f:
        json.dump(local_config, f)

    # Test API LLM config access
    # Import the module to set up the conditional imports
    import llm.implementations.api_llm as api_llm_module

    with patch.object(api_llm_module, 'OPENAI_AVAILABLE', True):
        with patch.object(api_llm_module, 'OpenAI'):
            with patch.object(api_llm_module, 'tiktoken'):
                api_llm = LLMFactory.from_config_file(str(api_file))
                config = api_llm.get_config()

                print("\n📋 API LLM Config:")
                print(f"  Model Name: {config.model_name}")
                print(f"  Model Type: {config.model_type}")
                print(f"  Temperature: {config.temperature}")
                print(f"  Max Tokens: {config.max_tokens}")
                print(f"  Context Length: {config.context_length}")
                print(f"  Parameter Count: {config.parameter_count}")
                print(f"  API Key: {'*' * len(config.api_key) if config.api_key else None}")

                # Verify config values
                assert config.model_name == "gpt-4o-mini"
                assert config.model_type == "api"
                assert config.temperature == 0.7
                assert config.max_tokens == 4096
                assert config.context_length == 128000
                assert config.parameter_count == 8000000000
                assert config.api_key == "test-key"

    # Test Local LLM config access (will fail to create due to missing LocalLLM, but config loading should work)
    try:
        local_llm = LLMFactory.from_config_file(str(local_file))
        config = local_llm.get_config()

        print("\n📋 Local LLM Config:")
        print(f"  Model Name: {config.model_name}")
        print(f"  Model Type: {config.model_type}")
        print(f"  Temperature: {config.temperature}")
        print(f"  Max Tokens: {config.max_tokens}")
        print(f"  Context Length: {config.context_length}")
        print(f"  Parameter Count: {config.parameter_count}")
        print(f"  Model Path: {config.model_path}")

        # Verify config values
        assert config.model_name == "meta-llama/Llama-2-7b-chat-hf"
        assert config.model_type == "local"
        assert config.temperature == 0.8
        assert config.max_tokens == 2048
        assert config.context_length == 4096
        assert config.parameter_count == 7000000000
        assert config.model_path == "/path/to/model"

    except NotImplementedError:
        # LocalLLM not implemented yet, but config loading should have worked
        print("\n⚠️  Local LLM not implemented yet, but config validation passed")

    print("\n✅ Config access test completed successfully!")
