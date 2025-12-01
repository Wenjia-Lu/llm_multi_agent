# LLM Model Configurations

This directory contains JSON configuration files for different LLM models. Each file defines the complete configuration needed to instantiate an LLM instance.

## Configuration Format

Each configuration file should contain all required fields for the LLM to function properly. There are no defaults - all necessary parameters must be explicitly specified.

### Required Fields (for all models)

- `model_name`: String identifier for the model
- `model_type`: Either "api" or "local"
- `temperature`: Float between 0.0 and 1.0 for generation randomness
- `max_tokens`: Integer maximum tokens to generate
- `context_length`: Integer maximum context length supported
- `parameter_count`: Integer number of parameters (used for FLOP calculations)

### API Model Fields

- `api_key`: Your API key for the service
- `api_url`: (Optional) Custom API endpoint URL

### Local Model Fields

- `model_path`: Path to the local model files
- `device`: (Optional) Device to run on ("cuda:0", "cpu", etc.)
- `quantization`: (Optional) Quantization scheme ("4bit", "8bit", etc.)

## Usage

```python
from llm import LLMFactory

# Create LLM from config file
llm = LLMFactory.from_config_file("llm/configs/gpt4o-mini.json")

# Use the LLM
response = llm.generate("Hello, world!")
```

## Example Configurations

### GPT-4o Mini (API)
```json
{
    "model_name": "gpt-4o-mini",
    "model_type": "api",
    "api_key": "your-api-key-here",
    "temperature": 0.7,
    "max_tokens": 4096,
    "context_length": 128000,
    "parameter_count": 8000000000
}
```

### Llama 2 7B (Local)
```json
{
    "model_name": "meta-llama/Llama-2-7b-chat-hf",
    "model_type": "local",
    "model_path": "/path/to/llama-2-7b-chat-hf",
    "device": "cuda:0",
    "quantization": "4bit",
    "temperature": 0.7,
    "max_tokens": 4096,
    "context_length": 4096,
    "parameter_count": 7000000000
}
```

## Validation

The factory will validate that:
- All required fields are present
- Model type is either "api" or "local"
- API models have an `api_key`
- Local models have a `model_path`
- JSON is valid and contains proper data types
