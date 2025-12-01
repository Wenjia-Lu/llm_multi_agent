# LocalLLM vLLM Migration Guide

This guide explains how to migrate from the old transformers-based LocalLLM to the new vLLM-based LocalLLM.

## Overview

The LocalLLM class has been completely rewritten to use vLLM instead of HuggingFace transformers directly. This provides better performance, GPU utilization, and maintains the same interface.

## Key Changes

### 1. Dependencies
- **Removed**: `torch`, `transformers`, `accelerate`, `huggingface_hub`
- **Added**: `vllm`, `tiktoken` (for token counting)

### 2. Configuration
The LLMConfig now supports additional vLLM-specific parameters:

```python
config = LLMConfig(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    model_type="local",  # Still "local"
    temperature=0.7,
    max_tokens=512,
    context_length=8192,
    parameter_count=8_030_000_000,

# New vLLM-specific options:
gpu_memory_utilization=0.9,  # Use 90% of GPU memory
tensor_parallel_size=1,      # Multi-GPU support
dtype="auto",                # Precision: auto/float16/float32
quantization="awq",          # Quantization: awq/gptq/squeezellm/4bit/8bit
)
```

### 3. Behavior Changes

#### Automatic vLLM Server Management
- LocalLLM now automatically starts/stops vLLM servers
- No need to manually run `python -m vllm.entrypoints.openai.api_server`
- Servers run on automatically allocated ports
- Automatic cleanup when LocalLLM objects are destroyed

#### Performance Improvements
- Better GPU utilization through vLLM optimizations
- Continuous batching and paged attention
- CUDA graphs for faster inference
- Support for concurrent requests

## Migration Steps

### Step 1: Update Dependencies
```bash
pip install vllm>=0.4.0 tiktoken>=0.5.0
# Remove old dependencies if no longer needed:
# pip uninstall torch transformers accelerate
```

### Step 2: Update Extension Scripts

#### Current Code (APILLM approach):
```python
from llm.implementations.api_llm import APILLM
from llm.core.config import LLMConfig

config = LLMConfig(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    model_type="api",
    temperature=0.7,
    max_tokens=512,
    context_length=8192,
    parameter_count=8_030_000_000,
    api_url="http://localhost:8000/v1",
    api_key="EMPTY"
)
mas_model = APILLM(config=config)
```

#### New Code (LocalLLM with vLLM):
```python
from llm.implementations.local_llm import LocalLLM  # Changed import
from llm.core.config import LLMConfig

config = LLMConfig(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    model_type="local",  # Changed from "api"
    temperature=0.7,
    max_tokens=512,
    context_length=8192,
    parameter_count=8_030_000_000,
    # Removed: api_url, api_key
# Optional: Add vLLM-specific params
gpu_memory_utilization=0.9,
quantization="awq",  # Enable quantization for memory efficiency
)
mas_model = LocalLLM(config=config)  # Changed class
```

### Step 3: Leverage Quantization for Memory Efficiency

The new LocalLLM supports quantization to reduce memory usage:

```python
config = LLMConfig(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    model_type="local",
    # ... other params ...
    quantization="awq",  # Reduces memory by ~75%
    gpu_memory_utilization=0.9,
)
```

#### Supported Quantization Methods:
- `"awq"` - Activation-aware Weight Quantization (best quality)
- `"gptq"` - GPT Quantization (good quality, fast)
- `"squeezellm"` - SqueezeLLM quantization
- `"4bit"` - 4-bit quantization via BitsAndBytes
- `"8bit"` - 8-bit quantization via BitsAndBytes

**Memory Savings:**
- 8B model without quantization: ~16GB (FP16)
- 8B model with AWQ: ~4GB
- 8B model with 4-bit: ~2GB

### Step 4: Remove Manual vLLM Server Management

#### Old Approach (Manual):
```bash
# Terminal 1: Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000

# Terminal 2: Run extension script
python extension/biography/gen_OS.py
```

#### New Approach (Automatic):
```bash
# Single command - LocalLLM handles vLLM server automatically
python extension/biography/gen_OS.py
```

## Multiple Models

### Old Approach:
```bash
# Terminal 1
python -m vllm.entrypoints.openai.api_server --model llama-model --port 8000

# Terminal 2
python -m vllm.entrypoints.openai.api_server --model mistral-model --port 8001

# Scripts use different API_URL environment variables
API_URL=http://localhost:8000/v1 python script1.py
API_URL=http://localhost:8001/v1 python script2.py
```

### New Approach:
```python
# Each LocalLLM instance manages its own vLLM server automatically
llama_config = LLMConfig(model_name="llama-model", model_type="local", ...)
mistral_config = LLMConfig(model_name="mistral-model", model_type="local", ...)

llama_llm = LocalLLM(llama_config)    # Auto-starts vLLM on random port
mistral_llm = LocalLLM(mistral_config) # Auto-starts vLLM on different random port
```

## Error Handling

### Common Issues:

1. **vLLM not installed**: `pip install vllm`
2. **Model not found**: Ensure HuggingFace login: `huggingface-cli login`
3. **CUDA not available**: vLLM will fall back to CPU (slow)
4. **Port conflicts**: LocalLLM automatically finds available ports

### Debugging:
- Check vLLM server logs in terminal where LocalLLM is running
- Use `ps aux | grep vllm` to see running vLLM processes
- LocalLLM logs server startup/shutdown events

## Performance Comparison

| Metric | Old (APILLM) | New (LocalLLM + vLLM) |
|--------|-------------|------------------------|
| Startup Time | Fast (pre-started server) | Medium (auto-start server) |
| Memory Usage | Low (shared server) | Low (optimized vLLM) |
| Inference Speed | Fast | **Faster** (vLLM optimizations) |
| GPU Utilization | Good | **Better** (continuous batching) |
| Ease of Use | Moderate (manage servers) | **Easy** (automatic) |
| Scalability | Good (multiple servers) | Good (multiple instances) |

## Apple Silicon Optimization Example

For running multiple models on Apple Silicon with limited RAM:

```python
from llm.core.config import LLMConfig
from llm.implementations.local_llm import LocalLLM

# Configuration optimized for Apple Silicon
configs = [
    LLMConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        model_type="local",
        temperature=0.7,
        max_tokens=512,
        context_length=4096,
        parameter_count=8_000_000_000,
        device="cpu",  # Force CPU mode for Apple Silicon
        quantization="awq",  # Reduce memory from ~32GB to ~8GB
        gpu_memory_utilization=0.8,
    ),
    LLMConfig(
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        model_type="local",
        temperature=0.7,
        max_tokens=512,
        context_length=4096,
        parameter_count=7_000_000_000,
        device="cpu",
        quantization="awq",  # Reduce memory from ~28GB to ~7GB
        gpu_memory_utilization=0.8,
    ),
    LLMConfig(
        model_name="microsoft/WizardLM-7B-V1.0",
        model_type="local",
        temperature=0.7,
        max_tokens=512,
        context_length=4096,
        parameter_count=7_000_000_000,
        device="cpu",
        quantization="awq",  # Reduce memory from ~28GB to ~7GB
        gpu_memory_utilization=0.8,
    )
]

# Initialize all models (total memory: ~22GB instead of ~88GB)
models = [LocalLLM(config) for config in configs]
```

**Memory Comparison:**
- Without quantization: 3 × 8B models = ~88GB (impossible on 48GB RAM)
- With AWQ quantization: 3 × 8B models = ~22GB (feasible on 48GB RAM)

## Benefits of Migration

1. **Simplified Deployment**: No manual server management
2. **Better Performance**: vLLM optimizations
3. **Resource Efficiency**: Automatic memory management with quantization
4. **Concurrent Requests**: Handle multiple generations simultaneously
5. **Unified Interface**: Same LocalLLM class for all use cases
6. **Apple Silicon Support**: Quantization enables multi-model inference on limited RAM

## Testing

Run the test script to verify functionality:
```bash
python test_local_llm_vllm.py
```

This will test the new LocalLLM with a small model to ensure everything works correctly.
