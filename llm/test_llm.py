#!/usr/bin/env python3
"""
Demo script for LocalLLM functionality.
Shows how to generate responses, FLOPs calculation, and confidence intervals.

Requirements:
    pip install torch transformers accelerate

Usage:
    python test_local_llm_demo.py
    # or make executable and run:
    chmod +x test_local_llm_demo.py && ./test_local_llm_demo.py

This script will:
1. Load a small DialoGPT model from HuggingFace
2. Generate a response to a prompt
3. Display FLOPs used and confidence intervals
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import LLMFactory, LLMConfig

def main():
    print("🤖 LocalLLM Demo")
    print("=" * 50)

    # Check if dependencies are available
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("✅ Dependencies found!")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install required packages:")
        print("  pip install torch transformers accelerate")
        print()
        print("Then run this script again.")
        return

    # Try to use the config file, fallback to hardcoded config
    # config_file = "llm/configs/lfm1200v2.json"
    # config_file = "llm/configs/gpt2.json"
    config_file = "llm/configs/mistral-7b-instruct.json"
    try:
        print(f"📄 Loading config from: {config_file}")
        llm = LLMFactory.from_config_file(config_file)
        print("✅ Config loaded successfully!")
        print()
    except Exception as e:
        print(f"⚠️  Could not load config file: {e}")
        print("🔧 Falling back to demo config...")

        # Configure a small model for demo (DistilGPT-2)
        config = LLMConfig(
            model_name="distilgpt2",
            model_type="local",
            model_path=None,  # Will download from HuggingFace
            device="cpu",  # Use CPU for demo
            quantization=None,  # No quantization for simplicity
            temperature=0.7,
            max_tokens=30,  # Keep response short for demo
            context_length=1024,
            parameter_count=82000000  # ~82M parameters
        )
        llm = LLMFactory.create(config)
        print("✅ Demo config loaded successfully!")
        print()

    config = llm.get_config()

    print(f"📋 Configuration:")
    print(f"   Model: {config.model_name}")
    print(f"   Device: {config.device}")
    print(f"   Temperature: {config.temperature}")
    print(f"   Max Tokens: {config.max_tokens}")
    print(f"   Parameters: {config.parameter_count:,}")
    print()

    try:
        print("🔄 Model already loaded, proceeding to generation...")
        print()

        # Test prompt
        prompt = "What is 2 plus 2, only answer this question and add no other information?"
        print(f"💬 Prompt: {prompt}")
        print()

        print("🤔 Generating response...")
        response = llm.generate_with_flops(prompt)
        print("✅ Response generated!")
        print()

        # Display results
        print("📝 Results:")
        print("-" * 30)
        print(f"Response: {response.text}")
        print(f"Finish Reason: {response.finish_reason}")
        print(f"Input Tokens: {response.input_tokens}")
        print(f"Output Tokens: {response.output_tokens}")
        print(f"Total Tokens: {response.total_tokens}")
        print(f"Generation Time: {response.generation_time:.2f}s")
        print()

        # Display FLOPs
        print("⚡ Performance:")
        print("-" * 30)
        if response.flops_used:
            flops_billions = response.flops_used / 1e9
            print(f"FLOPs Used: {flops_billions:.2f} billion")
            print(f"Tokens/sec: {response.total_tokens / response.generation_time:.1f}")
        else:
            print("FLOPs calculation not available")
        print()

        # Display confidence score
        print("📊 Confidence Score:")
        print("-" * 30)
        print(f"Score: {response.confidence_score:.4f}")
        print(f"Threshold Decision: {'Accept' if response.confidence_score > 0.8 else 'Debate'} (θ=0.8)")
        print()

        # Show raw response details
        print("🔍 Raw Response Details:")
        print("-" * 30)
        print(f"Model Name: {response.model_name}")
        print(f"Has Logits: {response.logits is not None}")
        if response.error:
            print(f"Error: {response.error}")

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please install required dependencies:")
        print("  pip install torch transformers accelerate")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
