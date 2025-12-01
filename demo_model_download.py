#!/usr/bin/env python3
"""
Demo script showing the new local model downloading functionality.
This demonstrates how models are automatically downloaded and cached locally.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def demo_models_directory():
    """Demo the models directory structure."""
    print("📁 Models Directory Structure")
    print("=" * 35)

    models_dir = Path("models")
    print(f"Models directory: {models_dir.absolute()}")

    if models_dir.exists():
        print("✅ Models directory exists")
        print("Contents:")

        for item in sorted(models_dir.iterdir()):
            if item.is_dir():
                files = list(item.iterdir())
                file_count = len([f for f in files if f.is_file() or f.is_dir()])
                print(f"  📂 {item.name}/ ({file_count} items)")
            else:
                size_mb = round(item.stat().st_size / (1024 * 1024), 2)
                print(f"  📄 {item.name} ({size_mb} MB)")
    else:
        print("❌ Models directory not found")

    print()

def demo_model_management():
    """Demo model management functionality."""
    print("🔧 Model Management Demo")
    print("=" * 27)

    try:
        from llm.core.config import LLMConfig

        # Create a config (we won't actually instantiate to avoid PyTorch issues)
        config = LLMConfig(
            model_name="microsoft/DialoGPT-small",
            model_type="local",
            temperature=0.7,
            max_tokens=50,
            context_length=512,
            parameter_count=117000000
        )

        print("✅ Created LLM config")
        print(f"   Model: {config.model_name}")
        print(f"   Type: {config.model_type}")

        # Simulate the OptimizedLocalLLM functionality
        from llm.implementations.optimized_local_llm import OptimizedLocalLLM

        # Create instance without calling __init__ to avoid backend detection
        llm = OptimizedLocalLLM.__new__(OptimizedLocalLLM)
        llm.config = config

        # Test model path generation
        print("\n🔍 Model Path Generation:")
        backends = ["mlx", "llama_cpp", "onnxruntime", "cpu_fallback"]

        for backend in backends:
            try:
                model_path = llm._get_model_path(config.model_name, backend)
                exists = model_path.exists()
                status = "✅ Exists" if exists else "📥 Will download"
                print(f"   {backend}: {status}")
                print(f"      Path: {model_path}")
            except Exception as e:
                print(f"   {backend}: ❌ Error - {e}")

        # Test model listing
        print("\n📋 Downloaded Models:")
        downloaded = llm.list_downloaded_models()
        if downloaded:
            for backend, models in downloaded.items():
                print(f"   {backend}: {models}")
        else:
            print("   ℹ️  No models downloaded yet")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

    print()

def demo_usage_examples():
    """Show usage examples."""
    print("📖 Usage Examples")
    print("=" * 19)

    print("1️⃣ Basic Usage (Automatic Download):")
    print("""
from llm.core.config import LLMConfig
from llm.implementations.local_llm import LocalLLM

config = LLMConfig(
    model_name="microsoft/DialoGPT-small",
    model_type="local",
    temperature=0.7,
    max_tokens=100,
    context_length=512,
    parameter_count=117000000
)

# First run will download the model automatically
llm = LocalLLM(config)
response = llm.generate("Hello!")
print(response.text)
""")

    print("2️⃣ Manual Model Path:")
    print("""
config = LLMConfig(
    model_name="my-model",
    model_type="local",
    model_path="./models/llama_cpp/my-model.gguf",  # Use existing GGUF file
    temperature=0.7,
    max_tokens=100,
    context_length=512,
    parameter_count=117000000
)
""")

    print("3️⃣ Check Downloaded Models:")
    print("""
llm = LocalLLM(config)
models = llm.optimized_llm.list_downloaded_models()
print("Downloaded models:", models)

# Get info about a specific model
info = llm.optimized_llm.get_model_info("microsoft/DialoGPT-small", "mlx")
print(f"Model size: {info['size_mb']} MB")
""")

def main():
    """Run the model download demo."""
    print("🚀 Local Model Download Demo (macOS)")
    print("=" * 40)
    print("This demo shows the new automatic model downloading feature.")
    print("Models are downloaded to ./models/ and cached for future use.")
    print()

    demo_models_directory()
    demo_model_management()
    demo_usage_examples()

    print("✅ Demo completed!")
    print("\n💡 Key Benefits:")
    print("   • Automatic model downloading and caching")
    print("   • Organized storage by backend type")
    print("   • No manual model management required")
    print("   • Optimized for your platform (Apple Silicon)")
    print("\n📂 Check the models/ directory to see downloaded models!")

if __name__ == "__main__":
    main()
