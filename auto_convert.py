import os
import json
import subprocess
import sys

# --- 1. Auto-Install Dependencies ---
def install_dependencies():
    print("Checking and installing required libraries... (This may take a moment)")
    packages = ["ai-edge-torch-nightly", "ai-edge-litert-nightly", "huggingface_hub", "mediapipe"]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U"] + packages)

try:
    import torch
    from huggingface_hub import snapshot_download
    from mediapipe.tasks.python.genai import bundler
except ImportError:
    install_dependencies()
    import torch
    from huggingface_hub import snapshot_download
    from mediapipe.tasks.python.genai import bundler

# --- 2. Configuration Mappings ---
# This maps the 'model_type' from HF config.json to the required settings
ARCH_CONFIG = {
    "gemma": {
        "import_path": "ai_edge_torch.generative.examples.gemma.gemma1",
        "start_token": "<bos>",
        "stop_tokens": ["<eos>", "<end_of_turn>"],
        "tokenizer_files": ["tokenizer.model"]
    },
    "gemma2": {
        "import_path": "ai_edge_torch.generative.examples.gemma.gemma2",
        "start_token": "<bos>",
        "stop_tokens": ["<eos>", "<end_of_turn>"],
        "tokenizer_files": ["tokenizer.model"]
    },
    "llama": {
        "import_path": "ai_edge_torch.generative.examples.llama.llama",
        "start_token": "<|begin_of_text|>",
        "stop_tokens": ["<|end_of_text|>", "<|eot_id|>"],
        "tokenizer_files": ["tokenizer.json", "tokenizer.model"]
    },
    "phi": {
        "import_path": "ai_edge_torch.generative.examples.phi.phi2",
        "start_token": "<|endoftext|>",
        "stop_tokens": ["<|endoftext|>"],
        "tokenizer_files": ["vocab.json", "tokenizer.json"]
    },
    # Qwen typically uses Llama-like architecture or its own depending on version
    "qwen2": {
        "import_path": "ai_edge_torch.generative.examples.qwen.qwen",
        "start_token": "<|im_start|>",
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
        "tokenizer_files": ["tokenizer.json"]
    },
    "qwen3": {
        "import_path": "ai_edge_torch.generative.examples.qwen.qwen3",
        "start_token": "<|im_start|>",
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
        "tokenizer_files": ["tokenizer.json"]
    }
}

# --- 3. Main Logic ---
def main():
    print("\n=== Auto-Detect AI Edge Converter ===")
    print("Enter the Hugging Face Model ID (e.g., google/gemma-2b-it):")
    repo_id = input("> ").strip()
    if not repo_id: return

    # A. Download Model
    print(f"\n[Step 1] Downloading {repo_id}...")
    try:
        model_path = snapshot_download(repo_id=repo_id)
    except Exception as e:
        print(f"Error downloading model: {e}")
        return

    # B. Auto-Detect Architecture
    print(f"\n[Step 2] Detecting Architecture...")
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        print("Error: Could not find config.json in model folder.")
        return

    with open(config_path, "r") as f:
        hf_config = json.load(f)
    
    # Read architecture type
    model_type = hf_config.get("model_type", "").lower()
    print(f"Detected model type: {model_type}")

    # Match to our supported list
    settings = ARCH_CONFIG.get(model_type)
    
    # Fallback/Error handling
    if not settings:
        print(f"Warning: Model type '{model_type}' is not explicitly mapped.")
        print("Attempting generic mapping (Llama-style fallback)...")
        settings = ARCH_CONFIG["llama"]
    
    print(f"Using start token: {settings['start_token']}")
    print(f"Using stop tokens: {settings['stop_tokens']}")

    # C. Convert
    print(f"\n[Step 3] Converting to LiteRT (Int8)...")
    from ai_edge_torch.generative.utilities import converter
    import importlib

    try:
        module = importlib.import_module(settings['import_path'])
        model = module.build_model(model_path)
        
        tflite_path = "temp_model.tflite"
        output_path = f"{repo_id.split('/')[-1]}.litertlm"

        converter.convert_to_tflite(
            model,
            tflite_path,
            quant_config=converter.QuantConfig(
                embedding_table="int8", input_layer="int8", output_layer="int8", params="int8"
            ),
            prefill_seq_len=1024,
            kv_cache_max_len=1280
        )
    except Exception as e:
        print(f"Conversion failed: {e}")
        return

    # D. Bundle
    print(f"\n[Step 4] Finalizing Bundle...")
    
    # Find tokenizer
    token_path = None
    for fname in settings['tokenizer_files']:
        test_path = os.path.join(model_path, fname)
        if os.path.exists(test_path):
            token_path = test_path
            print(f"Found tokenizer file: {fname}")
            break
    
    if not token_path:
        # Last resort fallback
        if os.path.exists(os.path.join(model_path, "tokenizer.json")):
            token_path = os.path.join(model_path, "tokenizer.json")
        elif os.path.exists(os.path.join(model_path, "vocab.json")):
            token_path = os.path.join(model_path, "vocab.json")
        else:
            print("Error: Could not find any tokenizer file.")
            return

    try:
        bundle_config = bundler.BundleConfig(
            tflite_model=tflite_path,
            tokenizer_model=token_path,
            start_token=settings['start_token'],
            stop_tokens=settings['stop_tokens'],
            output_filename=output_path
        )
        bundler.create_bundle(bundle_config)
        
        # Cleanup
        if os.path.exists(tflite_path): os.remove(tflite_path)
        print(f"\nSUCCESS! File ready: {os.path.abspath(output_path)}")
        
    except Exception as e:
        print(f"Bundling failed: {e}")

if __name__ == "__main__":
    main()
