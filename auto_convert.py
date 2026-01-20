import os
import sys
import json
import argparse
import subprocess
import importlib

# -----------------------------------------------------------------------------
# 1. AUTO-REPAIR ENVIRONMENT
# -----------------------------------------------------------------------------
def fix_environment():
    """
    Detects if the wrong MediaPipe version is installed and fixes it.
    The standard 'mediapipe' package is missing 'tasks.cc' needed for GenAI.
    """
    params = [sys.executable, "-m", "pip", "install"]
    
    # Check if we can import the critical GenAI module
    try:
        import mediapipe.tasks.python.genai as genai_test
    except ImportError:
        print("\n[!] Critical Dependency Issue Detected")
        print("    The standard 'mediapipe' package is missing GenAI bindings.")
        print("    Switching to 'mediapipe-nightly'...")
        
        # Uninstall standard mediapipe
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "mediapipe"])
        
        # Install nightly versions
        # We need both ai-edge-torch-nightly and mediapipe-nightly to match
        pkgs = [
            "ai-edge-torch-nightly",
            "ai-edge-litert-nightly", 
            "mediapipe",  # <--- The Fix
            "huggingface_hub"
        ]
        subprocess.check_call(params + ["-U"] + pkgs)
        print("    \n[+] Environment repaired. Please RERUN this script.")
        sys.exit(0)

# -----------------------------------------------------------------------------
# 2. CONFIGURATIONS
# -----------------------------------------------------------------------------
ARCH_MAP = {
    "gemma":  {
        "module": "ai_edge_torch.generative.examples.gemma.gemma1", 
        "start": "<bos>", 
        "stop": ["<eos>", "<end_of_turn>"], 
        "tokens": ["tokenizer.model"]
    },
    "gemma2": {
        "module": "ai_edge_torch.generative.examples.gemma.gemma2", 
        "start": "<bos>", 
        "stop": ["<eos>", "<end_of_turn>"], 
        "tokens": ["tokenizer.model"]
    },
    "llama":  {
        "module": "ai_edge_torch.generative.examples.llama.llama", 
        "start": "<|begin_of_text|>", 
        "stop": ["<|end_of_text|>", "<|eot_id|>"], 
        "tokens": ["tokenizer.json", "tokenizer.model"]
    },
    "phi":    {
        "module": "ai_edge_torch.generative.examples.phi.phi2", 
        "start": "<|endoftext|>", 
        "stop": ["<|endoftext|>"], 
        "tokens": ["vocab.json", "tokenizer.json"]
    },
    "qwen2":  {
        "module": "ai_edge_torch.generative.examples.qwen.qwen", 
        "start": "<|im_start|>", 
        "stop": ["<|im_end|>", "<|endoftext|>"], 
        "tokens": ["tokenizer.json"]
    },
    "qwen3": {
        "module": "ai_edge_torch.generative.examples.qwen.qwen3",
        "start": "<|im_start|>",
        "stop": ["<|im_end|>", "<|endoftext|>"], 
        "tokens": ["tokenizer.json"]
    },
    "tinyllama": {
        "module": "ai_edge_torch.generative.examples.tiny_llama.tiny_llama", 
        "start": "<s>", 
        "stop": ["</s>"], 
        "tokens": ["tokenizer.model"]
    }
}

# -----------------------------------------------------------------------------
# 3. MAIN LOGIC
# -----------------------------------------------------------------------------
def main():
    # Run Repair Check First
    fix_environment()
    
    # Late imports to ensure repair worked
    import torch
    from huggingface_hub import snapshot_download
    from ai_edge_torch.generative.utilities import converter
    from mediapipe.tasks.python.genai import bundler

    print("\n=== AI Edge Converter (Auto-Fix Edition) ===")
    repo_id = input("Enter HF Model ID (e.g. google/gemma-2b-it): ").strip()
    if not repo_id: return

    # 1. Download
    print(f"\n[1/4] Downloading {repo_id}...")
    try:
        model_path = snapshot_download(repo_id=repo_id)
    except Exception as e:
        print(f"Download Error: {e}")
        return

    # 2. Detect
    print(f"\n[2/4] Detecting Architecture...")
    config_path = os.path.join(model_path, "config.json")
    model_type = "llama"
    
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            hf_conf = json.load(f)
            model_type = hf_conf.get("model_type", "llama").lower()

    # Map 'qwen2.5' or others to base keys if needed
    if "qwen2" in model_type: model_type = "qwen2"
    
    conf = ARCH_MAP.get(model_type, ARCH_MAP["llama"])
    print(f"      Detected: {model_type}")

    # 3. Convert
    print(f"\n[3/4] Converting to TFLite (Int8)...")
    try:
        mod = importlib.import_module(conf["module"])
        model = mod.build_model(model_path)
        
        tflite_path = "temp.tflite"
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
        print(f"Conversion Error: {e}")
        return

    # 4. Bundle
    print(f"\n[4/4] Bundling...")
    token_path = None
    for t in conf["tokens"] + ["tokenizer.json", "vocab.json"]:
        p = os.path.join(model_path, t)
        if os.path.exists(p):
            token_path = p
            break
            
    if not token_path:
        print("Error: Tokenizer file not found.")
        return

    output_file = f"{repo_id.split('/')[-1]}.litertlm"
    try:
        config = bundler.BundleConfig(
            tflite_model=tflite_path,
            tokenizer_model=token_path,
            start_token=conf["start"],
            stop_tokens=conf["stop"],
            output_filename=output_file
        )
        bundler.create_bundle(config)
        print(f"\nSUCCESS! File ready: {os.path.abspath(output_file)}")
        if os.path.exists(tflite_path): os.remove(tflite_path)
    except Exception as e:
        print(f"Bundling Error: {e}")

if __name__ == "__main__":
    main()
