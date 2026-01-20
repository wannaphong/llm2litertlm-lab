import os
import sys
import json
import argparse
import subprocess
import importlib

# -----------------------------------------------------------------------------
# 1. SETUP & DEPENDENCIES
# -----------------------------------------------------------------------------
def check_dependencies():
    """Auto-installs missing libraries if needed."""
    required = ["ai_edge_torch", "ai_edge_litert", "huggingface_hub", "mediapipe", "torch"]
    missing = []
    for pkg in required:
        try:
            importlib.import_module(pkg)
        except ImportError:
            pkg_name = pkg.replace("_", "-")
            if "ai-edge" in pkg_name: pkg_name += "-nightly"
            missing.append(pkg_name)
    
    if missing:
        print(f"(!) Missing libraries: {', '.join(missing)}")
        print("    Installing now... (this may take a few minutes)")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U"] + missing)
        print("    Done.\n")

# -----------------------------------------------------------------------------
# 2. CONFIGURATIONS
# -----------------------------------------------------------------------------
# Maps HF architecture types to AI Edge Torch modules and tokens
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
    # ADDED QWEN3 SUPPORT
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

def get_quant_config(mode, converter_lib):
    """Returns the correct quantization config based on user choice."""
    if mode == "int8":
        print("    > Mode: Int8 (Best for Mobile/CPU)")
        return converter_lib.QuantConfig(
            embedding_table="int8", 
            input_layer="int8", 
            output_layer="int8", 
            params="int8"
        )
    elif mode == "fp16":
        print("    > Mode: Float16 (Good for Mobile GPU)")
        return converter_lib.QuantConfig(
            embedding_table="float16", 
            input_layer="float16", 
            output_layer="float16", 
            params="float16"
        )
    else: # fp32
        print("    > Mode: Float32 (No Quantization - Large Size)")
        return None

# -----------------------------------------------------------------------------
# 3. CLI ARGUMENTS
# -----------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Convert HF LLMs to LiteRT-LM (.litertlm)")
    
    # Core
    parser.add_argument("repo_id", nargs="?", help="Hugging Face Model ID (e.g. google/gemma-2b-it)")
    
    # Options
    parser.add_argument("--quant", "-q", choices=["int8", "fp16", "fp32"], help="Quantization mode")
    parser.add_argument("--output", "-o", help="Custom output filename")
    parser.add_argument("--hf-token", help="Hugging Face API Token")
    
    # Advanced
    parser.add_argument("--kv-cache", type=int, default=1280, help="KV Cache size (default: 1280)")
    parser.add_argument("--prefill", type=int, default=1024, help="Prefill length (default: 1024)")
    
    return parser.parse_args()

# -----------------------------------------------------------------------------
# 4. MAIN LOGIC
# -----------------------------------------------------------------------------
def main():
    check_dependencies()
    
    # Imports inside main to avoid crash before install
    import torch
    from huggingface_hub import snapshot_download, login
    from ai_edge_torch.generative.utilities import converter
    from mediapipe.tasks.python.genai import bundler

    args = get_args()

    # --- INTERACTIVE MODE ---
    if not args.repo_id:
        print("\n=== AI Edge Converter ===")
        args.repo_id = input("1. Enter HF Model ID (e.g. google/gemma-2b-it): ").strip()
        if not args.repo_id: return

    if not args.quant:
        print("\n2. Select Quantization:")
        print("   [1] Int8    (Standard - Smallest, 4x smaller)")
        print("   [2] Float16 (Better Precision - 2x smaller)")
        print("   [3] Float32 (Original - Huge file)")
        q_choice = input("   Choice (default 1): ").strip()
        args.quant = {"1": "int8", "2": "fp16", "3": "fp32"}.get(q_choice, "int8")

    # Optional Login
    if args.hf_token: login(token=args.hf_token)

    # --- STEP 1: DOWNLOAD ---
    print(f"\n[1/4] Downloading {args.repo_id}...")
    try:
        model_path = snapshot_download(repo_id=args.repo_id)
    except Exception as e:
        print(f"Error: {e}")
        return

    # --- STEP 2: AUTO-CONFIGURE ---
    print(f"\n[2/4] Detecting Architecture...")
    config_path = os.path.join(model_path, "config.json")
    model_type = "llama" # Fallback
    
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            hf_conf = json.load(f)
            model_type = hf_conf.get("model_type", "llama").lower()
            
    # Check if we have a mapping for this model type
    defaults = ARCH_MAP.get(model_type)
    if not defaults:
        print(f"(!) Warning: Model type '{model_type}' not found in ARCH_MAP. Using 'llama' default.")
        defaults = ARCH_MAP["llama"]
        
    print(f"      Detected: {model_type}")
    print(f"      Importer: {defaults['module']}")

    # --- STEP 3: CONVERT ---
    print(f"\n[3/4] Converting to TFLite...")
    try:
        # Load specific converter module
        mod = importlib.import_module(defaults["module"])
        model = mod.build_model(model_path)
        
        tflite_temp = "temp_model.tflite"
        
        # Get quantization config based on user selection
        q_config = get_quant_config(args.quant, converter)

        converter.convert_to_tflite(
            model,
            tflite_temp,
            quant_config=q_config,
            prefill_seq_len=args.prefill,
            kv_cache_max_len=args.kv_cache
        )
    except ImportError:
        print(f"\n[Error] The module '{defaults['module']}' could not be imported.")
        print("This usually means the 'ai-edge-torch-nightly' version installed does not support this model yet.")
        print("Try upgrading: pip install -U ai-edge-torch-nightly")
        return
    except Exception as e:
        print(f"Conversion Error: {e}")
        return

    # --- STEP 4: BUNDLE ---
    final_filename = args.output if args.output else f"{args.repo_id.split('/')[-1]}_{args.quant}.litertlm"
    print(f"\n[4/4] Bundling into {final_filename}...")
    
    # Locate tokenizer
    tokenizer_path = None
    # We look for files in the order defined in ARCH_MAP + some generic fallbacks
    candidates = defaults["tokens"] + ["tokenizer.json", "vocab.json"]
    
    for cand in candidates:
        p = os.path.join(model_path, cand)
        if os.path.exists(p):
            tokenizer_path = p
            break
            
    if not tokenizer_path:
        print("Error: Tokenizer not found. Checked: " + ", ".join(candidates))
        return

    try:
        bundler.create_bundle(bundler.BundleConfig(
            tflite_model=tflite_temp,
            tokenizer_model=tokenizer_path,
            start_token=defaults["start"],
            stop_tokens=defaults["stop"],
            output_filename=final_filename
        ))
        
        if os.path.exists(tflite_temp): os.remove(tflite_temp)
        print(f"\nSUCCESS! Model Ready: {os.path.abspath(final_filename)}")

    except Exception as e:
        print(f"Bundling Error: {e}")

if __name__ == "__main__":
    main()
