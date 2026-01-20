import os
import sys
import json
import argparse
import subprocess
import importlib

# -----------------------------------------------------------------------------
# 1. AUTO-REPAIR & DEPENDENCY CHECK
# -----------------------------------------------------------------------------
def check_environment():
    """Checks for required nightly packages and auto-installs if missing."""
    required = ["ai_edge_torch", "ai_edge_litert", "huggingface_hub", "mediapipe", "torch"]
    missing = []
    
    for pkg in required:
        try:
            importlib.import_module(pkg)
        except ImportError:
            # Handle package name mismatches (underscore vs hyphen)
            pkg_name = pkg.replace("_", "-")
            if "ai-edge" in pkg_name: pkg_name += "-nightly"
            if "mediapipe" in pkg_name: pkg_name += "-nightly" # Critical for GenAI
            missing.append(pkg_name)
    
    if missing:
        print(f"(!) Missing libraries: {', '.join(missing)}")
        print("    Installing now... (This may take a few minutes)")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U"] + missing)
        print("    [+] Environment repaired.")

# -----------------------------------------------------------------------------
# 2. CONFIGURATION DEFAULTS
# -----------------------------------------------------------------------------
ARCH_DEFAULTS = {
    "gemma":  {
        "module": "ai_edge_torch.generative.examples.gemma.gemma1", 
        "start": "<bos>", "stop": ["<eos>", "<end_of_turn>"], "tokens": ["tokenizer.model"]
    },
    "gemma2": {
        "module": "ai_edge_torch.generative.examples.gemma.gemma2", 
        "start": "<bos>", "stop": ["<eos>", "<end_of_turn>"], "tokens": ["tokenizer.model"]
    },
    "llama":  {
        "module": "ai_edge_torch.generative.examples.llama.llama", 
        "start": "<|begin_of_text|>", "stop": ["<|end_of_text|>", "<|eot_id|>"], "tokens": ["tokenizer.json", "tokenizer.model"]
    },
    "phi":    {
        "module": "ai_edge_torch.generative.examples.phi.phi2", 
        "start": "<|endoftext|>", "stop": ["<|endoftext|>"], "tokens": ["vocab.json", "tokenizer.json"]
    },
    "qwen2":  {
        "module": "ai_edge_torch.generative.examples.qwen.qwen", 
        "start": "<|im_start|>", "stop": ["<|im_end|>", "<|endoftext|>"], "tokens": ["tokenizer.json"]
    },
    "qwen3":  {
        "module": "ai_edge_torch.generative.examples.qwen.qwen3", 
        "start": "<|im_start|>", "stop": ["<|im_end|>", "<|endoftext|>"], "tokens": ["tokenizer.json"]
    },
    "tinyllama": {
        "module": "ai_edge_torch.generative.examples.tiny_llama.tiny_llama", 
        "start": "<s>", "stop": ["</s>"], "tokens": ["tokenizer.model"]
    }
}

# -----------------------------------------------------------------------------
# 3. HELPER: SMART BUILDER SELECTION
# -----------------------------------------------------------------------------
def get_model_builder(mod, hf_config):
    """
    Finds the correct build function in the module based on the model configuration.
    Solves the issue where 'build_model' does not exist for multi-size models (Qwen, Gemma).
    """
    # 1. If a generic build_model exists (e.g., TinyLlama), use it.
    if hasattr(mod, "build_model"):
        return mod.build_model

    # Extract key architecture parameters
    hidden_size = hf_config.get("hidden_size") or hf_config.get("d_model") or hf_config.get("n_embd")
    num_layers = hf_config.get("num_hidden_layers") or hf_config.get("num_layers") or hf_config.get("n_layer")

    print(f"    [i] Auto-detecting builder for hidden_size={hidden_size}, layers={num_layers}...")

    # 2. Qwen 2.5 (qwen.py)
    if "qwen" in mod.__name__ and "qwen3" not in mod.__name__:
        if hidden_size == 896: return getattr(mod, "build_0_5b_model", None)
        if hidden_size == 1536: return getattr(mod, "build_1_5b_model", None)
        if hidden_size == 2048: return getattr(mod, "build_3b_model", None)
        if hidden_size == 3584: return getattr(mod, "build_7b_model", None)
        if hidden_size == 5120: return getattr(mod, "build_14b_model", None)
        if hidden_size == 8192: return getattr(mod, "build_32b_model", None)

    # 3. Qwen 3 (qwen3.py)
    elif "qwen3" in mod.__name__:
        if hidden_size == 1024: return getattr(mod, "build_0_6b_model", None)
        if hidden_size == 2048: return getattr(mod, "build_1_7b_model", None)
        if hidden_size == 2560: return getattr(mod, "build_4b_model", None)

    # 4. Gemma 1 (gemma1.py)
    elif "gemma" in mod.__name__ and "gemma2" not in mod.__name__:
        if hidden_size == 2048: return getattr(mod, "build_2b_model", None)
        if hidden_size == 3072: return getattr(mod, "build_7b_model", None)

    # 5. Gemma 2 (gemma2.py)
    elif "gemma2" in mod.__name__:
        if hidden_size == 2304: return getattr(mod, "build_2b_model", None) # 2.6B actually
        if hidden_size == 3584: return getattr(mod, "build_9b_model", None)
        if hidden_size == 4608: return getattr(mod, "build_27b_model", None)

    return None

# -----------------------------------------------------------------------------
# 4. ARGUMENT PARSING
# -----------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face LLMs to LiteRT-LM (.litertlm) for Android/Edge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required / Core
    parser.add_argument("repo_id", nargs="?", default=None, help="Hugging Face Model ID (e.g. google/gemma-2b-it)")
    
    # Quantization
    parser.add_argument("--quant", "-q", choices=["int8", "fp16", "fp32"], default="int8", 
                        help="Quantization mode (int8=smallest, fp32=original)")

    # Model Configuration Overrides
    parser.add_argument("--start-token", help="Override start token (e.g. '<bos>')")
    parser.add_argument("--stop-tokens", nargs="+", help="Override stop tokens (space separated)")
    parser.add_argument("--kv-cache", type=int, default=1280, help="Max context length (KV cache size)")
    parser.add_argument("--prefill", type=int, default=1024, help="Max prefill sequence length")
    
    # Output & Auth
    parser.add_argument("--output", "-o", help="Custom output filename (.litertlm)")
    parser.add_argument("--hf-token", help="Hugging Face API Token for gated models")
    parser.add_argument("--clean", action="store_true", help="Delete intermediate .tflite files after bundling")

    return parser.parse_args()

# -----------------------------------------------------------------------------
# 5. MAIN CONVERSION LOGIC
# -----------------------------------------------------------------------------
def main():
    check_environment()
    
    # Import heavy libs after environment check
    import torch
    from huggingface_hub import snapshot_download, login
    from ai_edge_torch.generative.utilities import converter
    from mediapipe.tasks.python.genai import bundler

    args = parse_arguments()

    # --- Interactive Fallback ---
    if not args.repo_id:
        print("\n=== AI Edge Converter (Interactive Mode) ===")
        print("Tip: Use 'python convert_cli.py <repo_id>' to skip this menu.")
        args.repo_id = input("Enter HF Model ID (e.g. Qwen/Qwen2.5-0.5B-Instruct): ").strip()
        if not args.repo_id: return

    # --- Login ---
    if args.hf_token:
        login(token=args.hf_token)

    # --- Step 1: Download ---
    print(f"\n[1/4] Downloading {args.repo_id}...")
    try:
        model_path = snapshot_download(repo_id=args.repo_id)
    except Exception as e:
        print(f"Error downloading model: {e}")
        return

    # --- Step 2: Configure Architecture ---
    print(f"\n[2/4] Detecting Architecture...")
    
    # A. Detect from config.json
    config_path = os.path.join(model_path, "config.json")
    model_type = "llama" # default fallback
    hf_conf = {}
    
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            hf_conf = json.load(f)
            detected_type = hf_conf.get("model_type", "").lower()
            if detected_type: model_type = detected_type
            
    # B. Normalize names (e.g., qwen2.5 -> qwen2)
    if "qwen2" in model_type: model_type = "qwen2"
    if "qwen3" in model_type: model_type = "qwen3"

    # C. Load Defaults
    defaults = ARCH_DEFAULTS.get(model_type, ARCH_DEFAULTS["llama"])
    
    # D. Apply Args or Defaults
    final_module = defaults["module"]
    final_start = args.start_token if args.start_token else defaults["start"]
    final_stop = args.stop_tokens if args.stop_tokens else defaults["stop"]
    
    print(f"      Detected Type: {model_type}")
    print(f"      Import Path:   {final_module}")
    print(f"      Quantization:  {args.quant.upper()}")

    # --- Step 3: Convert to TFLite ---
    print(f"\n[3/4] Converting to TFLite...")
    tflite_path = "intermediate_model.tflite"
    
    try:
        # Dynamic Import
        mod = importlib.import_module(final_module)
        
        # --- FIX START: Use Smart Builder Selection ---
        build_func = get_model_builder(mod, hf_conf)
        
        if not build_func:
            raise ValueError(f"Could not find a suitable build function in {final_module} for config: {hf_conf.get('architectures', 'Unknown')}")
        
        print(f"      Building model using: {build_func.__name__}")
        model = build_func(model_path)
        # --- FIX END ---
        
        # Quant Config
        q_config = None
        if args.quant == "int8":
            q_config = converter.QuantConfig(embedding_table="int8", input_layer="int8", output_layer="int8", params="int8")
        elif args.quant == "fp16":
            q_config = converter.QuantConfig(embedding_table="float16", input_layer="float16", output_layer="float16", params="float16")
        
        converter.convert_to_tflite(
            model,
            tflite_path,
            quant_config=q_config,
            prefill_seq_len=args.prefill,
            kv_cache_max_len=args.kv_cache
        )
    except Exception as e:
        print(f"\n[Error] Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Step 4: Bundle ---
    print(f"\n[4/4] Bundling...")
    
    # Auto-find tokenizer
    tokenizer_path = None
    search_files = defaults["tokens"] + ["tokenizer.json", "vocab.json"]
    for fname in search_files:
        p = os.path.join(model_path, fname)
        if os.path.exists(p):
            tokenizer_path = p
            break
            
    if not tokenizer_path:
        print(f"Error: Could not find tokenizer. Searched: {search_files}")
        return

    # Determine Output Name
    if args.output:
        final_output = args.output
    else:
        # Auto-name: gemma-2b-it_int8.litertlm
        repo_name = args.repo_id.split("/")[-1]
        final_output = f"{repo_name}_{args.quant}.litertlm"

    try:
        config = bundler.BundleConfig(
            tflite_model=tflite_path,
            tokenizer_model=tokenizer_path,
            start_token=final_start,
            stop_tokens=final_stop,
            output_filename=final_output
        )
        bundler.create_bundle(config)
        
        print(f"\nSUCCESS! Model saved to: {os.path.abspath(final_output)}")
        
        if args.clean and os.path.exists(tflite_path):
            os.remove(tflite_path)
            print("Temporary .tflite file cleaned up.")
            
    except Exception as e:
        print(f"Bundling Error: {e}")

if __name__ == "__main__":
    main()
