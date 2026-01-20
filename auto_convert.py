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
        "module": "ai_edge_torch.gener
