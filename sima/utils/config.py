# sima/utils/config.py
import json
import os
import torch
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    return config

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to a JSON file"""
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

def default_config() -> Dict[str, Any]:
    """Get default configuration for SIMA"""
    return {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "vision": {
            "model_name": "openai/clip-vit-base-patch16"
        },
        "language": {
            "model_name": "sentence-transformers/all-mpnet-base-v2"
        },
        "action": {
            "embedding_dim": 512,
            "hidden_dim": 1024,
            "keyboard_keys": [
                "w", "a", "s", "d", "q", "e", "r", "f", "space", "shift", 
                "ctrl", "tab", "esc", "enter", "1", "2", "3", "4", "5"
            ]
        },
        "integration": {
            "embedding_dim": 512,
            "num_heads": 8
        },
        "observer": {
            "capture_region": None,  # Use default monitor
            "resize_shape": (224, 224)
        },
        "controller": {
            "action_delay": 0.1
        },
        "visual_change_threshold": 0.15,
        "action_wait_time": 0.5
    }
