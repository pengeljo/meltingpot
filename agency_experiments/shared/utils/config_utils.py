"""
Configuration utilities for agency experiments.
"""

import json
import os
from typing import Any, Dict
from dataclasses import asdict, is_dataclass


def save_experiment_config(config: Any, filepath: str) -> None:
    """
    Save experiment configuration to JSON file.
    
    Args:
        config: Configuration object (dataclass or dict)
        filepath: Path to save configuration file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert config to dictionary
    if is_dataclass(config):
        config_dict = asdict(config)
    elif hasattr(config, 'to_dict'):
        config_dict = config.to_dict()
    elif isinstance(config, dict):
        config_dict = config
    else:
        config_dict = vars(config)
    
    # Save to JSON file
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)


def load_experiment_config(filepath: str) -> Dict[str, Any]:
    """
    Load experiment configuration from JSON file.
    
    Args:
        filepath: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)