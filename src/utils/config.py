import os
import yaml
from typing import Dict, Any

class Config:
    """Configuration handler for the project."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Dictionary containing configuration parameters
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            try:
                config = yaml.safe_load(f)
                return config
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing config file: {e}")
    
    @staticmethod
    def save_config(config: Dict[str, Any], save_path: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary to save
            save_path: Path where to save the configuration
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    @staticmethod
    def get_default_device() -> str:
        """Get default device for training/evaluation.
        
        Returns:
            String indicating device (cuda:X or cpu)
        """
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @staticmethod
    def setup_environment(config: Dict[str, Any]) -> None:
        """Setup environment variables based on configuration.
        
        Args:
            config: Configuration dictionary
        """
        # Set CUDA devices if specified
        if "gpu_devices" in config.get("training", {}):
            os.environ["CUDA_VISIBLE_DEVICES"] = config["training"]["gpu_devices"]
        elif "gpu_device" in config.get("model", {}):
            os.environ["CUDA_VISIBLE_DEVICES"] = config["model"]["gpu_device"]
            
        # Set random seed if specified
        if "seed" in config.get("training", {}):
            import torch
            import numpy as np
            import random
            
            seed = config["training"]["seed"]
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)