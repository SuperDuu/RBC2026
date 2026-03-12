"""
Configuration Manager for RBC2026 Robocon Vision System.

This module provides configuration loading and validation functionality.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """Manages application configuration from YAML file."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ConfigManager.
        
        Args:
            config_path: Path to config YAML file. If None, uses default location.
        """
        self.logger = logging.getLogger(__name__)
        
        if config_path is None:
            # Default: config.yaml in parent directory
            base_dir = Path(__file__).resolve().parent.parent
            config_path = str(base_dir / "config.yaml")
        
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            # Resolve relative paths
            self._resolve_paths()
            
            # Validate configuration
            self._validate_config()
            
            self.logger.info(f"Configuration loaded successfully from {self.config_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _resolve_paths(self) -> None:
        """Resolve relative paths to absolute paths."""
        # Base directory is where config.yaml is located (obj_det_v2/)
        base_dir = self.config_path.parent
        
        if 'paths' in self.config:
            paths = self.config['paths']
            
            # Resolve model paths
            if 'models' in paths:
                models = paths['models']
                for key in models:
                    if isinstance(models[key], str):
                        # Check if it's relative path
                        if not Path(models[key]).is_absolute():
                            models[key] = str(base_dir / models[key])
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        required_keys = ['paths', 'hardware', 'models', 'detection']
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Validate paths exist
        if 'paths' in self.config and 'models' in self.config['paths']:
            models = self.config['paths']['models']
            for key, path in models.items():
                if not Path(path).exists():
                    self.logger.warning(f"Model path does not exist: {path}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key path.
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'models.yolo.device')
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"Config key not found: {key_path}")
    
    def get_path(self, key: str) -> str:
        """
        Get model path by key.
        
        Args:
            key: Path key (e.g., 'yolo_xml', 'cnn_xml', 'labels_json')
        
        Returns:
            Absolute path string
        """
        return self.get(f"paths.models.{key}")
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access."""
        return self.config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self.config
