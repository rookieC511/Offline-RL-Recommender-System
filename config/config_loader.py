"""
Configuration loader and validator for RL Crowdsourcing System
Provides utilities to load, validate, and override configurations.
"""

import os
import json
import argparse
from typing import Dict, Any, Optional
from dataclasses import dataclass

from . import base_config
from . import cql_config
from . import dpo_config
from . import ddqn_config


@dataclass
class ConfigValidationError(Exception):
    """Raised when configuration validation fails"""
    message: str
    config_key: str


class ConfigLoader:
    """
    Centralized configuration loader with validation and override capabilities.
    """
    
    SUPPORTED_ALGORITHMS = {
        'cql': cql_config,
        'dpo': dpo_config,
        'ddqn': ddqn_config,
    }
    
    def __init__(self):
        self.base_config = base_config
        self.current_config = None
        self.algorithm = None
    
    def load_config(self, algorithm: str, config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load configuration for specified algorithm with optional overrides.
        
        Args:
            algorithm: Algorithm name ('cql', 'dpo', 'ddqn')
            config_overrides: Dictionary of configuration overrides
            
        Returns:
            Complete configuration dictionary
            
        Raises:
            ConfigValidationError: If algorithm not supported or validation fails
        """
        if algorithm.lower() not in self.SUPPORTED_ALGORITHMS:
            raise ConfigValidationError(
                f"Unsupported algorithm: {algorithm}. Supported: {list(self.SUPPORTED_ALGORITHMS.keys())}",
                "algorithm"
            )
        
        self.algorithm = algorithm.lower()
        algorithm_config = self.SUPPORTED_ALGORITHMS[self.algorithm]
        
        # Start with base configuration
        config = self._extract_config_dict(self.base_config)
        
        # Override with algorithm-specific configuration
        algorithm_specific = self._extract_config_dict(algorithm_config)
        config.update(algorithm_specific)
        
        # Apply user overrides
        if config_overrides:
            config = self._apply_overrides(config, config_overrides)
        
        # Validate configuration
        self._validate_config(config)
        
        self.current_config = config
        return config
    
    def load_from_file(self, config_file: str, algorithm: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Args:
            config_file: Path to JSON configuration file
            algorithm: Algorithm name
            
        Returns:
            Complete configuration dictionary
        """
        if not os.path.exists(config_file):
            raise ConfigValidationError(f"Configuration file not found: {config_file}", "file_path")
        
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigValidationError(f"Invalid JSON in config file: {e}", "json_format")
        
        return self.load_config(algorithm, file_config)
    
    def load_from_args(self, args: argparse.Namespace) -> Dict[str, Any]:
        """
        Load configuration from command line arguments.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Complete configuration dictionary
        """
        # Convert argparse Namespace to dictionary, filtering None values
        overrides = {k: v for k, v in vars(args).items() if v is not None and k != 'algorithm'}
        
        return self.load_config(args.algorithm, overrides)
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with optional default.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'cql_config.alpha')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if self.current_config is None:
            raise ConfigValidationError("No configuration loaded. Call load_config() first.", "no_config")
        
        # Support dot notation for nested keys
        keys = key.split('.')
        value = self.current_config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def _extract_config_dict(self, config_module) -> Dict[str, Any]:
        """Extract configuration dictionary from module."""
        config_dict = {}
        
        for attr_name in dir(config_module):
            if not attr_name.startswith('_'):  # Skip private attributes
                attr_value = getattr(config_module, attr_name)
                
                # Include constants, dictionaries, and basic types
                if isinstance(attr_value, (int, float, str, bool, dict, list, tuple)):
                    config_dict[attr_name] = attr_value
        
        return config_dict
    
    def _apply_overrides(self, config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Apply configuration overrides with deep merge."""
        result = config.copy()
        
        for key, value in overrides.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Deep merge for nested dictionaries
                result[key] = {**result[key], **value}
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration values."""
        # Validate required keys
        required_keys = ['ALGORITHM_NAME', 'DEVICE', 'DATA_PATHS']
        for key in required_keys:
            if key not in config:
                raise ConfigValidationError(f"Required configuration key missing: {key}", key)
        
        # Validate algorithm-specific requirements
        algorithm = config['ALGORITHM_NAME'].lower()
        
        if algorithm == 'cql':
            self._validate_cql_config(config)
        elif algorithm == 'dpo':
            self._validate_dpo_config(config)
        elif algorithm == 'ddqn':
            self._validate_ddqn_config(config)
    
    def _validate_cql_config(self, config: Dict[str, Any]) -> None:
        """Validate CQL-specific configuration."""
        if 'CQL_ALPHA' not in config:
            raise ConfigValidationError("CQL_ALPHA required for CQL algorithm", "CQL_ALPHA")
        
        if not 0 < config['CQL_ALPHA'] < 10:
            raise ConfigValidationError("CQL_ALPHA should be between 0 and 10", "CQL_ALPHA")
    
    def _validate_dpo_config(self, config: Dict[str, Any]) -> None:
        """Validate DPO-specific configuration."""
        if 'DPO_BETA' not in config:
            raise ConfigValidationError("DPO_BETA required for DPO algorithm", "DPO_BETA")
        
        if not 0 < config['DPO_BETA'] < 5:
            raise ConfigValidationError("DPO_BETA should be between 0 and 5", "DPO_BETA")
    
    def _validate_ddqn_config(self, config: Dict[str, Any]) -> None:
        """Validate DDQN-specific configuration."""
        if 'GAMMA' not in config:
            raise ConfigValidationError("GAMMA required for DDQN algorithm", "GAMMA")
        
        if not 0 < config['GAMMA'] <= 1:
            raise ConfigValidationError("GAMMA should be between 0 and 1", "GAMMA")
    
    def save_config(self, filepath: str) -> None:
        """Save current configuration to JSON file."""
        if self.current_config is None:
            raise ConfigValidationError("No configuration loaded to save", "no_config")
        
        # Convert non-serializable objects to strings
        serializable_config = self._make_serializable(self.current_config)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_config, f, indent=2)
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert configuration to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return str(obj)  # Convert complex objects to string
        else:
            return obj


# Convenience functions
def load_config(algorithm: str, config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function to load configuration."""
    loader = ConfigLoader()
    return loader.load_config(algorithm, config_overrides)


def create_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser with common configuration options."""
    parser = argparse.ArgumentParser(description='RL Crowdsourcing Training')
    
    parser.add_argument('--algorithm', type=str, required=True, 
                       choices=['cql', 'dpo', 'ddqn'],
                       help='RL algorithm to use')
    
    parser.add_argument('--config-file', type=str,
                       help='Path to JSON configuration file')
    
    # Common parameters
    parser.add_argument('--learning-rate', type=float,
                       help='Learning rate override')
    
    parser.add_argument('--batch-size', type=int,
                       help='Batch size override')
    
    parser.add_argument('--num-epochs', type=int,
                       help='Number of training epochs')
    
    # Algorithm-specific parameters
    parser.add_argument('--cql-alpha', type=float,
                       help='CQL alpha parameter')
    
    parser.add_argument('--dpo-beta', type=float,
                       help='DPO beta parameter')
    
    parser.add_argument('--gamma', type=float,
                       help='Discount factor for Q-learning algorithms')
    
    return parser


# Example usage
if __name__ == "__main__":
    # Example 1: Load CQL configuration
    loader = ConfigLoader()
    cql_config = loader.load_config('cql')
    print(f"CQL Alpha: {loader.get_config_value('CQL_ALPHA')}")
    
    # Example 2: Load with overrides
    overrides = {'CQL_ALPHA': 0.2, 'LEARNING_RATE': 2e-5}
    cql_config_modified = loader.load_config('cql', overrides)
    
    # Example 3: Command line usage
    parser = create_arg_parser()
    args = parser.parse_args(['--algorithm', 'cql', '--cql-alpha', '0.1'])
    config = loader.load_from_args(args)

