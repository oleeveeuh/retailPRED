
"""
Configuration Manager

This module loads and manages configuration settings from YAML file
and environment variables for the RetailPRED forecasting system.
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

class Config:
    """Configuration manager for RetailPRED system"""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), 'config.yaml')
        self._config = {}
        self._load_config()

    def _load_config(self):
        """TODO: Load configuration from YAML file and environment variables"""
        try:
            # Load YAML configuration
            with open(self.config_path, 'r') as file:
                self._config = yaml.safe_load(file)

            # TODO: Substitute environment variables
            self._substitute_env_vars()

            # TODO: Validate required configuration
            self._validate_config()

        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
            raise

    def _substitute_env_vars(self):
        """Substitute environment variables in configuration"""
        import re

        def substitute_recursive(obj):
            if isinstance(obj, dict):
                return {k: substitute_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_recursive(item) for item in obj]
            elif isinstance(obj, str):
                # Replace ${VAR_NAME} with environment variable
                pattern = r'\$\{([^}]+)\}'
                def replace_var(match):
                    var_name = match.group(1)
                    return os.getenv(var_name, match.group(0))
                return re.sub(pattern, replace_var, obj)
            return obj

        self._config = substitute_recursive(self._config)

    def _validate_config(self):
        """Validate that required configuration keys exist"""
        required_sections = ['database', 'data_sources', 'models', 'targets']

        for section in required_sections:
            if section not in self._config:
                logging.warning(f"Missing required configuration section: {section}")

        # Validate database configuration
        if 'database' in self._config:
            db_config = self._config['database']
            if not db_config.get('sqlite', {}).get('path'):
                logging.warning("Missing SQLite database path configuration")

        # Create necessary directories (only for directories, not files)
        directory_paths = [
            os.path.dirname(self.get('database.sqlite.path', '')),
            self.get('paths.raw_data', ''),
            self.get('paths.processed_data', ''),
            self.get('paths.models', ''),
            self.get('paths.outputs', '')
        ]

        for path in directory_paths:
            if path and path != '.':
                Path(path).mkdir(parents=True, exist_ok=True)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Args:
            key_path: Dot-separated path to configuration key (e.g., 'database.sqlite.path')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation

        Args:
            key_path: Dot-separated path to configuration key
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    def get_database_config(self, db_type: str = 'sqlite') -> Dict[str, Any]:
        """Get database configuration for specified type"""
        return self.get(f'database.{db_type}', {})

    def get_timecopilot_config(self, model_type: str = 'econ') -> Dict[str, Any]:
        """Get TimeCopilot configuration for specified model type"""
        return self.get(f'timecopilot.{model_type}', {})

    def get_data_source_config(self, source: str) -> Dict[str, Any]:
        """Get data source configuration"""
        return self.get(f'data_sources.{source}', {})

    def get_model_config(self, model: str = None) -> Dict[str, Any]:
        """Get model configuration"""
        if model:
            return self.get(f'models.{model}', {})
        return self.get('models', {})

    def get_target_config(self, category: str) -> list:
        """Get target variables for specified category"""
        return self.get(f'targets.{category}', [])

    def get_scheduling_config(self, frequency: str = None) -> list:
        """Get scheduling configuration"""
        if frequency:
            return self.get(f'scheduling.{frequency}', [])
        return self.get('scheduling', {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.get('logging', {})

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return self.get('performance', {})

    def get_notification_config(self, service: str = None) -> Dict[str, Any]:
        """Get notification configuration"""
        if service:
            return self.get(f'notifications.{service}', {})
        return self.get('notifications', {})

    def get_development_config(self) -> Dict[str, Any]:
        """Get development configuration"""
        return self.get('development', {})

    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return self.get('security', {})

    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return self.get('monitoring', {})

    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment configuration"""
        return self.get('environment', {})

    def is_development_mode(self) -> bool:
        """Check if running in development mode"""
        return self.get('development.debug_mode', False)

    def is_production_env(self) -> bool:
        """Check if running in production environment"""
        return self.get('environment.env_type', 'development') == 'production'

    def get_path(self, path_key: str) -> str:
        """Get file path with proper directory handling"""
        path = self.get(f'paths.{path_key}', '')
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)
        return path

    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for specified service"""
        return os.getenv(f'{service.upper()}_API_KEY') or self.get(f'{service}.api_key')

    def validate_api_keys(self) -> Dict[str, bool]:
        """Validate that required API keys are present"""
        required_services = ['fred', 'yahoo', 'snowflake']
        validation_status = {}

        for service in required_services:
            api_key = self.get_api_key(service)
            validation_status[service] = bool(api_key)

        return validation_status

    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        def deep_merge(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_merge(base_dict[key], value)
                else:
                    base_dict[key] = value
            return base_dict

        self._config = deep_merge(self._config, updates)

    def save_config(self, output_path: str = None):
        """Save current configuration to file"""
        output_path = output_path or self.config_path
        with open(output_path, 'w') as file:
            yaml.dump(self._config, file, default_flow_style=False)

    def reload_config(self):
        """Reload configuration from file"""
        self._load_config()

    def get_all_config(self) -> Dict[str, Any]:
        """Get complete configuration dictionary"""
        return self._config.copy()

    def print_config(self, section: str = None):
        """Print configuration section(s)"""
        config_to_print = self.get(section) if section else self._config
        print(yaml.dump(config_to_print, default_flow_style=False))

# Global configuration instance
_config_instance = None

def load_config(config_path: str = None) -> Config:
    """Load and return global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance

def get_config() -> Config:
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance

def reload_config(config_path: str = None):
    """Reload global configuration"""
    global _config_instance
    _config_instance = Config(config_path)

# Configuration shortcuts
def get_database_config(db_type: str = 'sqlite') -> Dict[str, Any]:
    """Get database configuration shortcut"""
    return get_config().get_database_config(db_type)

def get_timecopilot_config(model_type: str = 'econ') -> Dict[str, Any]:
    """Get TimeCopilot configuration shortcut"""
    return get_config().get_timecopilot_config(model_type)

def is_development_mode() -> bool:
    """Check if in development mode shortcut"""
    return get_config().is_development_mode()

def is_production_env() -> bool:
    """Check if in production environment shortcut"""
    return get_config().is_production_env()
