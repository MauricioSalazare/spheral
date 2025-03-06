import json
from spheral.logger import update_log_config, logger

class SpheralConfig:
    """Global configuration class for Spheral package."""

    def __init__(self):
        self._default_config = {
            "log_level": "INFO",
            "log_dir": None,  # If None, defaults to ~/.spheral/logs/
            "log_file": None,  # If None, no log file is created, only console logging
            "parallel_processing": True,
            "threshold": 0.05,
            "max_iterations": 100,
        }
        self._config = self._default_config.copy()

        # Apply default logging configuration
        update_log_config(self._config["log_level"], self._config["log_dir"], self._config["log_file"])

    def set_option(self, key, value):
        """Set a configuration option."""
        if key not in self._config:
            raise KeyError(f"Invalid config key: {key}")
        self._config[key] = value

        # Update logging configuration dynamically
        if key in ["log_level", "log_dir", "log_file"]:
            update_log_config(self._config["log_level"], self._config["log_dir"], self._config["log_file"])

    def get_option(self, key):
        """Get the current value of a configuration option."""
        if key not in self._config:
            raise KeyError(f"Invalid config key: {key}")
        return self._config[key]

    def reset_options(self):
        """Reset all options to their default values."""
        self._config = self._default_config.copy()
        update_log_config(self._config["log_level"], self._config["log_dir"], self._config["log_file"])  # Reset log config

    def load_from_file(self, filepath):
        """Load configuration from a JSON file."""
        try:
            with open(filepath, "r") as f:
                new_config = json.load(f)
            for key, value in new_config.items():
                self.set_option(key, value)
        except FileNotFoundError:
            logger.warning(f"Config file {filepath} not found. Using defaults.")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {filepath}.")

# Create a global instance of the config class
spheral_config = SpheralConfig()
