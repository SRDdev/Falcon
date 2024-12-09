import yaml
import os

class Config:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        """Load the configuration from a YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get(self, key: str):
        """Retrieve a specific config value."""
        return self.config.get(key)
