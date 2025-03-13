import tomllib
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "config.toml"

def parse_config(config_path = CONFIG_PATH):
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)
    return config