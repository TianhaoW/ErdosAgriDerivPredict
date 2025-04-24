import tomllib
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "config.toml"

def parse_config(config_path = CONFIG_PATH):
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)

    # Resolve project_root relative to the config file location
    raw_root = config["path"]["project_root"]
    project_root = (CONFIG_PATH.parent / raw_root).resolve()

    config["path"]["project_root"] = project_root

    return config