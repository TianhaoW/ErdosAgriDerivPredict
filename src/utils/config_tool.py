import tomllib
from pathlib import Path
import os
from dotenv import load_dotenv

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"

def parse_config(config_dir=CONFIG_DIR):
    """
    Parses all .toml configuration files in the specified directory,
    merges them, and resolves relative paths. Also loads API keys
    from a .env file.
    """
    # Load API keys from .env file
    load_dotenv(dotenv_path=config_dir.parent / ".env")
    
    config = {
        "api": {
            "USDA_api_key": os.getenv("USDA_api_key"),
            "NCDC_api_key": os.getenv("NCDC_api_key")
        }
    }

    # Load all .toml files from the config directory
    for config_file in config_dir.glob("*.toml"):
        with open(config_file, 'rb') as f:
            data = tomllib.load(f)
            # Deep merge the loaded data into the main config dictionary
            for key, value in data.items():
                if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                    config[key].update(value)
                else:
                    config[key] = value

    # Resolve paths relative to the config directory's parent (project root)
    project_root_str = config.get("path", {}).get("project_root", "..")
    project_root = (config_dir.parent / project_root_str).resolve()
    config["path"]["project_root"] = project_root

    if "data" in config and "raw_cdl_path" in config["data"]:
        cdl_path_str = config["data"]["raw_cdl_path"]
        config['data']['raw_cdl_path'] = (config_dir.parent / cdl_path_str).resolve()

    if "data" in config and "us_map_path" in config["data"]:
        us_map_path_str = config["data"]["us_map_path"]
        config['data']['us_map_path'] = (config_dir.parent / us_map_path_str).resolve()

    if "path" in config and "run_log_dir" in config["path"]:
        run_log_dir_str = config["path"]["run_log_dir"]
        config['path']['run_log_dir'] = (config_dir.parent / run_log_dir_str).resolve()

    return config
