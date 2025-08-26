import logging
import sys
from logging import Logger

_loggers = {}

def get_logger(name: str, config) -> Logger:
    """
    Returns a logger that logs to a timestamped file and to the console.
    """
    if name in _loggers:
        return _loggers[name]

    log_dir = config['path']['run_log_dir']
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"

    logger = logging.getLogger(name)
    # Ignore any messages below level INFO. Log INFO, WARNING, ERROR, and CRITICAL
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    _loggers[name] = logger
    return logger
