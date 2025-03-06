import logging
import os
import sys
import functools
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Default log directory (overridden by config if set)
DEFAULT_LOG_DIR = Path.home() / ".spheral" / "logs"

# Create a single logger instance (NO reassignment)
logger = logging.getLogger("spheral")
logger.setLevel(logging.INFO)  # Default level
logger.handlers.clear()  # Ensure no duplicate handlers

# Formatter for all handlers
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Console handler (always active)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def update_log_config(level_name: str, log_dir: str, log_file: str):
    """
    Update the logging configuration dynamically without replacing the logger instance.
    """
    level = getattr(logging, level_name.upper(), logging.INFO)
    logger.setLevel(level)

    # Remove existing file handlers
    logger.handlers = [h for h in logger.handlers if not isinstance(h, RotatingFileHandler)]

    if log_file:
        # Determine log directory
        log_path = Path(log_dir) / log_file if log_dir else DEFAULT_LOG_DIR / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure log directory exists

        # Create a rotating file handler
        file_handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=1)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)

        # Add new file handler
        logger.addHandler(file_handler)

    logger.info(f"Logging updated - Level: {level_name}, Dir: {log_dir}, File: {log_file}")


def log_function_call(func):
    """
    Decorator to automatically log function calls.
    Logs when the function starts and finishes.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Calling function: {func.__name__}")
        result = func(*args, **kwargs)
        logger.info(f"Function {func.__name__} finished execution")
        return result

    return wrapper
