"""Logging helpers."""

import os
import logging
import sys
from typing import Optional


def configure_logging(level: str = "INFO", logger_name: Optional[str] = None) -> logging.Logger:
    """Configure a stream logger once and return it."""
    logger = logging.getLogger(logger_name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    if not logger.handlers:
        rank = os.environ.get("RANK", "0")
        world_size = os.environ.get("WORLD_SIZE", "1")
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                f"%(asctime)s | rank={rank}/{world_size} | %(levelname)s | %(name)s | %(message)s"
            )
        )
        logger.addHandler(handler)
    logger.propagate = False
    return logger
