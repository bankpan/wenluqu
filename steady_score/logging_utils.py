from __future__ import annotations

import logging
from pathlib import Path


def setup_file_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "processing.log"
    logger = logging.getLogger("steady_score")
    logger.setLevel(logging.INFO)

    if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_file)
               for handler in logger.handlers):
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
